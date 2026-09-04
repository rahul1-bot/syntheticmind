from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from typing import cast, override

import torch
from torch import nn
from torch.distributed.constants import default_pg_timeout
from torch.nn.parallel import DistributedDataParallel

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.core.module import Module
from syntheticmind.plugins.environments import ClusterEnvironment
from syntheticmind.plugins.io import CheckpointIO
from syntheticmind.plugins.layer_sync import LayerSync
from syntheticmind.precision.precision import Precision
from syntheticmind.strategies.checkpoint_connector import (
    StrategyCheckpoint,
    _CheckpointConnector
)
from syntheticmind.strategies.distributed_runtime import _DistributedRuntime
from syntheticmind.strategies.forward_redirection import (
    StepName,
    _DDPForwardRedirection,
    _ForwardRedirection
)
from syntheticmind.strategies.launcher import ProcessStartMethod, _ProcessLauncher
from syntheticmind.strategies.parallel import ParallelStrategy
from syntheticmind.utilities.model_initialization import _ModuleInitializer
from syntheticmind.utilities.types import (
    BackwardInputs,
    Batch,
    ModelOutput,
    OptimizerClosure,
    OptimizerStepOutput,
    OptimizerStepValue,
    StepArgument,
    StepOutput
)

__all__: list[str] = ["DDPStrategy"]


class DDPStrategy(ParallelStrategy):
    def __init__(
        self,
        accelerator: Accelerator | None = None,
        parallel_devices: list[torch.device] | None = None,
        cluster_environment: ClusterEnvironment | None = None,
        checkpoint_io: CheckpointIO | None = None,
        precision_plugin: Precision | None = None,
        process_group_backend: str | None = None,
        timeout: timedelta | None = default_pg_timeout,
        start_method: ProcessStartMethod = "popen",
        layer_sync: LayerSync | None = None
    ) -> None:
        self.accelerator: Accelerator | None = accelerator
        self.parallel_devices: list[torch.device] = list(parallel_devices or [])
        self.num_nodes: int = 1
        self.is_distributed: bool = True
        self._module: Module | None = None
        self.model: nn.Module | None = None
        self.optimizers: list[torch.optim.Optimizer] = []
        self.checkpoint: StrategyCheckpoint = _CheckpointConnector(checkpoint_io)
        self.precision_plugin: Precision = precision_plugin or Precision()
        self.configured_precision_plugin: Precision | None = precision_plugin
        self._layer_sync: LayerSync | None = layer_sync
        self._configured_layer_sync: LayerSync | None = layer_sync
        self._forward_redirection: _ForwardRedirection = _DDPForwardRedirection()
        self._runtime: _DistributedRuntime = _DistributedRuntime(
            self,
            cluster_environment,
            process_group_backend,
            timeout,
            start_method
        )

    @override
    def configure_topology(
        self, accelerator: Accelerator, parallel_devices: list[torch.device],
        num_nodes: int
    ) -> None:
        if not parallel_devices or num_nodes < 1:
            raise ValueError("Distributed topology requires devices and positive nodes.")
        self.accelerator: Accelerator | None = accelerator
        self.parallel_devices: list[torch.device] = list(parallel_devices)
        self.num_nodes: int = num_nodes
        self._runtime.configure_environment()

    @override
    def connect(self, model: nn.Module) -> None:
        if not isinstance(model, Module):
            raise TypeError("DDPStrategy requires a SyntheticMind Module.")
        self._module: Module | None = model
        self.model: nn.Module | None = model
        self.checkpoint.connect_model(model)

    @override
    def setup_environment(self) -> None:
        self._runtime.setup_environment()

    @override
    def setup(self, model: nn.Module, device: torch.device) -> nn.Module:
        placed_model: nn.Module = model.to(device)
        wrapped_model: DistributedDataParallel = DistributedDataParallel(
            placed_model,
            device_ids=[device.index] if device.type == "cuda" else None
        )
        self.model: nn.Module | None = wrapped_model
        return wrapped_model

    @override
    def training_step(
        self, model: nn.Module, batch: Batch, batch_idx: int
    ) -> StepOutput:
        return cast(StepOutput, self._execute_step(model, "training_step", batch, batch_idx))

    @override
    def validation_step(
        self, model: nn.Module, *args: StepArgument, **kwargs: StepArgument
    ) -> StepOutput:
        return cast(StepOutput, self._execute_step(model, "validation_step", *args, **kwargs))

    @override
    def test_step(
        self, model: nn.Module, *args: StepArgument, **kwargs: StepArgument
    ) -> StepOutput:
        return cast(StepOutput, self._execute_step(model, "test_step", *args, **kwargs))

    @override
    def predict_step(
        self, model: nn.Module, *args: StepArgument, **kwargs: StepArgument
    ) -> ModelOutput:
        return cast(ModelOutput, self._execute_step(model, "predict_step", *args, **kwargs))

    @override
    def backward(
        self, closure_loss: torch.Tensor,
        optimizer: torch.optim.Optimizer | None = None,
        gradient: torch.Tensor | None = None,
        retain_graph: bool | None = None,
        create_graph: bool = False,
        inputs: BackwardInputs = None
    ) -> torch.Tensor:
        module: Module = self._require_module()
        prepared_loss: torch.Tensor = self.precision_plugin.pre_backward(
            closure_loss,
            module
        )
        self.precision_plugin.backward(
            prepared_loss,
            module,
            optimizer,
            gradient,
            retain_graph,
            create_graph,
            inputs
        )
        return self.precision_plugin.post_backward(prepared_loss, module)

    @override
    def optimizer_step(
        self, optimizer: torch.optim.Optimizer,
        closure: OptimizerClosure,
        model: nn.Module | None = None,
        **kwargs: OptimizerStepValue
    ) -> OptimizerStepOutput:
        resolved_model: nn.Module | None = model or self._module
        if not isinstance(resolved_model, Module):
            raise TypeError("Strategy optimizer_step requires a SyntheticMind Module.")
        return cast(
            OptimizerStepOutput,
            self.precision_plugin.optimizer_step(
                optimizer,
                model=resolved_model,
                closure=closure,
                **kwargs
            )
        )

    @override
    def teardown(self) -> None:
        if isinstance(self.model, DistributedDataParallel):
            self.model: nn.Module | None = self.model.module

    @override
    def on_exception(self, exception: BaseException) -> None:
        del exception
        self._runtime.on_exception()

    @contextmanager
    @override
    def tensor_init_context(self, empty_init: bool | None = None) -> Iterator[None]:
        if empty_init is not None and type(empty_init) is not bool:
            raise TypeError("`empty_init` must be a bool or None.")
        with (
            _ModuleInitializer(enabled=bool(empty_init)),
            self.root_device,
            self.precision_plugin.tensor_init_context()
        ):
            yield

    @override
    def barrier(self) -> None:
        self._runtime.barrier()

    @override
    def broadcast[T](self, value: T, source_rank: int = 0) -> T:
        return self._runtime.broadcast(value, source_rank)

    @override
    def reduce_boolean_decision(self, decision: bool, all: bool = True) -> bool:
        return self._runtime.reduce_boolean_decision(decision, all)

    @override
    def all_gather(
        self, tensor: torch.Tensor,
        group: torch.distributed.ProcessGroup | None = None,
        sync_grads: bool = False
    ) -> torch.Tensor:
        return self._runtime.all_gather(tensor, group, sync_grads)

    @override
    def sync_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        self._runtime.sync_gradients(optimizer)

    @override
    def attach_precision_plugin(self, precision_plugin: Precision) -> None:
        if not isinstance(precision_plugin, Precision):
            raise TypeError("precision_plugin must be a Precision instance.")
        self.precision_plugin: Precision = precision_plugin

    @override
    def attach_cluster_environment(self, cluster_environment: ClusterEnvironment) -> None:
        self._runtime.attach_cluster_environment(cluster_environment)

    def attach_layer_sync(self, layer_sync: LayerSync | None) -> None:
        if layer_sync is not None and not isinstance(layer_sync, LayerSync):
            raise TypeError("layer_sync must be a LayerSync instance or None.")
        self._layer_sync: LayerSync | None = layer_sync

    @property
    @override
    def distributed_sampler_kwargs(self) -> dict[str, int] | None:
        return {"num_replicas": self.world_size, "rank": self.global_rank}

    @property
    @override
    def launcher(self) -> _ProcessLauncher | None:
        return self._runtime.launcher

    @property
    @override
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices else 1

    @property
    @override
    def global_rank(self) -> int:
        return self._runtime.global_rank

    @property
    @override
    def local_rank(self) -> int:
        return self._runtime.local_rank

    @property
    @override
    def node_rank(self) -> int:
        return self._runtime.node_rank

    @property
    @override
    def world_size(self) -> int:
        return self._runtime.world_size

    @property
    @override
    def is_global_zero(self) -> bool:
        return self.global_rank == 0

    @property
    @override
    def root_device(self) -> torch.device:
        return self._runtime.root_device

    @property
    @override
    def cluster_environment(self) -> ClusterEnvironment:
        return self._runtime.cluster_environment

    @property
    @override
    def configured_cluster_environment(self) -> ClusterEnvironment | None:
        return self._runtime.configured_cluster_environment

    @property
    def layer_sync(self) -> LayerSync | None:
        return self._layer_sync

    @property
    def configured_layer_sync(self) -> LayerSync | None:
        return self._configured_layer_sync

    def _execute_step(
        self, model: nn.Module, step_name: StepName,
        *args: StepArgument,
        **kwargs: StepArgument
    ) -> ModelOutput:
        return self._forward_redirection.execute(
            model,
            self._require_module(),
            step_name,
            *args,
            **kwargs
        )

    def _require_module(self) -> Module:
        if self._module is None:
            raise RuntimeError("Strategy has no connected Module.")
        return self._module
