import warnings
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from datetime import timedelta
from typing import Literal, cast, override

import torch
from torch import nn
from torch.distributed.constants import default_pg_timeout
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.core.hooks import ModelHooks
from syntheticmind.core.module import Module
from syntheticmind.plugins.environments import ClusterEnvironment
from syntheticmind.precision.precision import Precision
from syntheticmind.strategies.checkpoint_connector import StrategyCheckpoint
from syntheticmind.strategies.distributed_runtime import _DistributedRuntime
from syntheticmind.strategies.forward_redirection import StepName, _ForwardRedirection
from syntheticmind.strategies.launcher import _ProcessLauncher
from syntheticmind.strategies.model_parallel_checkpointing.checkpoint import (
    _ModelParallelCheckpoint
)
from syntheticmind.strategies.model_parallel_topology import _ModelParallelTopology
from syntheticmind.strategies.parallel import ParallelStrategy
from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.model_helpers import is_overridden
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

__all__: list[str] = ["ModelParallelStrategy"]


class ModelParallelStrategy(ParallelStrategy):
    def __init__(
        self,
        data_parallel_size: Literal["auto"] | int = "auto",
        tensor_parallel_size: Literal["auto"] | int = "auto",
        save_distributed_checkpoint: bool = True,
        process_group_backend: str | None = None,
        timeout: timedelta | None = default_pg_timeout
    ) -> None:
        if type(save_distributed_checkpoint) is not bool:
            raise TypeError("`save_distributed_checkpoint` must be a bool.")
        self.accelerator: Accelerator | None = None
        self.parallel_devices: list[torch.device] = []
        self.num_nodes: int = 1
        self.is_distributed: bool = True
        self.model: nn.Module | None = None
        self.optimizers: list[torch.optim.Optimizer] = []
        self.precision_plugin: Precision = Precision()
        self.configured_precision_plugin: Precision | None = None
        self._module: Module | None = None
        self._forward_redirection: _ForwardRedirection = _ForwardRedirection()
        self._runtime: _DistributedRuntime = _DistributedRuntime(
            self,
            None,
            process_group_backend,
            timeout,
            "popen"
        )
        self._topology: _ModelParallelTopology = _ModelParallelTopology(
            data_parallel_size,
            tensor_parallel_size
        )
        self.checkpoint: StrategyCheckpoint = _ModelParallelCheckpoint(
            "sharded" if save_distributed_checkpoint else "full",
            self.barrier,
            self.broadcast
        )

    @override
    def configure_topology(
        self, accelerator: Accelerator, parallel_devices: list[torch.device],
        num_nodes: int
    ) -> None:
        if not parallel_devices or num_nodes < 1:
            raise ValueError("Model-parallel topology requires devices and positive nodes.")
        self.accelerator: Accelerator | None = accelerator
        self.parallel_devices: list[torch.device] = list(parallel_devices)
        self.num_nodes: int = num_nodes
        self._runtime.configure_environment()

    @override
    def connect(self, model: nn.Module) -> None:
        if not isinstance(model, Module):
            raise TypeError("ModelParallelStrategy requires a SyntheticMind Module.")
        self._module: Module | None = model
        self.model: nn.Module | None = model
        self._topology.connect(model)
        self.checkpoint.connect_model(model)

    @override
    def setup_environment(self) -> None:
        self._runtime.setup_environment()
        self._topology.setup(
            self.world_size,
            self.num_nodes,
            self.num_processes,
            self.root_device.type
        )

    @override
    def setup(self, model: nn.Module, device: torch.device) -> nn.Module:
        if not isinstance(model, Module):
            raise TypeError("ModelParallelStrategy requires a SyntheticMind Module.")
        if not is_overridden("configure_model", model, ModelHooks):
            raise TypeError(
                "ModelParallelStrategy requires Module.configure_model() to apply "
                "its parallelization plan."
            )
        submodule: nn.Module
        for submodule in model.modules():
            if isinstance(submodule, FullyShardedDataParallel):
                raise TypeError("ModelParallelStrategy supports FSDP2, not FSDP1.")
        initializer: _ModuleInitializer = _ModuleInitializer(enabled=False)
        uninitialized_modules: set[str] = initializer.materialize_distributed(model, device)
        if uninitialized_modules and self.is_global_zero:
            warnings.warn(
                "Parameter initialization is incomplete for modules without "
                f"reset_parameters(): {', '.join(sorted(uninitialized_modules))}.",
                UserWarning,
                stacklevel=3
            )
        placed_model: nn.Module = model.to(device)
        self.connect(placed_model)
        self.barrier()
        return placed_model

    @override
    def training_step(self, model: nn.Module, batch: Batch, batch_idx: int) -> StepOutput:
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
        self._topology.teardown()

    @override
    def on_exception(self, exception: BaseException) -> None:
        self._topology.teardown()
        self._runtime.on_exception()

    @contextmanager
    @override
    def tensor_init_context(self, empty_init: bool | None = None) -> Iterator[None]:
        if empty_init is not None and type(empty_init) is not bool:
            raise TypeError("`empty_init` must be a bool or None.")
        empty_context: AbstractContextManager[None] = (
            torch.device("meta") if empty_init else nullcontext()
        )
        with empty_context, self.precision_plugin.tensor_init_context():
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
        del optimizer

    @override
    def attach_precision_plugin(self, precision_plugin: Precision) -> None:
        supported_precisions: tuple[str, ...] = ("32-true", "bf16-mixed", "bf16-true", "16-true")
        if precision_plugin.precision not in supported_precisions:
            raise MisconfigurationError(
                f"ModelParallelStrategy does not support {precision_plugin.precision!r}. "
                f"Choose one of {supported_precisions}."
            )
        self.precision_plugin: Precision = precision_plugin

    @override
    def attach_cluster_environment(self, cluster_environment: ClusterEnvironment) -> None:
        self._runtime.attach_cluster_environment(cluster_environment)

    @property
    @override
    def distributed_sampler_kwargs(self) -> dict[str, int] | None:
        return self._topology.distributed_sampler_kwargs

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
    def device_mesh(self) -> DeviceMesh:
        return self._topology.device_mesh

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
