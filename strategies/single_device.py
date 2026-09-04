from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast, override

import torch
from torch import nn

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.core.module import Module
from syntheticmind.plugins.io import CheckpointIO
from syntheticmind.precision.precision import Precision
from syntheticmind.strategies.checkpoint_connector import (
    StrategyCheckpoint,
    _CheckpointConnector
)
from syntheticmind.strategies.forward_redirection import StepName, _ForwardRedirection
from syntheticmind.strategies.launcher import _ProcessLauncher
from syntheticmind.strategies.strategy import Strategy
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

__all__: list[str] = ["SingleDeviceStrategy"]


class SingleDeviceStrategy(Strategy):
    def __init__(
        self,
        accelerator: Accelerator | None = None,
        parallel_devices: list[torch.device] | None = None,
        checkpoint_io: CheckpointIO | None = None,
        precision_plugin: Precision | None = None
    ) -> None:
        self.accelerator: Accelerator | None = accelerator
        self.parallel_devices: list[torch.device] = list(parallel_devices or [])
        self.num_nodes: int = 1
        self.is_distributed: bool = False
        self._device: torch.device = (
            self.parallel_devices[0]
            if self.parallel_devices
            else torch.device("cpu")
        )
        self._module: Module | None = None
        self.model: nn.Module | None = None
        self.optimizers: list[torch.optim.Optimizer] = []
        self.checkpoint: StrategyCheckpoint = _CheckpointConnector(checkpoint_io)
        self.precision_plugin: Precision = precision_plugin or Precision()
        self.configured_precision_plugin: Precision | None = precision_plugin
        self._forward_redirection: _ForwardRedirection = _ForwardRedirection()

    @override
    def configure_topology(
        self, accelerator: Accelerator, parallel_devices: list[torch.device],
        num_nodes: int
    ) -> None:
        if num_nodes != 1 or len(parallel_devices) != 1:
            raise ValueError("SingleDeviceStrategy requires one node and one device.")
        self.accelerator: Accelerator | None = accelerator
        self.parallel_devices: list[torch.device] = list(parallel_devices)
        self._device: torch.device = parallel_devices[0]

    @override
    def connect(self, model: nn.Module) -> None:
        if not isinstance(model, Module):
            raise TypeError("SingleDeviceStrategy requires a SyntheticMind Module.")
        self._module: Module | None = model
        self.model: nn.Module | None = model
        self.checkpoint.connect_model(model)

    @override
    def setup_environment(self) -> None:
        pass

    @override
    def setup(self, model: nn.Module, device: torch.device) -> nn.Module:
        self._device: torch.device = device
        placed_model: nn.Module = model.to(device)
        self.model: nn.Module | None = placed_model
        return placed_model

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
        pass

    @override
    def on_exception(self, exception: BaseException) -> None:
        del exception

    @contextmanager
    @override
    def tensor_init_context(
        self, empty_init: bool | None = None
    ) -> Iterator[None]:
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
        pass

    @override
    def broadcast[T](self, value: T, source_rank: int = 0) -> T:
        del source_rank
        return value

    @override
    def reduce_boolean_decision(
        self, decision: bool, all: bool = True
    ) -> bool:
        if type(all) is not bool:
            raise TypeError("`all` must be a bool.")
        return decision

    @override
    def all_gather(
        self, tensor: torch.Tensor,
        group: torch.distributed.ProcessGroup | None = None,
        sync_grads: bool = False
    ) -> torch.Tensor:
        del group, sync_grads
        return tensor

    @override
    def sync_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer

    @override
    def attach_precision_plugin(self, precision_plugin: Precision) -> None:
        if not isinstance(precision_plugin, Precision):
            raise TypeError("precision_plugin must be a Precision instance.")
        self.precision_plugin: Precision = precision_plugin

    @property
    @override
    def distributed_sampler_kwargs(self) -> dict[str, int] | None:
        return None

    @property
    @override
    def launcher(self) -> _ProcessLauncher | None:
        return None

    @property
    @override
    def num_processes(self) -> int:
        return 1

    @property
    @override
    def global_rank(self) -> int:
        return 0

    @property
    @override
    def local_rank(self) -> int:
        return 0

    @property
    @override
    def node_rank(self) -> int:
        return 0

    @property
    @override
    def world_size(self) -> int:
        return 1

    @property
    @override
    def is_global_zero(self) -> bool:
        return self.global_rank == 0

    @property
    @override
    def root_device(self) -> torch.device:
        return self._device

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
