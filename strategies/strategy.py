from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Literal, cast

import torch
from torch import nn

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.core.module import Module
from syntheticmind.plugins.checkpoint_io import CheckpointIO
from syntheticmind.plugins.precision.precision import Precision
from syntheticmind.strategies.checkpoint import (
    StrategyCheckpoint,
    _SingleFileCheckpoint
)
from syntheticmind.strategies.launchers.launcher import _ProcessLauncher
from syntheticmind.utilities.model_initialization import _ModuleInitializer
from syntheticmind.utilities.types import (
    BackwardInputs,
    Batch,
    ModelOutput,
    OptimizerClosure,
    OptimizerStepOutput,
    OptimizerStepValue,
    StepArgument,
    StepOutput,
    TrainingStepOutput
)

__all__: list[str] = ["Strategy"]

type StepName = Literal[
    "predict_step",
    "test_step",
    "training_step",
    "validation_step"
]


class Strategy:
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
        self.model: nn.Module | None = None
        self.optimizers: list[torch.optim.Optimizer] = []
        self.checkpoint: StrategyCheckpoint = _SingleFileCheckpoint(checkpoint_io)
        self.precision_plugin: Precision = precision_plugin or Precision()
        self.configured_precision_plugin: Precision | None = precision_plugin
        self._module: Module | None = None
        self._root_device: torch.device = (
            self.parallel_devices[0]
            if self.parallel_devices
            else torch.device("cpu")
        )
        self._forward_redirection: _ForwardRedirection = _ForwardRedirection()

    def configure_topology(
        self,
        accelerator: Accelerator,
        parallel_devices: list[torch.device],
        num_nodes: int
    ) -> None:
        if num_nodes != 1 or len(parallel_devices) != 1:
            raise ValueError("Single-device execution requires one node and one device.")
        self.accelerator: Accelerator | None = accelerator
        self.parallel_devices: list[torch.device] = list(parallel_devices)
        self.num_nodes: int = num_nodes
        self._root_device: torch.device = parallel_devices[0]

    def connect(self, model: nn.Module) -> None:
        if not isinstance(model, Module):
            raise TypeError("Strategy requires a SyntheticMind Module.")
        self._module: Module | None = model
        self.model: nn.Module | None = model
        self.checkpoint.connect_model(model)

    def setup_environment(self) -> None:
        pass

    def setup(self, model: nn.Module, device: torch.device) -> nn.Module:
        self._root_device: torch.device = device
        placed_model: nn.Module = model.to(device)
        self.model: nn.Module | None = placed_model
        return placed_model

    def training_step(
        self,
        model: nn.Module,
        batch: Batch,
        batch_idx: int
    ) -> TrainingStepOutput:
        return cast(
            TrainingStepOutput,
            self._execute_step(model, "training_step", batch, batch_idx)
        )

    def validation_step(
        self,
        model: nn.Module,
        *args: StepArgument,
        **kwargs: StepArgument
    ) -> StepOutput:
        return cast(
            StepOutput,
            self._execute_step(model, "validation_step", *args, **kwargs)
        )

    def test_step(
        self,
        model: nn.Module,
        *args: StepArgument,
        **kwargs: StepArgument
    ) -> StepOutput:
        return cast(
            StepOutput,
            self._execute_step(model, "test_step", *args, **kwargs)
        )

    def predict_step(
        self,
        model: nn.Module,
        *args: StepArgument,
        **kwargs: StepArgument
    ) -> ModelOutput:
        return self._execute_step(model, "predict_step", *args, **kwargs)

    def backward(
        self,
        closure_loss: torch.Tensor,
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

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        closure: OptimizerClosure,
        model: nn.Module | None = None,
        **kwargs: OptimizerStepValue
    ) -> OptimizerStepOutput:
        resolved_model: nn.Module | None = model if model is not None else self._module
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

    def teardown(self) -> None:
        pass

    def on_exception(self, exception: BaseException) -> None:
        del exception

    @contextmanager
    def tensor_init_context(
        self,
        empty_init: bool | None = None
    ) -> Iterator[None]:
        if empty_init is not None and type(empty_init) is not bool:
            raise TypeError("`empty_init` must be a bool or None.")
        with (
            _ModuleInitializer(enabled=bool(empty_init)),
            self.root_device,
            self.precision_plugin.tensor_init_context()
        ):
            yield

    def barrier(self) -> None:
        pass

    def broadcast[T](self, value: T, source_rank: int = 0) -> T:
        del source_rank
        return value

    def reduce_boolean_decision(
        self,
        decision: bool,
        all: bool = True
    ) -> bool:
        if type(all) is not bool:
            raise TypeError("`all` must be a bool.")
        return decision

    def all_gather(
        self,
        tensor: torch.Tensor,
        group: torch.distributed.ProcessGroup | None = None,
        sync_grads: bool = False
    ) -> torch.Tensor:
        del group, sync_grads
        return tensor

    def sync_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer

    def attach_precision_plugin(self, precision_plugin: Precision) -> None:
        if not isinstance(precision_plugin, Precision):
            raise TypeError("precision_plugin must be a Precision instance.")
        self.precision_plugin: Precision = precision_plugin

    @property
    def distributed_sampler_kwargs(self) -> dict[str, int] | None:
        return None

    @property
    def launcher(self) -> _ProcessLauncher | None:
        return None

    @property
    def num_processes(self) -> int:
        return 1

    @property
    def global_rank(self) -> int:
        return 0

    @property
    def local_rank(self) -> int:
        return 0

    @property
    def node_rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1

    @property
    def is_global_zero(self) -> bool:
        return self.global_rank == 0

    @property
    def root_device(self) -> torch.device:
        return self._root_device

    def _execute_step(
        self,
        model: nn.Module,
        step_name: StepName,
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


class _ForwardRedirection:
    def __init__(self) -> None:
        self._active: bool = False
        self._module: Module | None = None
        self._original_forward: Callable[..., ModelOutput] | None = None
        self._step: Callable[..., ModelOutput] | None = None
        self._wrapper: nn.Module | None = None

    def execute(
        self,
        wrapper: nn.Module,
        module: Module,
        step_name: StepName,
        *args: StepArgument,
        **kwargs: StepArgument
    ) -> ModelOutput:
        step: Callable[..., ModelOutput] = cast(
            Callable[..., ModelOutput],
            getattr(module, step_name)
        )
        if wrapper is module:
            return step(*args, **kwargs)
        if self._active:
            raise RuntimeError("A wrapped module step is already active.")
        original_forward: Callable[..., ModelOutput] = cast(
            Callable[..., ModelOutput],
            module.forward
        )
        self._active: bool = True
        self._module: Module | None = module
        self._original_forward: Callable[..., ModelOutput] | None = original_forward
        self._step: Callable[..., ModelOutput] | None = step
        self._wrapper: nn.Module | None = wrapper
        module.forward: Callable[..., ModelOutput] = self._redirected_forward
        try:
            output: ModelOutput = wrapper(*args, **kwargs)
            self._after_outer_forward(wrapper, module)
            return output
        finally:
            module.forward: Callable[..., ModelOutput] = original_forward
            self._active: bool = False
            self._module: Module | None = None
            self._original_forward: Callable[..., ModelOutput] | None = None
            self._step: Callable[..., ModelOutput] | None = None
            self._wrapper: nn.Module | None = None

    def _redirected_forward(
        self,
        *args: StepArgument,
        **kwargs: StepArgument
    ) -> ModelOutput:
        module: Module | None = self._module
        original_forward: Callable[..., ModelOutput] | None = self._original_forward
        step: Callable[..., ModelOutput] | None = self._step
        wrapper: nn.Module | None = self._wrapper
        if module is None or original_forward is None or step is None or wrapper is None:
            raise RuntimeError("Wrapped module step redirection is not active.")
        module.forward: Callable[..., ModelOutput] = original_forward
        output: ModelOutput = step(*args, **kwargs)
        self._after_inner_forward(wrapper, module)
        return output

    def _after_inner_forward(self, wrapper: nn.Module, module: Module) -> None:
        del wrapper, module

    def _after_outer_forward(self, wrapper: nn.Module, module: Module) -> None:
        del wrapper, module
