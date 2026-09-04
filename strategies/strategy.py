from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

import torch
from torch import nn

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.precision.precision import Precision
from syntheticmind.strategies.checkpoint_connector import StrategyCheckpoint
from syntheticmind.strategies.launcher import _ProcessLauncher
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

__all__: list[str] = ["Strategy"]


@runtime_checkable
class Strategy(Protocol):
    accelerator: Accelerator | None
    checkpoint: StrategyCheckpoint
    configured_precision_plugin: Precision | None
    is_distributed: bool
    model: nn.Module | None
    num_nodes: int
    optimizers: list[torch.optim.Optimizer]
    parallel_devices: list[torch.device]
    precision_plugin: Precision

    def configure_topology(
        self,
        accelerator: Accelerator,
        parallel_devices: list[torch.device],
        num_nodes: int
    ) -> None: ...

    def connect(self, model: nn.Module) -> None: ...

    def setup_environment(self) -> None: ...

    def setup(self, model: nn.Module, device: torch.device) -> nn.Module: ...

    def training_step(
        self,
        model: nn.Module,
        batch: Batch,
        batch_idx: int
    ) -> StepOutput: ...

    def validation_step(
        self,
        model: nn.Module,
        *args: StepArgument,
        **kwargs: StepArgument
    ) -> StepOutput: ...

    def test_step(
        self,
        model: nn.Module,
        *args: StepArgument,
        **kwargs: StepArgument
    ) -> StepOutput: ...

    def predict_step(
        self,
        model: nn.Module,
        *args: StepArgument,
        **kwargs: StepArgument
    ) -> ModelOutput: ...

    def backward(
        self,
        closure_loss: torch.Tensor,
        optimizer: torch.optim.Optimizer | None = None,
        gradient: torch.Tensor | None = None,
        retain_graph: bool | None = None,
        create_graph: bool = False,
        inputs: BackwardInputs = None
    ) -> torch.Tensor: ...

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        closure: OptimizerClosure,
        model: nn.Module | None = None,
        **kwargs: OptimizerStepValue
    ) -> OptimizerStepOutput: ...

    def teardown(self) -> None: ...

    def on_exception(self, exception: BaseException) -> None: ...

    def tensor_init_context(
        self,
        empty_init: bool | None = None
    ) -> AbstractContextManager[None]: ...

    def barrier(self) -> None: ...

    def broadcast[T](self, value: T, source_rank: int = 0) -> T: ...

    def reduce_boolean_decision(
        self,
        decision: bool,
        all: bool = True
    ) -> bool: ...

    def all_gather(
        self,
        tensor: torch.Tensor,
        group: torch.distributed.ProcessGroup | None = None,
        sync_grads: bool = False
    ) -> torch.Tensor: ...

    def sync_gradients(self, optimizer: torch.optim.Optimizer) -> None: ...

    def attach_precision_plugin(self, precision_plugin: Precision) -> None: ...

    @property
    def distributed_sampler_kwargs(self) -> dict[str, int] | None: ...

    @property
    def launcher(self) -> _ProcessLauncher | None: ...

    @property
    def num_processes(self) -> int: ...

    @property
    def global_rank(self) -> int: ...

    @property
    def local_rank(self) -> int: ...

    @property
    def node_rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    @property
    def is_global_zero(self) -> bool: ...

    @property
    def root_device(self) -> torch.device: ...
