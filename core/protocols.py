from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Protocol

import torch

__all__: list[str] = [
    "OptimizerEpochLoopProtocol",
    "OptimizerFitLoopProtocol",
    "OptimizerModuleProtocol",
    "OptimizerStrategyProtocol",
    "OptimizerTrainerProtocol"
]


class OptimizerStrategyProtocol(Protocol):
    @property
    def model(self) -> torch.nn.Module | None: ...


class OptimizerModuleProtocol(Protocol):
    def toggled_optimizer(
        self,
        optimizer: torch.optim.Optimizer
    ) -> AbstractContextManager[None]: ...


class OptimizerEpochLoopProtocol(Protocol):
    def optimizer_step(
        self,
        trainer: OptimizerTrainerProtocol,
        module: OptimizerModuleProtocol,
        optimizer: torch.optim.Optimizer,
        closure: Callable[[], object] | None = None,
        kwargs: dict[str, object] | None = None,
        automatic: bool = False,
        increment_global_step: bool = True
    ) -> object: ...


class OptimizerFitLoopProtocol(Protocol):
    epoch_loop: OptimizerEpochLoopProtocol | None


class OptimizerTrainerProtocol(Protocol):
    strategy: OptimizerStrategyProtocol
    fit_loop: OptimizerFitLoopProtocol
