from __future__ import annotations

from typing import TYPE_CHECKING

import torch.multiprocessing as mp

from syntheticmind.strategies.launchers.contracts import (
    ProcessStartMethod,
    StageResult
)
from syntheticmind.strategies.launchers.environment import _DistributedEnvironment
from syntheticmind.strategies.launchers.multiprocessing import _MultiprocessingLauncher
from syntheticmind.strategies.launchers.state_transfer import _StageInvoker
from syntheticmind.strategies.launchers.subprocess import _SubprocessLauncher
from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.types import TrainerStage

if TYPE_CHECKING:
    from syntheticmind.core.datamodule import DataModule
    from syntheticmind.core.module import Module
    from syntheticmind.core.trainer import Trainer
    from syntheticmind.strategies.ddp import DDPStrategy

__all__: list[str] = ["ProcessStartMethod"]


class _ProcessLauncher:
    def __init__(
        self,
        strategy: DDPStrategy,
        start_method: ProcessStartMethod
    ) -> None:
        available_start_methods: set[str] = {"popen", *mp.get_all_start_methods()}
        if start_method not in available_start_methods:
            raise ValueError(
                f"The process start method {start_method!r} is unavailable. "
                f"Available methods: {sorted(available_start_methods)}."
            )
        self._strategy: DDPStrategy = strategy
        self._start_method: ProcessStartMethod = start_method
        self._environment: _DistributedEnvironment = _DistributedEnvironment(strategy)
        self._subprocess_launcher: _SubprocessLauncher = _SubprocessLauncher(
            strategy,
            self._environment
        )
        self._multiprocessing_launcher: _MultiprocessingLauncher = (
            _MultiprocessingLauncher(strategy, self._environment, start_method)
        )

    def launch(
        self,
        stage: TrainerStage,
        trainer: Trainer,
        module: Module,
        datamodule: DataModule,
        stage_kwargs: dict[str, object]
    ) -> StageResult:
        if not self.should_launch:
            return _StageInvoker.invoke(
                stage,
                trainer,
                module,
                datamodule,
                stage_kwargs
            )
        if self._strategy.num_nodes != 1:
            raise MisconfigurationError(
                "Local process launch supports num_nodes=1. Launch multi-node jobs through "
                "torchrun or a cluster scheduler that exports the distributed environment."
            )
        if stage == "predict" and self._start_method != "popen":
            return_predictions: object = stage_kwargs.get("return_predictions", True)
            if return_predictions is True:
                raise MisconfigurationError(
                    "Returning predictions is not supported by multiprocessing DDP launch. "
                    "Set return_predictions=False, use strategy='ddp', or launch with torchrun."
                )
        if self._start_method == "popen":
            return self._subprocess_launcher.launch(
                stage,
                trainer,
                module,
                datamodule,
                stage_kwargs
            )
        return self._multiprocessing_launcher.launch(
            stage,
            trainer,
            module,
            datamodule,
            stage_kwargs
        )

    @property
    def start_method(self) -> ProcessStartMethod:
        return self._start_method

    @property
    def is_interactive_compatible(self) -> bool:
        return self._start_method == "fork"

    @property
    def creates_processes_externally(self) -> bool:
        return self._environment.creates_processes_externally

    @property
    def should_launch(self) -> bool:
        return not self.creates_processes_externally

    @property
    def active_process_count(self) -> int:
        return (
            len(self._subprocess_launcher.processes)
            + len(self._multiprocessing_launcher.processes)
        )
