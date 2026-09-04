# This module:
# 1. Defines the process-launch coordinator selecting script re-execution,
#    spawn, fork, forkserver, or externally created worker behavior
# 2. Applies signal scope, local multi-node restrictions, prediction-return
#    constraints, and process termination across concrete launchers
#
# Design decisions:
# - ParallelStrategy owns one ProcessLauncher and clients select its start method
#   through Strategy configuration rather than invoking launcher classes directly
# - Externally created processes re-enter the Trainer stage immediately and never
#   launch another worker set
# - Local launch supports one node only; multi-node execution must arrive through
#   torchrun or a scheduler-owned ClusterEnvironment
# - Multiprocessing prediction cannot return arbitrary worker outputs to the
#   parent and therefore requires return_predictions=False; script re-execution
#   may preserve its ordinary process-local result
# - SignalConnector wraps the complete launch so SIGTERM reaches owned workers
#   and is checked before and after distributed execution
#
# Author: Rahul Sawhney

from typing import cast

import syntheticmind
import torch.multiprocessing as mp

from syntheticmind.data.datamodule import DataModule
from syntheticmind.core.module import Module
from syntheticmind.strategies.launchers.contracts import (
    ProcessStartMethod,
    StageArguments,
    StageResult
)
from syntheticmind.strategies.launchers.environment import _DistributedEnvironment
from syntheticmind.strategies.launchers.multiprocessing import _MultiprocessingLauncher
from syntheticmind.strategies.launchers.subprocess import _SubprocessLauncher
from syntheticmind.strategies.launchers.protocols import LaunchStrategy
from syntheticmind.primitives.exceptions import MisconfigurationError
from syntheticmind.primitives.types import TrainerStage

__all__: list[str] = ["ProcessStartMethod"]


class _ProcessLauncher:
    # Internal dispatcher over subprocess and torch multiprocessing policies.
    def __init__(
        self,
        strategy: LaunchStrategy,
        start_method: ProcessStartMethod
    ) -> None:
        # Validates host-supported start methods and constructs both concrete
        # launchers around one reversible distributed environment.
        available_start_methods: set[str] = {"popen", *mp.get_all_start_methods()}
        if start_method not in available_start_methods:
            raise ValueError(
                f"The process start method {start_method!r} is unavailable. "
                f"Available methods: {sorted(available_start_methods)}."
            )
        self._strategy: LaunchStrategy = strategy
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
        trainer: syntheticmind.Trainer,
        module: Module,
        datamodule: DataModule | None,
        stage_kwargs: StageArguments
    ) -> StageResult:
        # Executes directly inside externally launched workers or delegates local
        # process creation after validating node and prediction constraints.
        with trainer._signal_connector.launch_context():
            trainer._raise_if_sigterm()
            if not self.should_launch:
                result: StageResult = trainer._invoke_stage(
                    stage,
                    module,
                    datamodule,
                    stage_kwargs
                )
                trainer._raise_if_sigterm()
                return result
            if self._strategy.num_nodes != 1:
                raise MisconfigurationError(
                    "Local process launch supports num_nodes=1. Launch multi-node jobs through "
                    "torchrun or a cluster scheduler that exports the distributed environment."
                )
            if stage == "predict" and self._start_method != "popen":
                return_predictions: bool | None = cast(
                    bool | None,
                    stage_kwargs.get("return_predictions", True)
                )
                if return_predictions is True:
                    raise MisconfigurationError(
                        "Returning predictions is not supported by multiprocessing DDP launch. "
                        "Set return_predictions=False, use strategy='ddp', or launch with torchrun."
                    )
            if self._start_method == "popen":
                result: StageResult = self._subprocess_launcher.launch(
                    stage,
                    trainer,
                    module,
                    datamodule,
                    stage_kwargs
                )
            else:
                result: StageResult = self._multiprocessing_launcher.launch(
                    stage,
                    trainer,
                    module,
                    datamodule,
                    stage_kwargs
                )
            trainer._raise_if_sigterm(synchronize=False)
            return result

    def kill(self, signum: int) -> None:
        # Propagates one signal to every process owned by either concrete launcher.
        self._subprocess_launcher.kill(signum)
        self._multiprocessing_launcher.kill(signum)

    @property
    def start_method(self) -> ProcessStartMethod:
        return self._start_method

    @property
    def is_interactive_compatible(self) -> bool:
        # Fork is the only supported local method that does not require an
        # importable guarded program entry point.
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
