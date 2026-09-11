# This module:
# 1. Implements spawn, fork, and forkserver local process launch through
#    torch.multiprocessing
# 2. Executes one complete Trainer stage in each worker and reconciles the
#    rank-zero runtime outcome into the dormant parent
#
# Design decisions:
# - Spawn and forkserver validate that Trainer, Module, DataModule, callbacks,
#   and stage arguments are pickleable before process creation; fork retains its
#   interactive use but is prohibited with CUDA
# - The guarded-program requirement is checked before recursive Python spawn so
#   missing ``if __name__ == "__main__"`` fails with corrective guidance
# - Every worker restores parent random and deterministic process state before
#   execution and receives rank-specific rendezvous and thread variables
# - CPU forked tensors may share storage with the parent, so workers clone Module
#   parameter and buffer storage before mutation to preserve parent reconciliation
# - Only local rank zero serializes WorkerOutput; all ranks execute the stage and
#   collectives, while one authoritative result restores parent model, Trainer,
#   callbacks, metrics, data, precision, optimizer, scheduler, and RNG state
# - Process exit and SIGTERM failures interrupt Trainer state and preserve signal
#   semantics rather than returning a partial rank-zero snapshot
#
# Author: Rahul Sawhney

import os
import pickle
import tempfile
from pathlib import Path
from typing import cast, override

import syntheticmind
import torch
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import (
    ProcessContext,
    ProcessExitedException,
    start_processes
)

from syntheticmind.accelerators.cpu import CPUAccelerator
from syntheticmind.data.datamodule import DataModule
from syntheticmind.core.module import Module
from syntheticmind.strategies.launchers.contracts import (
    ProcessStartMethod,
    StageArguments,
    StageResult,
    _GlobalState,
    _WorkerOutput
)
from syntheticmind.strategies.launchers.environment import _DistributedEnvironment
from syntheticmind.strategies.launchers.protocols import LaunchStrategy
from syntheticmind.strategies.launchers.state_transfer import _WorkerStateTransfer
from syntheticmind.primitives.exceptions import MisconfigurationError, SIGTERMException
from syntheticmind.primitives.types import TrainerStage

__all__: list[str] = []

type _MultiprocessingLauncherStateValue = (
    LaunchStrategy
    | _DistributedEnvironment
    | ProcessStartMethod
    | list[mp.Process]
)
type _MultiprocessingLauncherState = dict[str, _MultiprocessingLauncherStateValue]


class _MultiprocessingWorker:
    # Importable worker entry point required by torch multiprocessing start
    # methods. It reconstructs process state, executes one stage, and publishes
    # the sole parent snapshot from local rank zero.
    @classmethod
    def run(
        cls: type[_MultiprocessingWorker],
        local_rank: int,
        trainer: syntheticmind.Trainer,
        stage: TrainerStage,
        module: Module,
        datamodule: DataModule | None,
        stage_kwargs: StageArguments,
        environment: dict[str, str],
        output_path: Path,
        initial_global_state: _GlobalState
    ) -> None:
        # Specializes environment and thread count, restores process-global state,
        # isolates CPU storage, invokes Trainer, and writes the rank-zero result.
        del cls
        launch_strategy: LaunchStrategy = cast(
            LaunchStrategy,
            trainer.strategy
        )
        worker_environment: dict[str, str] = _DistributedEnvironment(
            launch_strategy
        ).rank_values(environment, local_rank)
        os.environ.update(worker_environment)
        torch.set_num_threads(int(worker_environment["OMP_NUM_THREADS"]))
        _WorkerStateTransfer.restore_global_state(trainer, initial_global_state)
        if isinstance(trainer.accelerator, CPUAccelerator):
            _WorkerStateTransfer.unshare_module_storage(module)
        result: StageResult = trainer._invoke_stage(
            stage,
            module,
            datamodule,
            stage_kwargs
        )
        if local_rank != 0:
            return
        worker_output: _WorkerOutput = _WorkerStateTransfer.collect(
            trainer,
            module,
            datamodule,
            result
        )
        torch.save(worker_output, output_path)


class _MultiprocessingLauncher:
    # Parent-side owner for spawn, fork, or forkserver process lifetime and state
    # reconciliation.
    def __init__(
        self,
        strategy: LaunchStrategy,
        environment: _DistributedEnvironment,
        start_method: ProcessStartMethod
    ) -> None:
        # Binds topology, reversible environment, and start method with no active
        # child process state.
        self._strategy: LaunchStrategy = strategy
        self._environment: _DistributedEnvironment = environment
        self._start_method: ProcessStartMethod = start_method
        self._processes: list[mp.Process] = []

    def launch(
        self,
        stage: TrainerStage,
        trainer: syntheticmind.Trainer,
        module: Module,
        datamodule: DataModule | None,
        stage_kwargs: StageArguments
    ) -> StageResult:
        # Validates transport, captures process globals, launches and joins every
        # worker, loads the rank-zero snapshot, and restores it into the parent.
        self._validate_launch(trainer, module, datamodule, stage_kwargs)
        environment: dict[str, str] = self._environment.local_values()
        initial_global_state: _GlobalState = (
            _WorkerStateTransfer.capture_global_state(trainer)
        )
        temporary_directory: tempfile.TemporaryDirectory[str] = (
            tempfile.TemporaryDirectory(prefix="syntheticmind-launch-")
        )
        with temporary_directory:
            output_path: Path = Path(temporary_directory.name) / "rank_zero_output.pt"
            self._start_and_join(
                trainer,
                stage,
                module,
                datamodule,
                stage_kwargs,
                environment,
                output_path,
                initial_global_state
            )
            if not output_path.is_file():
                trainer.state.interrupt()
                raise RuntimeError(
                    "Distributed workers completed without producing rank-zero state."
                )
            worker_output: _WorkerOutput = cast(
                _WorkerOutput,
                torch.load(output_path, map_location="cpu", weights_only=False)
            )
        _WorkerStateTransfer.restore(
            worker_output,
            trainer,
            module,
            datamodule,
            stage
        )
        return worker_output.result

    def kill(self, signum: int) -> None:
        # Sends a signal to every live multiprocessing child and tolerates races
        # with processes that exited between inspection and delivery.
        process: mp.Process
        for process in self._processes:
            if process.pid is not None and process.is_alive():
                try:
                    os.kill(process.pid, signum)
                except ProcessLookupError:
                    continue

    @property
    def processes(self) -> list[mp.Process]:
        # Defensive list of currently tracked child process objects.
        return list(self._processes)

    def _validate_launch(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        datamodule: DataModule | None,
        stage_kwargs: StageArguments
    ) -> None:
        # Enforces guarded entry, CUDA fork safety, and serialization requirements
        # before any worker is created.
        if getattr(mp.current_process(), "_inheriting", False):
            raise RuntimeError(
                "Multiprocessing DDP requires the program entry point to be guarded by "
                "if __name__ == '__main__'. Use strategy='ddp' for script re-execution "
                "or strategy='ddp_fork' in an interactive process."
            )
        if self._start_method == "fork" and self._strategy.root_device.type == "cuda":
            raise MisconfigurationError(
                "CUDA execution cannot use the fork start method safely. Use strategy='ddp', "
                "strategy='ddp_spawn', or DDPStrategy(start_method='forkserver')."
            )
        if self._start_method not in {"spawn", "forkserver"}:
            return
        try:
            pickle.dumps((trainer, module, datamodule, stage_kwargs))
        except (AttributeError, pickle.PickleError, TypeError) as exception:
            raise MisconfigurationError(
                f"The {self._start_method!r} start method requires the Trainer, module, "
                "datamodule, callbacks, and stage arguments to be pickleable."
            ) from exception

    def _start_and_join(
        self,
        trainer: syntheticmind.Trainer,
        stage: TrainerStage,
        module: Module,
        datamodule: DataModule | None,
        stage_kwargs: StageArguments,
        environment: dict[str, str],
        output_path: Path,
        initial_global_state: _GlobalState
    ) -> None:
        # Starts all local ranks, repeatedly joins their shared ProcessContext,
        # converts termination exit codes into SIGTERMException, and clears child
        # ownership in finally.
        try:
            process_context_result: ProcessContext | None = start_processes(
                _MultiprocessingWorker.run,
                args=(
                    trainer,
                    stage,
                    module,
                    datamodule,
                    stage_kwargs,
                    environment,
                    output_path,
                    initial_global_state
                ),
                nprocs=self._strategy.num_processes,
                join=False,
                start_method=self._start_method
            )
            if process_context_result is None:
                raise RuntimeError("The multiprocessing launcher returned no process context.")
            process_context: ProcessContext = process_context_result
            self._processes: list[mp.Process] = list(process_context.processes)
            while not process_context.join():
                pass
        except BaseException as exception:
            trainer.state.interrupt()
            process_received_sigterm: bool = (
                isinstance(exception, ProcessExitedException)
                and exception.exit_code in {-15, 143}
            )
            if trainer.received_sigterm or process_received_sigterm:
                trainer._signal_connector.received_sigterm: bool = True
                raise SIGTERMException() from exception
            raise
        finally:
            self._processes.clear()

    @override
    def __getstate__(self) -> _MultiprocessingLauncherState:
        # Removes live process handles from pickle state while preserving launch
        # topology and environment policy.
        state: _MultiprocessingLauncherState = cast(
            _MultiprocessingLauncherState,
            dict(self.__dict__)
        )
        state["_processes"]: _MultiprocessingLauncherStateValue = []
        return state
