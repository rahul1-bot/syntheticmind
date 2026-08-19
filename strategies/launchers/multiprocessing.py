from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import (
    ProcessContext,
    ProcessExitedException,
    start_processes
)

from syntheticmind.accelerators.cpu import CPUAccelerator
from syntheticmind.strategies.launchers.contracts import (
    ProcessStartMethod,
    StageResult,
    _GlobalState,
    _WorkerOutput
)
from syntheticmind.strategies.launchers.environment import _DistributedEnvironment
from syntheticmind.strategies.launchers.state_transfer import (
    _StageInvoker,
    _WorkerStateTransfer
)
from syntheticmind.utilities.exceptions import MisconfigurationError, SIGTERMException
from syntheticmind.utilities.types import TrainerStage

if TYPE_CHECKING:
    from syntheticmind.core.datamodule import DataModule
    from syntheticmind.core.module import Module
    from syntheticmind.core.trainer import Trainer
    from syntheticmind.strategies.ddp import DDPStrategy

__all__: list[str] = []


class _MultiprocessingWorker:
    @classmethod
    def run(
        cls,
        local_rank: int,
        trainer: Trainer,
        stage: TrainerStage,
        module: Module,
        datamodule: DataModule,
        stage_kwargs: dict[str, object],
        environment: dict[str, str],
        output_path: Path,
        initial_global_state: _GlobalState
    ) -> None:
        worker_environment: dict[str, str] = dict(environment)
        worker_environment["LOCAL_RANK"] = str(local_rank)
        worker_environment["RANK"] = str(local_rank)
        os.environ.update(worker_environment)
        torch.set_num_threads(int(worker_environment["OMP_NUM_THREADS"]))
        initial_global_state.restore()
        if isinstance(trainer.accelerator, CPUAccelerator):
            _WorkerStateTransfer.unshare_module_storage(module)
        result: StageResult = _StageInvoker.invoke(
            stage,
            trainer,
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
    def __init__(
        self,
        strategy: DDPStrategy,
        environment: _DistributedEnvironment,
        start_method: ProcessStartMethod
    ) -> None:
        self._strategy: DDPStrategy = strategy
        self._environment: _DistributedEnvironment = environment
        self._start_method: ProcessStartMethod = start_method
        self._processes: list[mp.Process] = []

    def launch(
        self,
        stage: TrainerStage,
        trainer: Trainer,
        module: Module,
        datamodule: DataModule,
        stage_kwargs: dict[str, object]
    ) -> StageResult:
        self._validate_launch(trainer, module, datamodule, stage_kwargs)
        environment: dict[str, str] = self._environment.local_values()
        initial_global_state: _GlobalState = _GlobalState.capture()
        with tempfile.TemporaryDirectory(prefix="syntheticmind-launch-") as directory:
            output_path: Path = Path(directory) / "rank_zero_output.pt"
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

    @property
    def processes(self) -> list[mp.Process]:
        return list(self._processes)

    def kill(self, signum: int) -> None:
        for process in self._processes:
            if process.pid is not None and process.is_alive():
                try:
                    os.kill(process.pid, signum)
                except ProcessLookupError:
                    continue

    def _validate_launch(
        self,
        trainer: Trainer,
        module: Module,
        datamodule: DataModule,
        stage_kwargs: dict[str, object]
    ) -> None:
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
        trainer: Trainer,
        stage: TrainerStage,
        module: Module,
        datamodule: DataModule,
        stage_kwargs: dict[str, object],
        environment: dict[str, str],
        output_path: Path,
        initial_global_state: _GlobalState
    ) -> None:
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
            self._processes = list(process_context.processes)
            while not process_context.join():
                pass
        except BaseException as exception:
            trainer.state.interrupt()
            process_received_sigterm: bool = (
                isinstance(exception, ProcessExitedException)
                and exception.exit_code in {-15, 143}
            )
            if trainer.received_sigterm or process_received_sigterm:
                trainer._signal_connector.received_sigterm = True
                raise SIGTERMException() from exception
            raise
        finally:
            self._processes.clear()

    def __getstate__(self) -> dict[str, object]:
        state: dict[str, object] = dict(self.__dict__)
        state["_processes"] = []
        return state
