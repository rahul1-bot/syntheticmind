from __future__ import annotations

import __main__
import atexit
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from syntheticmind.strategies.launchers.contracts import StageResult
from syntheticmind.strategies.launchers.environment import _DistributedEnvironment
from syntheticmind.strategies.launchers.process_observer import _ChildProcessObserver
from syntheticmind.strategies.launchers.state_transfer import _StageInvoker
from syntheticmind.utilities.exceptions import SIGTERMException
from syntheticmind.utilities.types import TrainerStage

if TYPE_CHECKING:
    from syntheticmind.core.datamodule import DataModule
    from syntheticmind.core.module import Module
    from syntheticmind.core.trainer import Trainer
    from syntheticmind.strategies.ddp import DDPStrategy

__all__: list[str] = []


class _SubprocessLauncher:
    def __init__(
        self,
        strategy: DDPStrategy,
        environment: _DistributedEnvironment
    ) -> None:
        self._strategy: DDPStrategy = strategy
        self._environment: _DistributedEnvironment = environment
        self._processes: list[subprocess.Popen[bytes]] = []
        self._observer: _ChildProcessObserver | None = None
        self._environment_snapshot: dict[str, str | None] | None = None
        self._num_threads_snapshot: int | None = None
        atexit.register(self._close)

    def launch(
        self,
        stage: TrainerStage,
        trainer: Trainer,
        module: Module,
        datamodule: DataModule | None,
        stage_kwargs: dict[str, object]
    ) -> StageResult:
        if self._processes:
            raise RuntimeError("The subprocess launcher already has active workers.")
        environment: dict[str, str] = self._environment.local_values()
        self._environment_snapshot = self._environment.capture()
        try:
            command: list[str] = self._command()
            os.environ.update(self._environment.rank_values(environment, 0))
            self._num_threads_snapshot = torch.get_num_threads()
            torch.set_num_threads(int(environment["OMP_NUM_THREADS"]))
            self._start_workers(command, environment)
            self._observer = _ChildProcessObserver(self._processes, os.getpid())
            self._observer.start()
            return _StageInvoker.invoke(
                stage,
                trainer,
                module,
                datamodule,
                stage_kwargs
            )
        except BaseException as exception:
            trainer.state.interrupt()
            self._stop_observer()
            self._terminate_workers()
            self._restore_environment()
            if trainer.received_sigterm and not isinstance(exception, SIGTERMException):
                raise SIGTERMException() from exception
            raise

    @property
    def processes(self) -> list[subprocess.Popen[bytes]]:
        return [process for process in self._processes if process.poll() is None]

    def kill(self, signum: int) -> None:
        for process in self._processes:
            if process.poll() is None:
                try:
                    process.send_signal(signum)
                except ProcessLookupError:
                    continue

    def _start_workers(
        self,
        command: list[str],
        environment: dict[str, str]
    ) -> None:
        for local_rank in range(1, self._strategy.num_processes):
            child_environment: dict[str, str] = dict(os.environ)
            child_environment.update(
                self._environment.rank_values(environment, local_rank)
            )
            process: subprocess.Popen[bytes] = subprocess.Popen(
                command,
                env=child_environment,
                cwd=os.getcwd()
            )
            self._processes.append(process)

    def _command(self) -> list[str]:
        main_spec: object | None = __main__.__spec__
        module_name: str | None = getattr(main_spec, "name", None)
        if module_name:
            return [sys.executable, "-m", module_name, *sys.argv[1:]]
        if not sys.argv or sys.argv[0] in {"", "-", "-c"}:
            from syntheticmind.utilities.exceptions import MisconfigurationError

            raise MisconfigurationError(
                "strategy='ddp' requires an importable module or a Python script file. "
                "Use strategy='ddp_fork' for an interactive session."
            )
        script_path: Path = Path(sys.argv[0]).expanduser().resolve()
        if not script_path.is_file():
            from syntheticmind.utilities.exceptions import MisconfigurationError

            raise MisconfigurationError(
                f"Cannot re-execute the current program because {script_path} is not a file. "
                "Use strategy='ddp_fork' for an interactive session."
            )
        return [sys.executable, str(script_path), *sys.argv[1:]]

    def _terminate_workers(self) -> None:
        for process in self._processes:
            if process.poll() is None:
                process.terminate()
        for process in self._processes:
            if process.poll() is not None:
                continue
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def _close(self) -> None:
        for process in self._processes:
            if process.poll() is not None:
                continue
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if self._observer is not None:
            self._observer.check()
        self._stop_observer()
        self._processes.clear()
        self._restore_environment()

    def _restore_environment(self) -> None:
        if self._environment_snapshot is None:
            return
        self._environment.restore(self._environment_snapshot)
        self._environment_snapshot = None
        if self._num_threads_snapshot is not None:
            torch.set_num_threads(self._num_threads_snapshot)
            self._num_threads_snapshot = None

    def _stop_observer(self) -> None:
        if self._observer is None:
            return
        self._observer.stop()
        if self._observer is not threading.current_thread():
            self._observer.join(timeout=1)
        self._observer = None

    def __getstate__(self) -> dict[str, object]:
        state: dict[str, object] = dict(self.__dict__)
        state["_processes"] = []
        state["_observer"] = None
        state["_environment_snapshot"] = None
        state["_num_threads_snapshot"] = None
        return state
