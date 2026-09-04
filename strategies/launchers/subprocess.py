import __main__
import atexit
import os
import subprocess
import sys
import threading
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import cast, override

import syntheticmind
import torch

from syntheticmind.core.datamodule import DataModule
from syntheticmind.core.module import Module
from syntheticmind.strategies.launchers.contracts import (
    StageArguments,
    StageResult
)
from syntheticmind.strategies.launchers.environment import _DistributedEnvironment
from syntheticmind.strategies.launchers.process_observer import _ChildProcessObserver
from syntheticmind.strategies.launchers.protocols import LaunchStrategy
from syntheticmind.exceptions import MisconfigurationError, SIGTERMException
from syntheticmind.types import TrainerStage

__all__: list[str] = []

type _SubprocessLauncherStateValue = (
    LaunchStrategy
    | _DistributedEnvironment
    | _ChildProcessObserver
    | list[subprocess.Popen[bytes]]
    | dict[str, str | None]
    | int
    | None
)
type _SubprocessLauncherState = dict[str, _SubprocessLauncherStateValue]


class _SubprocessLauncher:
    def __init__(
        self,
        strategy: LaunchStrategy,
        environment: _DistributedEnvironment
    ) -> None:
        self._strategy: LaunchStrategy = strategy
        self._environment: _DistributedEnvironment = environment
        self._processes: list[subprocess.Popen[bytes]] = []
        self._observer: _ChildProcessObserver | None = None
        self._environment_snapshot: dict[str, str | None] | None = None
        self._num_threads_snapshot: int | None = None
        atexit.register(self._close)

    def launch(
        self,
        stage: TrainerStage,
        trainer: syntheticmind.Trainer,
        module: Module,
        datamodule: DataModule | None,
        stage_kwargs: StageArguments
    ) -> StageResult:
        if self._processes:
            raise RuntimeError("The subprocess launcher already has active workers.")
        environment: dict[str, str] = self._environment.local_values()
        self._environment_snapshot: dict[str, str | None] | None = (
            self._environment.capture()
        )
        try:
            command: list[str] = self._command()
            os.environ.update(self._environment.rank_values(environment, 0))
            self._num_threads_snapshot: int | None = torch.get_num_threads()
            torch.set_num_threads(int(environment["OMP_NUM_THREADS"]))
            self._start_workers(command, environment)
            self._observer: _ChildProcessObserver | None = _ChildProcessObserver(
                self._processes,
                os.getpid()
            )
            self._observer.start()
            return trainer._invoke_stage(
                stage,
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

    def kill(self, signum: int) -> None:
        process: subprocess.Popen[bytes]
        for process in self._processes:
            if process.poll() is None:
                try:
                    process.send_signal(signum)
                except ProcessLookupError:
                    continue

    @property
    def processes(self) -> list[subprocess.Popen[bytes]]:
        active_processes: list[subprocess.Popen[bytes]] = []
        process: subprocess.Popen[bytes]
        for process in self._processes:
            if process.poll() is None:
                active_processes.append(process)
        return active_processes

    def _start_workers(
        self,
        command: list[str],
        environment: dict[str, str]
    ) -> None:
        local_rank: int
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
        main_spec: ModuleSpec | None = __main__.__spec__
        module_name: str | None = main_spec.name if main_spec is not None else None
        if module_name:
            return [sys.executable, "-m", module_name, *sys.argv[1:]]
        if not sys.argv or sys.argv[0] in {"", "-", "-c"}:
            raise MisconfigurationError(
                "strategy='ddp' requires an importable module or a Python script file. "
                "Use strategy='ddp_fork' for an interactive session."
            )
        script_path: Path = Path(sys.argv[0]).expanduser().resolve()
        if not script_path.is_file():
            raise MisconfigurationError(
                f"Cannot re-execute the current program because {script_path} is not a file. "
                "Use strategy='ddp_fork' for an interactive session."
            )
        return [sys.executable, str(script_path), *sys.argv[1:]]

    def _terminate_workers(self) -> None:
        process: subprocess.Popen[bytes]
        for process in self._processes:
            if process.poll() is None:
                process.terminate()
        process: subprocess.Popen[bytes]
        for process in self._processes:
            if process.poll() is not None:
                continue
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def _close(self) -> None:
        process: subprocess.Popen[bytes]
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
        self._environment_snapshot: dict[str, str | None] | None = None
        if self._num_threads_snapshot is not None:
            torch.set_num_threads(self._num_threads_snapshot)
            self._num_threads_snapshot: int | None = None

    def _stop_observer(self) -> None:
        if self._observer is None:
            return
        self._observer.stop()
        if self._observer is not threading.current_thread():
            self._observer.join(timeout=1)
        self._observer: _ChildProcessObserver | None = None

    @override
    def __getstate__(self) -> _SubprocessLauncherState:
        state: _SubprocessLauncherState = cast(
            _SubprocessLauncherState,
            dict(self.__dict__)
        )
        state["_processes"]: _SubprocessLauncherStateValue = []
        state["_observer"]: _SubprocessLauncherStateValue = None
        state["_environment_snapshot"]: _SubprocessLauncherStateValue = None
        state["_num_threads_snapshot"]: _SubprocessLauncherStateValue = None
        return state
