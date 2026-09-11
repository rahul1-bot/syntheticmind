# This module:
# 1. Defines the daemon observer that detects failed re-executed DDP workers and
#    terminates the complete local process group
#
# Design decisions:
# - Child return codes are polled outside Trainer so a rank blocked in a
#   collective still receives failure from a peer that exited early
# - Clean completion requires every child to report zero; active children keep
#   observation alive and the first nonzero result triggers job termination
# - SIGTERM exit forms propagate graceful termination, while other worker
#   failures escalate to the strongest available kill signal
# - The observer writes one causal diagnostic before signalling siblings and the
#   parent, preserving the failed local rank and return code
#
# Author: Rahul Sawhney

import os
import signal
import subprocess
import sys
import threading
from typing import override

__all__: list[str] = []


class _ChildProcessObserver(threading.Thread):
    # Daemon monitor over nonzero local subprocess ranks and their parent.
    @override
    def __init__(
        self,
        subprocesses: list[subprocess.Popen[bytes]],
        parent_pid: int
    ) -> None:
        super().__init__(daemon=True, name="syntheticmind-worker-observer")
        self._subprocesses: list[subprocess.Popen[bytes]] = subprocesses
        self._parent_pid: int = parent_pid
        self._stop_event: threading.Event = threading.Event()

    def stop(self) -> None:
        # Requests observer shutdown without mutating child process state.
        self._stop_event.set()

    def check(self) -> bool:
        # Returns true after complete clean exit or after terminating a failed job;
        # returns false while at least one child remains active.
        return_codes: list[int | None] = []
        process: subprocess.Popen[bytes]
        for process in self._subprocesses:
            return_codes.append(process.poll())
        failed_worker: tuple[int, int] | None = self._failed_worker(return_codes)
        if failed_worker is not None:
            self._terminate_job(failed_worker)
            return True
        return_code: int | None
        for return_code in return_codes:
            if return_code != 0:
                return False
        return True

    @override
    def run(self) -> None:
        # Polls at a bounded interval until stopped, completed, or failed.
        while not self._stop_event.wait(0.1):
            if self.check():
                return

    def _failed_worker(
        self,
        return_codes: list[int | None]
    ) -> tuple[int, int] | None:
        # Finds the first nonzero child return code and maps list order to local
        # ranks beginning at one because the parent owns local rank zero.
        local_rank: int
        return_code: int | None
        for local_rank, return_code in enumerate(return_codes, start=1):
            if return_code is not None and return_code != 0:
                return local_rank, return_code
        return None

    def _terminate_job(self, failed_worker: tuple[int, int]) -> None:
        # Reports the causal worker, terminates or kills remaining children, and
        # signals the parent with matching termination semantics.
        local_rank: int = failed_worker[0]
        return_code: int = failed_worker[1]
        sys.stderr.write(
            f"Distributed worker local_rank={local_rank} exited with code "
            f"{return_code}; terminating the local process group.\n"
        )
        sys.stderr.flush()
        received_sigterm: bool = return_code in {-signal.SIGTERM, 128 + signal.SIGTERM}
        process: subprocess.Popen[bytes]
        for process in self._subprocesses:
            if process.poll() is None:
                if received_sigterm:
                    process.terminate()
                else:
                    process.kill()
        termination_signal: signal.Signals = (
            signal.SIGTERM
            if received_sigterm
            else getattr(signal, "SIGKILL", signal.SIGTERM)
        )
        os.kill(self._parent_pid, termination_signal)
