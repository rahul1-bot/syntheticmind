# This module:
# 1. Defines Profiler, the stateful runtime-measurement contract for named action
#    scopes, stage and rank identity, report output, module attachment, and
#    deterministic teardown
# 2. Defines PassThroughProfiler, the explicit zero-measurement policy used when
#    profiling is disabled
#
# Design decisions:
# - Profiler is a regular inheritance contract rather than an ABC; concrete
#   profilers replace start and stop while sharing one lifecycle and report path
# - profile is the canonical action boundary used by Trainer, loops, connectors,
#   and HookDispatcher. If client code fails, a profiler stop failure is attached
#   as a note and never replaces the causal exception
# - One profiler instance may serve sequential stages but cannot own two stages
#   concurrently; setup and describe create an explicit acquire-report-release
#   transaction
# - A filename produces rank- and stage-qualified artifacts through StoragePath;
#   otherwise reports go to loguru only on local rank zero
# - Teardown clears live module, rank, stage, stream, and file state even when a
#   concrete profiler needs additional measurement-specific cleanup
#
# Author: Rahul Sawhney

import io
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import override

from loguru import logger as log
from torch import nn

from syntheticmind.storage.filesystem import StoragePath
from syntheticmind.primitives.types import TrainerStage

__all__: list[str] = ["PassThroughProfiler", "Profiler"]

type _WriteStream = Callable[[str], int | None]


class Profiler:
    # Base runtime measurement policy consumed by Trainer. Client profilers
    # subclass it, implement start and stop, optionally replace summary, and let
    # Trainer own setup, module attachment, reporting, and teardown.
    def __init__(
        self,
        dirpath: str | Path | None = None,
        filename: str | None = None
    ) -> None:
        # Configures report destination identity without opening a stream.
        #
        # Args:
        #     dirpath: Local or supported remote report directory. None defers to
        #         the active Trainer log directory. Default: ``None``.
        #     filename: Optional report identity. Without both filename and a
        #         resolved directory, summaries use rank-zero logging.
        self.dirpath: str | Path | None = dirpath
        self.filename: str | None = filename
        self._output_file: io.TextIOBase | None = None
        self._write_stream: _WriteStream | None = None
        self._local_rank: int | None = None
        self._stage: TrainerStage | None = None
        self._module: nn.Module | None = None

    def start(self, action_name: str) -> None:
        # Opens one named measurement region. Concrete profilers must reject any
        # nesting or duplicate state their technique cannot represent.
        raise NotImplementedError(
            f"{type(self).__name__}.start() must record action {action_name!r}."
        )

    def stop(self, action_name: str) -> None:
        # Closes the matching named measurement region and records its result.
        raise NotImplementedError(
            f"{type(self).__name__}.stop() must finish action {action_name!r}."
        )

    def summary(self) -> str:
        # Returns the stage report; profilers without a textual product may
        # return an empty string while exporting trace artifacts separately.
        return ""

    @contextmanager
    def profile(self, action_name: str) -> Iterator[str]:
        # Brackets one framework or client operation and preserves the operation's
        # exception as causal if measurement cleanup also fails.
        self.start(action_name)
        try:
            yield action_name
        except BaseException as exception:
            try:
                self.stop(action_name)
            except BaseException as stop_exception:
                exception.add_note(
                    f"Profiler failed to stop {action_name!r}: "
                    f"{type(stop_exception).__name__}: {stop_exception}"
                )
            raise
        else:
            self.stop(action_name)

    def setup(
        self,
        stage: TrainerStage,
        local_rank: int | None = None,
        log_dir: str | Path | None = None
    ) -> None:
        # Claims one Trainer stage, rank identity, and fallback log directory.
        #
        # Raises:
        #     RuntimeError: If this profiler already owns an active stage.
        if self._stage is not None:
            raise RuntimeError(
                f"Profiler is already active for stage {self._stage!r}."
            )
        self._stage: TrainerStage | None = stage
        self._local_rank: int | None = local_rank
        if self.dirpath is None:
            self.dirpath: str | Path | None = log_dir

    def teardown(self, stage: TrainerStage | None) -> None:
        # Releases all shared stage state and closes an owned local report stream;
        # the optional stage argument supports uniform Trainer cleanup calls.
        del stage
        output_file: io.TextIOBase | None = self._output_file
        self._write_stream: _WriteStream | None = None
        self._output_file: io.TextIOBase | None = None
        self._module: nn.Module | None = None
        self._stage: TrainerStage | None = None
        self._local_rank: int | None = None
        if output_file is not None:
            output_file.close()

    def describe(self) -> None:
        # Prepares the output sink, emits the completed summary, flushes local
        # output, and unconditionally releases the profiler stage.
        self._prepare_stream()
        try:
            report: str = self.summary()
            if report and self._write_stream is not None:
                self._write_stream(report)
                if not report.endswith("\n"):
                    self._write_stream("\n")
            if self._output_file is not None:
                self._output_file.flush()
        finally:
            self.teardown(self._stage)

    @property
    def local_rank(self) -> int:
        # Returns zero for non-distributed stages and the attached local rank for
        # rank-qualified distributed reports.
        return 0 if self._local_rank is None else self._local_rank

    @property
    def is_active(self) -> bool:
        # Reports whether setup currently owns a Trainer stage.
        return self._stage is not None

    def _attach_module(self, module: nn.Module) -> None:
        # Attaches the current client model for profilers that instrument module
        # execution; the base contract retains no hooks itself.
        self._module: nn.Module | None = module

    def _prepare_filename(
        self,
        action_name: str | None = None,
        extension: str = ".txt"
    ) -> str:
        # Builds a stage-, rank-, and action-qualified safe artifact basename.
        components: list[str] = []
        if self._stage is not None:
            components.append(self._stage)
        if self.filename:
            components.append(self.filename)
        if self._local_rank is not None:
            components.append(str(self._local_rank))
        if action_name is not None:
            components.append(action_name)
        basename: str = "-".join(components)
        basename: str = re.sub(r"[\\/:*?\"<>|\n\r\t]", "_", basename)
        basename: str = re.sub(r"_+", "_", basename).strip()
        return f"{basename or 'profile'}{extension}"

    def _prepare_stream(self) -> None:
        # Acquires one local file, remote append operation, or rank-filtered log
        # sink for the current report.
        if self._write_stream is not None:
            return
        if self.filename and self.dirpath is not None:
            path: str = StoragePath(self.dirpath).join(self._prepare_filename())
            storage_path: StoragePath = StoragePath(path)
            if storage_path.is_local:
                self._output_file: io.TextIOBase | None = storage_path.open_text(
                    "a"
                )
                self._write_stream: _WriteStream | None = self._output_file.write
            else:
                self._write_stream: _WriteStream | None = storage_path.append_text
            return
        self._write_stream: _WriteStream | None = self._write_to_log

    def _write_to_log(self, message: str) -> None:
        # Emits terminal reports only from local rank zero on each node.
        if self._local_rank not in (None, 0):
            return
        log.info(message.rstrip("\n"))

    def _format_statistics(self, statistics: dict[str, str]) -> str:
        # Creates one stage-aware textual report from per-action backend tables.
        prefix: str = f"{self._stage.upper()} " if self._stage is not None else ""
        lines: list[str] = [f"{prefix}Profiler Report"]
        action_name: str
        value: str
        for action_name, value in statistics.items():
            header: str = f"Profile stats for: {action_name}"
            if self._local_rank is not None:
                header: str = f"{header} rank: {self._local_rank}"
            lines.extend((header, value))
        return "\n".join(lines)


class PassThroughProfiler(Profiler):
    # Explicit disabled policy that preserves every profile context call without
    # allocating timers, traces, hooks, or report artifacts.
    @override
    def start(self, action_name: str) -> None:
        # Accepts the action boundary without measurement.
        del action_name

    @override
    def stop(self, action_name: str) -> None:
        # Completes the no-op action boundary.
        del action_name
