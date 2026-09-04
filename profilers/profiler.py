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
    def __init__(
        self,
        dirpath: str | Path | None = None,
        filename: str | None = None
    ) -> None:
        self.dirpath: str | Path | None = dirpath
        self.filename: str | None = filename
        self._output_file: io.TextIOBase | None = None
        self._write_stream: _WriteStream | None = None
        self._local_rank: int | None = None
        self._stage: TrainerStage | None = None
        self._module: nn.Module | None = None

    def start(self, action_name: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}.start() must record action {action_name!r}."
        )

    def stop(self, action_name: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}.stop() must finish action {action_name!r}."
        )

    def summary(self) -> str:
        return ""

    @contextmanager
    def profile(self, action_name: str) -> Iterator[str]:
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
        if self._stage is not None:
            raise RuntimeError(
                f"Profiler is already active for stage {self._stage!r}."
            )
        self._stage: TrainerStage | None = stage
        self._local_rank: int | None = local_rank
        if self.dirpath is None:
            self.dirpath: str | Path | None = log_dir

    def teardown(self, stage: TrainerStage | None) -> None:
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
        return 0 if self._local_rank is None else self._local_rank

    @property
    def is_active(self) -> bool:
        return self._stage is not None

    def _attach_module(self, module: nn.Module) -> None:
        self._module: nn.Module | None = module

    def _prepare_filename(
        self,
        action_name: str | None = None,
        extension: str = ".txt"
    ) -> str:
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
        if self._local_rank not in (None, 0):
            return
        log.info(message.rstrip("\n"))

    def _format_statistics(self, statistics: dict[str, str]) -> str:
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
    @override
    def start(self, action_name: str) -> None:
        del action_name

    @override
    def stop(self, action_name: str) -> None:
        del action_name
