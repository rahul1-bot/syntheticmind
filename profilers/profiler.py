from __future__ import annotations

import re
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

from torch import nn

__all__: list[str] = ["Profiler"]


class Profiler(ABC):
    def __init__(
        self,
        dirpath: str | Path | None = None,
        filename: str | None = None
    ) -> None:
        self.dirpath: str | Path | None = dirpath
        self.filename: str | None = filename
        self._output_file: TextIO | None = None
        self._write_stream: Callable[[str], object] | None = None
        self._local_rank: int | None = None
        self._stage: str | None = None
        self._module: nn.Module | None = None

    @abstractmethod
    def start(self, action_name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self, action_name: str) -> None:
        raise NotImplementedError

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

    def _rank_zero_info(self, *args: object, **kwargs: object) -> None:
        del kwargs
        if self._local_rank not in (None, 0):
            return
        message: str = " ".join(str(argument) for argument in args)
        try:
            sys.stderr.write(message)
            if not message.endswith("\n"):
                sys.stderr.write("\n")
            sys.stderr.flush()
        except (OSError, ValueError):
            return

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
        basename = re.sub(r"[\\/:*?\"<>|\n\r\t]", "_", basename)
        basename = re.sub(r"_+", "_", basename).strip()
        return f"{basename or 'profile'}{extension}"

    def _prepare_streams(self) -> None:
        if self._write_stream is not None:
            return
        if self.filename and self.dirpath:
            directory: Path = Path(self.dirpath).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            path: Path = directory / self._prepare_filename()
            self._output_file = path.open("a", encoding="utf-8")
            self._write_stream = self._output_file.write
            return
        self._write_stream = self._rank_zero_info

    def describe(self) -> None:
        self._prepare_streams()
        try:
            report: str = self.summary()
            if report and self._write_stream is not None:
                self._write_stream(report)
                if not report.endswith("\n"):
                    self._write_stream("\n")
            if self._output_file is not None:
                self._output_file.flush()
        finally:
            self.teardown(stage=self._stage)

    def _stats_to_str(self, stats: dict[str, str]) -> str:
        prefix: str = f"{self._stage.upper()} " if self._stage is not None else ""
        lines: list[str] = [f"{prefix}Profiler Report"]
        for action_name, value in stats.items():
            header: str = f"Profile stats for: {action_name}"
            if self._local_rank is not None:
                header = f"{header} rank: {self._local_rank}"
            lines.extend((header, value))
        return "\n".join(lines)

    def setup(
        self,
        stage: str,
        local_rank: int | None = None,
        log_dir: str | None = None
    ) -> None:
        self._stage = stage
        self._local_rank = local_rank
        if self.dirpath is None:
            self.dirpath = log_dir

    def teardown(self, stage: str | None) -> None:
        del stage
        self._write_stream = None
        if self._output_file is not None:
            self._output_file.close()
            self._output_file = None

    def _attach_module(self, module: nn.Module) -> None:
        self._module = module

    def __del__(self) -> None:
        try:
            self.teardown(stage=getattr(self, "_stage", None))
        except BaseException:
            return

    @property
    def local_rank(self) -> int:
        return 0 if self._local_rank is None else self._local_rank
