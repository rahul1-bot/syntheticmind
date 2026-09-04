import math
import shutil
import sys
import textwrap
import time
from typing import TextIO

__all__: list[str] = ["ConsoleRenderer"]


class ConsoleRenderer:
    def __init__(self, refresh_interval: float = 0.08) -> None:
        if refresh_interval < 0.0:
            raise ValueError(f"refresh_interval must be non-negative, got {refresh_interval}")
        self.refresh_interval: float = refresh_interval
        self._stream: TextIO = sys.stderr
        self._interactive: bool = False
        self._enabled: bool = True
        self._active_line: bool = False
        self._spinner_index: int = 0
        self._last_render_time: float = 0.0
        self.reset()

    @property
    def interactive(self) -> bool:
        return self._interactive

    @property
    def enabled(self) -> bool:
        return self._enabled

    def reset(self) -> None:
        self._stream: TextIO = sys.stderr
        self._interactive: bool = self._stream.isatty()
        self._enabled: bool = True
        self._active_line: bool = False
        self._spinner_index: int = 0
        self._last_render_time: float = 0.0

    def __getstate__(self) -> dict[str, float]:
        return {"refresh_interval": self.refresh_interval}

    def __setstate__(self, state: dict[str, float]) -> None:
        self.__init__(state["refresh_interval"])

    def next_spinner(self) -> str:
        frames: tuple[str, ...] = (
            "⠋",
            "⠙",
            "⠹",
            "⠸",
            "⠼",
            "⠴",
            "⠦",
            "⠧",
            "⠇",
            "⠏"
        )
        frame: str = frames[self._spinner_index % len(frames)]
        self._spinner_index += 1
        return frame

    def metric_parts(self, metrics: dict[str, float]) -> list[str]:
        parts: list[str] = []
        for name, value in metrics.items():
            clean_name: str = " ".join(name.split())
            parts.append(f"{clean_name} {self.format_metric(value)}")
            if len(parts) == 2:
                break
        return parts

    def format_metric(self, value: float) -> str:
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        absolute: float = abs(value)
        if absolute != 0.0 and (absolute < 0.001 or absolute >= 10000.0):
            return f"{value:.2e}"
        return f"{value:.3f}".rstrip("0").rstrip(".")

    def format_duration(self, seconds: float) -> str:
        if seconds < 1.0:
            return "<1s"
        whole_seconds: int = int(seconds)
        if whole_seconds < 60:
            return f"{whole_seconds}s"
        minute_parts: tuple[int, int] = divmod(whole_seconds, 60)
        minutes: int = minute_parts[0]
        remaining_seconds: int = minute_parts[1]
        if minutes < 60:
            return f"{minutes}m{remaining_seconds:02d}s"
        hour_parts: tuple[int, int] = divmod(minutes, 60)
        hours: int = hour_parts[0]
        remaining_minutes: int = hour_parts[1]
        if hours < 24:
            return f"{hours}h{remaining_minutes:02d}m"
        day_parts: tuple[int, int] = divmod(hours, 24)
        days: int = day_parts[0]
        remaining_hours: int = day_parts[1]
        return f"{days}d{remaining_hours:02d}h"

    def write_live(self, line: str, force: bool) -> None:
        if not self._enabled or not self._interactive:
            return
        current_time: float = time.perf_counter()
        if not force and current_time - self._last_render_time < self.refresh_interval:
            return
        self._last_render_time: float = current_time
        rendered: str = self._fit_terminal(line)
        try:
            self._stream.write(f"\r\x1b[2K{rendered}")
            self._stream.flush()
            self._active_line: bool = True
        except (OSError, ValueError):
            self._enabled: bool = False

    def write_permanent(self, line: str) -> None:
        if not self._enabled:
            return
        rendered: str = self._fit_terminal(line) if self._interactive else line
        prefix: str = "\r\x1b[2K" if self._interactive and self._active_line else ""
        try:
            self._stream.write(f"{prefix}{rendered}\n")
            self._stream.flush()
            self._active_line: bool = False
        except (OSError, ValueError):
            self._enabled: bool = False

    def write_record(self, label: str, fields: list[str]) -> None:
        record: str = " · ".join((label, *fields))
        lines: list[str] = textwrap.wrap(
            record,
            width=self._terminal_width(),
            subsequent_indent="  ",
            break_long_words=True,
            break_on_hyphens=False
        )
        for line in lines or [label]:
            self.write_permanent(line)

    def write_metrics(
        self,
        stage: str,
        results: list[dict[str, float]]
    ) -> None:
        result_count: int = len(results)
        for result_index, result in enumerate(results):
            if not result:
                continue
            label: str = (
                f"{stage} results"
                if result_count == 1
                else f"{stage} {result_index + 1}/{result_count}"
            )
            fields: list[str] = []
            for name, value in sorted(
                result.items(),
                key=self._metric_sort_key
            ):
                display_name: str = " ".join(
                    name.split("/dataloader_idx_", 1)[0].split()
                )
                fields.append(f"{display_name} {self.format_metric(value)}")
            self.write_record(label, fields)

    def clear_live(self) -> None:
        if not self._enabled or not self._interactive or not self._active_line:
            return
        try:
            self._stream.write("\r\x1b[2K")
            self._stream.flush()
            self._active_line: bool = False
        except (OSError, ValueError):
            self._enabled: bool = False

    def _fit_terminal(self, line: str) -> str:
        width: int = self._terminal_width()
        if len(line) <= width:
            return line
        return f"{line[:width - 1]}…"

    def _metric_sort_key(self, metric: tuple[str, float]) -> str:
        return metric[0].split("/dataloader_idx_", 1)[0]

    def _terminal_width(self) -> int:
        return max(20, shutil.get_terminal_size(fallback=(100, 24)).columns)
