import math
import shutil
import sys
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
        self._stream = sys.stderr
        self._interactive = self._stream.isatty()
        self._enabled = True
        self._active_line = False
        self._spinner_index = 0
        self._last_render_time = 0.0

    def next_spinner(self) -> str:
        frames: tuple[str, ...] = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
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
        minutes: int
        remaining_seconds: int
        minutes, remaining_seconds = divmod(whole_seconds, 60)
        if minutes < 60:
            return f"{minutes}m{remaining_seconds:02d}s"
        hours: int
        remaining_minutes: int
        hours, remaining_minutes = divmod(minutes, 60)
        if hours < 24:
            return f"{hours}h{remaining_minutes:02d}m"
        days: int
        remaining_hours: int
        days, remaining_hours = divmod(hours, 24)
        return f"{days}d{remaining_hours:02d}h"

    def write_live(self, line: str, force: bool) -> None:
        if not self._enabled or not self._interactive:
            return
        current_time: float = time.perf_counter()
        if not force and current_time - self._last_render_time < self.refresh_interval:
            return
        self._last_render_time = current_time
        rendered: str = self._fit_terminal(line)
        try:
            self._stream.write(f"\r\x1b[2K{rendered}")
            self._stream.flush()
            self._active_line = True
        except (OSError, ValueError):
            self._enabled = False

    def write_permanent(self, line: str) -> None:
        if not self._enabled:
            return
        rendered: str = self._fit_terminal(line) if self._interactive else line
        prefix: str = "\r\x1b[2K" if self._interactive and self._active_line else ""
        try:
            self._stream.write(f"{prefix}{rendered}\n")
            self._stream.flush()
            self._active_line = False
        except (OSError, ValueError):
            self._enabled = False

    def clear_live(self) -> None:
        if not self._enabled or not self._interactive or not self._active_line:
            return
        try:
            self._stream.write("\r\x1b[2K")
            self._stream.flush()
            self._active_line = False
        except (OSError, ValueError):
            self._enabled = False

    def _fit_terminal(self, line: str) -> str:
        width: int = max(20, shutil.get_terminal_size(fallback=(100, 24)).columns)
        if len(line) <= width:
            return line
        return f"{line[:width - 1]}…"
