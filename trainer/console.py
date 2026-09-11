# This module:
# 1. Defines ConsoleRenderer, the terminal mechanics used by Trainer and the
#    progress callback for live lines, permanent records, metric formatting,
#    duration and rate formatting, progress glyphs, styling, record packing,
#    and output failure tolerance
# 2. Owns diagnostics capture, which routes loguru records and Python warnings
#    through the same stream state while a stage renders so they never splice
#    into an active live line
#
# Design decisions:
# - Console rendering is presentation only; runtime state, metrics, timing, and
#   progress ownership remain with Trainer, loops, callbacks, and profilers
# - Interactive terminals receive one carriage-return live line truncated to the
#   terminal width, while permanent records wrap naturally and redirected
#   output receives stable plain text without control sequences
# - Interactive live refresh is throttled by monotonic performance time and force
#   may bypass cadence at lifecycle boundaries; redirected output instead
#   receives a throttled permanent heartbeat so batch or scheduler logs of long
#   epochs still show progress, and force never bypasses that heartbeat cadence
# - ANSI styling is applied only when the stream is an interactive terminal,
#   NO_COLOR is absent, and TERM is not dumb; every styled string degrades to its
#   plain text so redirected records never carry escape sequences
# - Metric rendering uses fixed-width decimal or scientific notation so a live
#   line does not jitter between frames, without changing stored metric precision
# - Diagnostics capture replaces loguru's pre-installed default handler only
#   while it is still present and restores an equivalent handler on release, so
#   client-configured sinks are never removed; a client that already replaced
#   the default handler may see duplicated records, which is preferred over
#   silently discarding client sinks. Warnings addressed to an explicit file keep
#   Python's original hook semantics
# - Output failures disable future rendering instead of aborting training
# - Pickle state retains only configuration and reconstructs stream-dependent
#   state in the worker process
#
# Author: Rahul Sawhney

import math
import os
import re
import shutil
import sys
import time
import warnings
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import ClassVar, Literal, TextIO, override

from loguru import logger as log

__all__: list[str] = ["ConsoleRenderer"]

type ConsoleStyle = Literal["bold", "dim", "green", "red", "yellow", "cyan"]
type _ShowWarning = Callable[
    [Warning | str, type[Warning], str, int, TextIO | None, str | None],
    None
]


class ConsoleRenderer:
    # Stateful terminal writer shared by progress presentation, Trainer stage
    # markers, evaluation result records, diagnostics capture, and rank-safe
    # client printing. One instance owns the live-line state of one stderr
    # stream, so every permanent write that must clear an active live line has
    # to reach the same instance.
    _spinner_frames: ClassVar[tuple[str, ...]] = (
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
    _style_codes: ClassVar[dict[ConsoleStyle, str]] = {
        "bold": "\x1b[1m",
        "dim": "\x1b[2m",
        "green": "\x1b[32m",
        "red": "\x1b[31m",
        "yellow": "\x1b[33m",
        "cyan": "\x1b[36m"
    }
    _style_pattern: ClassVar[re.Pattern[str]] = re.compile(r"\x1b\[[0-9;]*m")
    _log_level_styles: ClassVar[dict[str, ConsoleStyle]] = {
        "TRACE": "dim",
        "DEBUG": "dim",
        "SUCCESS": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red"
    }

    def __init__(
        self,
        refresh_interval: float = 0.08,
        redirected_interval: float | None = 60.0,
        capture_diagnostics: bool = True
    ) -> None:
        # Validates both cadences and the capture policy, binds stderr, and
        # initializes terminal state.
        #
        # Args:
        #     refresh_interval: Minimum seconds between non-forced interactive
        #         live rewrites. Default: ``0.08``.
        #     redirected_interval: Minimum seconds between permanent progress
        #         heartbeats when stderr is not a terminal, or None to emit no
        #         heartbeat at all. Default: ``60.0``.
        #     capture_diagnostics: Whether capture_diagnostics routes loguru
        #         records and Python warnings through this renderer.
        #         Default: ``True``.
        #
        # Raises:
        #     ValueError: If either interval is negative.
        #     TypeError: If capture_diagnostics is not a bool.
        if refresh_interval < 0.0:
            raise ValueError(f"refresh_interval must be non-negative, got {refresh_interval}")
        if redirected_interval is not None and redirected_interval < 0.0:
            raise ValueError(
                f"redirected_interval must be non-negative or None, got {redirected_interval}"
            )
        if type(capture_diagnostics) is not bool:
            raise TypeError("capture_diagnostics must be a bool.")
        self.refresh_interval: float = refresh_interval
        self.redirected_interval: float | None = redirected_interval
        self._diagnostics_enabled: bool = capture_diagnostics
        self._stream: TextIO = sys.stderr
        self._interactive: bool = False
        self._colored: bool = False
        self._enabled: bool = True
        self._active_line: bool = False
        self._spinner_index: int = 0
        self._last_render_time: float = 0.0
        self._last_permanent_time: float = 0.0
        self._loguru_handler_id: int | None = None
        self._default_loguru_removed: bool = False
        self._previous_showwarning: _ShowWarning | None = None
        self.reset()

    @property
    def interactive(self) -> bool:
        # Reports whether the current process stderr supports live rewriting.
        return self._interactive

    @property
    def colored(self) -> bool:
        # Reports whether style requests currently produce ANSI sequences.
        return self._colored

    @property
    def enabled(self) -> bool:
        # Reports whether stream output remains healthy after prior writes.
        return self._enabled

    @property
    def diagnostics_enabled(self) -> bool:
        # Reports the configured capture policy independent of capture state.
        return self._diagnostics_enabled

    def reset(self) -> None:
        # Rebinds stderr, TTY capability, and color policy in the current process
        # and clears all live rendering state. Diagnostics capture is left as it
        # is because capture and release own their own idempotent lifecycle.
        self._stream: TextIO = sys.stderr
        self._interactive: bool = self._stream.isatty()
        self._colored: bool = (
            self._interactive
            and "NO_COLOR" not in os.environ
            and os.environ.get("TERM", "") != "dumb"
        )
        self._enabled: bool = True
        self._active_line: bool = False
        self._spinner_index: int = 0
        self._last_render_time: float = 0.0
        self._last_permanent_time: float = time.perf_counter()

    @override
    def __getstate__(self) -> tuple[float, float | None, bool]:
        # Serializes only stable configuration across launcher boundaries.
        return (
            self.refresh_interval,
            self.redirected_interval,
            self._diagnostics_enabled
        )

    def __setstate__(self, state: tuple[float, float | None, bool]) -> None:
        # Reconstructs stream-dependent state inside the receiving worker.
        self.__init__(state[0], state[1], state[2])

    def style(self, text: str, style: ConsoleStyle) -> str:
        # Wraps text in one ANSI style when color is active and otherwise returns
        # the text unchanged, so callers compose lines without checking policy.
        if not self._colored:
            return text
        return f"{self._style_codes[style]}{text}\x1b[0m"

    def spinner_frame(self) -> str:
        # Returns the current spinner frame. The frame advances only when a live
        # line is actually written, so the spinner animates at the render cadence
        # rather than at the batch rate.
        return self._spinner_frames[self._spinner_index % len(self._spinner_frames)]

    def progress_bar(self, completed: int, total: int, width: int = 12) -> str:
        # Renders a fixed-width glyph bar for a finite horizon. The remainder uses
        # a distinct glyph so the bar stays readable without color.
        if total <= 0 or width <= 0:
            return ""
        filled: int = max(0, min(width, int(width * completed / total)))
        remaining: int = width - filled
        return f"{self.style('━' * filled, 'cyan')}{self.style('╌' * remaining, 'dim')}"

    def metric_parts(self, metrics: dict[str, float], limit: int) -> list[str]:
        # Sanitizes names and formats at most limit owner-ordered metrics.
        parts: list[str] = []
        for name, value in metrics.items():
            if len(parts) >= limit:
                break
            clean_name: str = " ".join(name.split())
            parts.append(f"{clean_name} {self.format_metric(value)}")
        return parts

    def format_metric(self, value: float) -> str:
        # Formats non-finite values explicitly and selects fixed four-decimal or
        # scientific notation by magnitude so consecutive frames keep one width.
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        absolute: float = abs(value)
        if absolute != 0.0 and (absolute < 0.001 or absolute >= 10000.0):
            return f"{value:.2e}"
        return f"{value:.4f}"

    def format_duration(self, seconds: float) -> str:
        # Compacts elapsed or estimated duration from subsecond through day scale.
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

    def format_rate(self, completed: int, elapsed: float) -> str | None:
        # Formats batch throughput as iterations per second, or seconds per
        # iteration below one, and returns None before timing evidence exists.
        if completed < 2 or elapsed < 0.5:
            return None
        rate: float = completed / elapsed
        if rate >= 1.0:
            return f"{rate:.1f} it/s"
        return f"{1.0 / rate:.1f} s/it"

    def write_live(self, line: str, force: bool) -> None:
        # Throttles and rewrites one interactive line, or emits one throttled
        # permanent heartbeat when output is redirected, disabling output after
        # a stream failure without affecting the training lifecycle.
        if not self._enabled:
            return
        current_time: float = time.perf_counter()
        if not self._interactive:
            if (
                self.redirected_interval is None
                or current_time - self._last_permanent_time < self.redirected_interval
            ):
                return
            self.write_permanent(line)
            return
        if not force and current_time - self._last_render_time < self.refresh_interval:
            return
        self._last_render_time: float = current_time
        rendered: str = self._fit_terminal(line)
        try:
            self._stream.write(f"\r\x1b[2K{rendered}")
            self._stream.flush()
            self._active_line: bool = True
            self._spinner_index += 1
        except (OSError, ValueError):
            self._enabled: bool = False

    def write_permanent(self, line: str) -> None:
        # Clears any live line and emits one durable record. Permanent records
        # are never truncated because the terminal wraps them without losing
        # information; redirected records drop styling so logs stay plain text.
        if not self._enabled:
            return
        rendered: str = line if self._interactive else self._strip_styles(line)
        prefix: str = "\r\x1b[2K" if self._interactive and self._active_line else ""
        try:
            self._stream.write(f"{prefix}{rendered}\n")
            self._stream.flush()
            self._active_line: bool = False
            self._last_permanent_time: float = time.perf_counter()
        except (OSError, ValueError):
            self._enabled: bool = False

    def write_record(
        self,
        label: str,
        fields: list[str],
        style: ConsoleStyle | None = None
    ) -> None:
        # Packs a labeled field record into terminal-width lines, breaking only
        # between whole fields so a separator never opens a continuation line,
        # and applies one optional style after packing so escape sequences never
        # distort the width computation.
        width: int = self._terminal_width()
        lines: list[str] = [label]
        for field in fields:
            candidate: str = f"{lines[-1]} · {field}"
            if len(candidate) <= width:
                lines[-1] = candidate
            else:
                lines.append(f"  {field}")
        for line in lines:
            self.write_permanent(line if style is None else self.style(line, style))

    def write_metrics(
        self,
        stage: str,
        results: list[dict[str, float]]
    ) -> None:
        # Emits one aligned name and value block per non-empty dataloader result
        # with stable metric ordering and hidden internal loader suffixes.
        result_count: int = len(results)
        for result_index, result in enumerate(results):
            if not result:
                continue
            label: str = (
                f"{stage} results"
                if result_count == 1
                else f"{stage} results {result_index + 1}/{result_count}"
            )
            rows: list[tuple[str, str]] = []
            for name, value in sorted(
                result.items(),
                key=self._metric_sort_key
            ):
                display_name: str = " ".join(
                    name.split("/dataloader_idx_", 1)[0].split()
                )
                rows.append((display_name, self.format_metric(value)))
            name_width: int = max(len(row[0]) for row in rows)
            value_width: int = max(len(row[1]) for row in rows)
            self.write_permanent(self.style(label, "bold"))
            for display_name, formatted_value in rows:
                self.write_permanent(
                    f"  {display_name:<{name_width}}  {formatted_value:>{value_width}}"
                )

    def clear_live(self) -> None:
        # Erases an active interactive line during disable, print, or teardown.
        if not self._enabled or not self._interactive or not self._active_line:
            return
        try:
            self._stream.write("\r\x1b[2K")
            self._stream.flush()
            self._active_line: bool = False
        except (OSError, ValueError):
            self._enabled: bool = False

    def capture_diagnostics(self) -> None:
        # Routes loguru records and Python warnings through this renderer until
        # release_diagnostics. Loguru's pre-installed default handler is removed
        # only when it is still present, so client-configured sinks stay
        # untouched. Repeated capture without release is a no-op.
        if not self._diagnostics_enabled:
            return
        if self._previous_showwarning is None:
            self._previous_showwarning: _ShowWarning | None = warnings.showwarning
            warnings.showwarning = self._show_warning
        if self._loguru_handler_id is None:
            try:
                log.remove(0)
                self._default_loguru_removed: bool = True
            except ValueError:
                self._default_loguru_removed: bool = False
            self._loguru_handler_id: int | None = log.add(
                self._write_log_record,
                format="{level.name}|{message}",
                level="DEBUG",
                colorize=False
            )

    def release_diagnostics(self) -> None:
        # Restores the previous warning hook and loguru's default stderr handler
        # idempotently, so a release after failure followed by teardown performs
        # no double release, and a client hook installed mid-stage is never
        # overwritten.
        if self._loguru_handler_id is not None:
            handler_id: int = self._loguru_handler_id
            self._loguru_handler_id: int | None = None
            with suppress(ValueError):
                log.remove(handler_id)
        if self._default_loguru_removed:
            log.add(sys.stderr)
            self._default_loguru_removed: bool = False
        if self._previous_showwarning is not None:
            if warnings.showwarning == self._show_warning:
                warnings.showwarning = self._previous_showwarning
            self._previous_showwarning: _ShowWarning | None = None

    def _write_log_record(self, message: str) -> None:
        # Loguru sink receiving the level-prefixed formatted record. The first
        # line carries a styled level label and continuation lines are indented,
        # so multi-line exception records remain readable beneath the live line.
        text: str = str(message).rstrip("\n")
        split: tuple[str, str, str] = text.partition("|")
        level_name: str = split[0]
        body: str = split[2]
        style: ConsoleStyle | None = self._log_level_styles.get(level_name)
        label: str = level_name.lower()
        prefix: str = label if style is None else self.style(label, style)
        for index, line in enumerate(body.split("\n")):
            self.write_permanent(f"{prefix} · {line}" if index == 0 else f"  {line}")

    def _show_warning(
        self,
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: TextIO | None = None,
        line: str | None = None
    ) -> None:
        # warnings.showwarning replacement. Warnings addressed to an explicit
        # file keep Python's original semantics; terminal warnings become one
        # compact record with their source location.
        if file is not None and self._previous_showwarning is not None:
            self._previous_showwarning(message, category, filename, lineno, file, line)
            return
        location: str = f"{Path(filename).name}:{lineno}"
        self.write_permanent(
            f"{self.style('warning', 'yellow')} · {category.__name__}: {message} · {location}"
        )

    def _fit_terminal(self, line: str) -> str:
        # Truncates a live line to the current terminal width with one visible
        # overflow marker so carriage-return rewriting stays on one row. Style
        # sequences are copied through untouched and a final reset is appended,
        # so a cut keeps its colors and never leaves styling open.
        width: int = self._terminal_width()
        if self._visible_length(line) <= width:
            return line
        budget: int = width - 1
        pieces: list[str] = []
        visible: int = 0
        position: int = 0
        for match in self._style_pattern.finditer(line):
            segment: str = line[position:match.start()]
            take: int = min(len(segment), budget - visible)
            pieces.append(segment[:take])
            visible += take
            pieces.append(match.group(0))
            position: int = match.end()
        tail: str = line[position:]
        pieces.append(tail[:min(len(tail), budget - visible)])
        reset: str = "\x1b[0m" if self._colored else ""
        return f"{''.join(pieces)}…{reset}"

    def _visible_length(self, line: str) -> int:
        # Measures printed columns by ignoring ANSI style sequences.
        return len(self._strip_styles(line))

    def _strip_styles(self, line: str) -> str:
        # Removes every ANSI style sequence produced by style.
        return self._style_pattern.sub("", line)

    def _metric_sort_key(self, metric: tuple[str, float]) -> str:
        # Orders multi-loader metrics by their public base identity rather than
        # internal dataloader suffix.
        return metric[0].split("/dataloader_idx_", 1)[0]

    def _terminal_width(self) -> int:
        # Enforces a readable minimum when terminal metadata is unavailable or
        # implausibly narrow.
        return max(20, shutil.get_terminal_size(fallback=(100, 24)).columns)
