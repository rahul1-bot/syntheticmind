# This module:
# 1. Pins ConsoleRenderer formatting, styling, live-line, record, result, and
#    diagnostics-capture contracts against in-memory stderr doubles
#
# Design decisions:
# - Every renderer is constructed after sys.stderr is replaced, because reset
#   binds the stream and terminal policy at construction time
# - Color assertions remove NO_COLOR explicitly; the suite otherwise runs with
#   NO_COLOR set so plain-text assertions stay deterministic
# - Diagnostics assertions cover both directions of the capture lifecycle so a
#   leaked sink or warning hook fails here rather than in an unrelated test
#
# Author: Rahul Sawhney

import io
import sys
import warnings
from collections.abc import Callable
from typing import TextIO

import pytest
from loguru import logger as log

from syntheticmind.trainer.console import ConsoleRenderer

from tests.support import TerminalStringIO

type _ShowWarning = Callable[
    [Warning | str, type[Warning], str, int, TextIO | None, str | None],
    None
]


class TestFormatting:
    # Value formatting must be width-stable and honest about non-finite values.
    def test_metric_widths_are_fixed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "stderr", io.StringIO())
        renderer: ConsoleRenderer = ConsoleRenderer()
        assert renderer.format_metric(0.5) == "0.5000"
        assert renderer.format_metric(1.0) == "1.0000"
        assert renderer.format_metric(0.0) == "0.0000"
        assert renderer.format_metric(0.0001) == "1.00e-04"
        assert renderer.format_metric(123456.0) == "1.23e+05"
        assert renderer.format_metric(float("nan")) == "nan"
        assert renderer.format_metric(float("-inf")) == "-inf"

    def test_duration_scales(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "stderr", io.StringIO())
        renderer: ConsoleRenderer = ConsoleRenderer()
        assert renderer.format_duration(0.4) == "<1s"
        assert renderer.format_duration(59.9) == "59s"
        assert renderer.format_duration(61.0) == "1m01s"
        assert renderer.format_duration(3725.0) == "1h02m"
        assert renderer.format_duration(90000.0) == "1d01h"

    def test_rate_requires_evidence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "stderr", io.StringIO())
        renderer: ConsoleRenderer = ConsoleRenderer()
        assert renderer.format_rate(1, 10.0) is None
        assert renderer.format_rate(10, 0.1) is None
        assert renderer.format_rate(20, 2.0) == "10.0 it/s"
        assert renderer.format_rate(2, 8.0) == "4.0 s/it"

    def test_metric_parts_respect_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "stderr", io.StringIO())
        renderer: ConsoleRenderer = ConsoleRenderer()
        metrics: dict[str, float] = {"a": 1.0, "b  c": 2.0, "d": 3.0}
        assert renderer.metric_parts(metrics, 2) == ["a 1.0000", "b c 2.0000"]

    def test_invalid_configuration_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "stderr", io.StringIO())
        with pytest.raises(ValueError):
            ConsoleRenderer(refresh_interval=-1.0)
        with pytest.raises(ValueError):
            ConsoleRenderer(redirected_interval=-1.0)
        with pytest.raises(TypeError):
            ConsoleRenderer(capture_diagnostics="no")


class TestStyling:
    # Styling exists only on an interactive stream without NO_COLOR.
    def test_plain_when_redirected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "stderr", io.StringIO())
        renderer: ConsoleRenderer = ConsoleRenderer()
        assert not renderer.interactive
        assert not renderer.colored
        assert renderer.style("x", "bold") == "x"
        assert renderer.progress_bar(3, 12) == "━━━╌╌╌╌╌╌╌╌╌"

    def test_plain_on_terminal_with_no_color(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "stderr", TerminalStringIO())
        renderer: ConsoleRenderer = ConsoleRenderer()
        assert renderer.interactive
        assert not renderer.colored

    def test_styled_on_terminal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")
        monkeypatch.setattr(sys, "stderr", TerminalStringIO())
        renderer: ConsoleRenderer = ConsoleRenderer()
        assert renderer.colored
        assert renderer.style("x", "green") == "\x1b[32mx\x1b[0m"
        full_bar: str = "\x1b[36m━━━━━━━━━━━━\x1b[0m\x1b[2m\x1b[0m"
        assert renderer.progress_bar(12, 12) == full_bar

    def test_dumb_terminal_is_plain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "dumb")
        monkeypatch.setattr(sys, "stderr", TerminalStringIO())
        assert not ConsoleRenderer().colored


class TestLiveLine:
    # Interactive lines rewrite in place and truncate; redirected output
    # receives a throttled heartbeat and never a control sequence.
    def test_interactive_rewrite_and_spinner_cadence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stream: TerminalStringIO = TerminalStringIO()
        monkeypatch.setattr(sys, "stderr", stream)
        renderer: ConsoleRenderer = ConsoleRenderer(refresh_interval=1000.0)
        first: str = renderer.spinner_frame()
        renderer.write_live("one", force=True)
        renderer.write_live("throttled", force=False)
        second: str = renderer.spinner_frame()
        renderer.write_permanent("done")
        assert stream.getvalue() == "\r\x1b[2Kone\r\x1b[2Kdone\n"
        assert first != second

    def test_truncation_preserves_styles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm")
        monkeypatch.setenv("COLUMNS", "24")
        stream: TerminalStringIO = TerminalStringIO()
        monkeypatch.setattr(sys, "stderr", stream)
        renderer: ConsoleRenderer = ConsoleRenderer()
        line: str = f"{renderer.style('abcdefghij', 'cyan')} {'x' * 40}"
        renderer.write_live(line, force=True)
        written: str = stream.getvalue()
        assert written.startswith("\r\x1b[2K\x1b[36mabcdefghij\x1b[0m ")
        assert written.endswith("…\x1b[0m")
        visible: str = written[len("\r\x1b[2K"):].replace("\x1b[36m", "").replace("\x1b[0m", "")
        assert len(visible) == 24

    def test_redirected_heartbeat_cadence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stream: io.StringIO = io.StringIO()
        monkeypatch.setattr(sys, "stderr", stream)
        renderer: ConsoleRenderer = ConsoleRenderer(redirected_interval=0.0)
        renderer.write_live("beat", force=False)
        renderer.write_live("beat", force=True)
        assert stream.getvalue() == "beat\nbeat\n"
        silent: ConsoleRenderer = ConsoleRenderer(redirected_interval=None)
        silent.write_live("never", force=True)
        assert stream.getvalue() == "beat\nbeat\n"

    def test_redirected_records_drop_styles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stream: io.StringIO = io.StringIO()
        monkeypatch.setattr(sys, "stderr", stream)
        renderer: ConsoleRenderer = ConsoleRenderer()
        renderer.write_permanent("\x1b[1mbold\x1b[0m text")
        assert stream.getvalue() == "bold text\n"

    def test_stream_failure_disables_rendering(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stream: io.StringIO = io.StringIO()
        monkeypatch.setattr(sys, "stderr", stream)
        renderer: ConsoleRenderer = ConsoleRenderer()
        stream.close()
        renderer.write_permanent("lost")
        assert not renderer.enabled


class TestRecords:
    # Records pack whole fields per line and result blocks align values.
    def test_record_packing_never_opens_with_separator(
        self,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("COLUMNS", "30")
        stream: io.StringIO = io.StringIO()
        monkeypatch.setattr(sys, "stderr", stream)
        renderer: ConsoleRenderer = ConsoleRenderer()
        renderer.write_record(
            "accelerator cpu",
            ["devices 1", "strategy SingleDeviceStrategy", "seed 7"]
        )
        lines: list[str] = stream.getvalue().splitlines()
        assert lines == [
            "accelerator cpu · devices 1",
            "  strategy SingleDeviceStrategy",
            "  seed 7"
        ]

    def test_metric_block_alignment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stream: io.StringIO = io.StringIO()
        monkeypatch.setattr(sys, "stderr", stream)
        renderer: ConsoleRenderer = ConsoleRenderer()
        renderer.write_metrics(
            "test",
            [{"val_loss/dataloader_idx_0": 1.5, "acc": 0.25}, {}]
        )
        assert stream.getvalue().splitlines() == [
            "test results 1/2",
            "  acc       0.2500",
            "  val_loss  1.5000"
        ]


class TestDiagnostics:
    # Capture routes loguru and warnings through the renderer; release restores
    # the previous hooks exactly.
    def test_capture_and_release(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stream: io.StringIO = io.StringIO()
        monkeypatch.setattr(sys, "stderr", stream)
        original_hook: _ShowWarning = warnings.showwarning
        renderer: ConsoleRenderer = ConsoleRenderer()
        renderer.capture_diagnostics()
        renderer.capture_diagnostics()
        log.warning("captured {}", 1)
        warnings.warn("captured warning", UserWarning, stacklevel=1)
        captured: str = stream.getvalue()
        assert "warning · captured 1\n" in captured
        assert "warning · UserWarning: captured warning · test_console.py:" in captured
        renderer.release_diagnostics()
        renderer.release_diagnostics()
        assert warnings.showwarning is original_hook
        log.warning("released")
        assert "warning · released" not in stream.getvalue()

    def test_capture_disabled_by_configuration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stream: io.StringIO = io.StringIO()
        monkeypatch.setattr(sys, "stderr", stream)
        original_hook: _ShowWarning = warnings.showwarning
        renderer: ConsoleRenderer = ConsoleRenderer(capture_diagnostics=False)
        renderer.capture_diagnostics()
        assert warnings.showwarning is original_hook
        renderer.release_diagnostics()
