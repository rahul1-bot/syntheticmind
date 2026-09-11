# This module:
# 1. Pins ProgressBarCallback output through complete Trainer stages on
#    redirected and interactive stderr doubles
#
# Design decisions:
# - Assertions read complete stderr text so ordering between the live line,
#   diagnostics, client prints, and permanent records is part of the contract
# - Failure and interruption records are asserted with their coordinates because
#   those coordinates are the first thing a researcher reads after a crash
#
# Author: Rahul Sawhney

import io
import pickle
import sys
from pathlib import Path

import pytest

from syntheticmind import Trainer
from syntheticmind.callbacks.progress_bar import ProgressBarCallback

from tests.support import LinearRegressor, ProbeData, TerminalStringIO


class TestRedirectedOutput:
    # Redirected runs receive plain-text headers, epoch records, and diagnostics.
    def test_fit_header_epochs_and_completion(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stream: io.StringIO = io.StringIO()
        monkeypatch.setattr(sys, "stderr", stream)
        data: ProbeData = ProbeData(samples=32, batch_size=8)
        model: LinearRegressor = LinearRegressor(emit_at_batch=1)
        trainer: Trainer = Trainer(
            accelerator="cpu",
            max_epochs=2,
            seed=5,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            default_root_dir=str(tmp_path)
        )
        trainer.fit(model, train_dataloaders=data.loader(), val_dataloaders=data.loader())
        text: str = stream.getvalue()
        lines: list[str] = text.splitlines()
        assert lines[0] == "syntheticmind · fit"
        execution: str = (
            "accelerator cpu · devices 1 · nodes 1 · world 1 · "
            "strategy SingleDeviceStrategy · precision 32-true · seed 5"
        )
        assert lines[1] == execution
        assert lines[2] == "max epochs 2 · max steps unbounded"
        assert "warning · loguru record from training_step" in lines
        warning_prefix: str = (
            "warning · UserWarning: python warning from training_step · support.py:"
        )
        assert any(line.startswith(warning_prefix) for line in lines)
        epoch_lines: list[str] = [line for line in lines if line.startswith("✓ epoch ")]
        assert len(epoch_lines) == 2
        assert epoch_lines[0].startswith("✓ epoch 1/2 · train_loss_epoch ")
        assert " · val_loss " in epoch_lines[0]
        assert "train_loss_step" not in epoch_lines[0]
        assert lines[-1].startswith("✓ fit complete · val_loss ")
        assert "\x1b[" not in text
        assert "\r" not in text

    def test_failure_record_carries_coordinates(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stream: io.StringIO = io.StringIO()
        monkeypatch.setattr(sys, "stderr", stream)
        data: ProbeData = ProbeData(samples=32, batch_size=8)
        model: LinearRegressor = LinearRegressor(fail_at_batch=2)
        trainer: Trainer = Trainer(
            accelerator="cpu",
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            default_root_dir=str(tmp_path)
        )
        with pytest.raises(RuntimeError, match="injected training failure"):
            trainer.fit(model, train_dataloaders=data.loader())
        lines: list[str] = stream.getvalue().splitlines()
        failure: str = (
            "✗ fit failed · epoch 1 · step 2 · batch 2 · "
            "RuntimeError: injected training failure"
        )
        assert lines[-1] == failure

    def test_verbose_results_render_as_a_block(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stream: io.StringIO = io.StringIO()
        monkeypatch.setattr(sys, "stderr", stream)
        data: ProbeData = ProbeData(samples=16, batch_size=8)
        trainer: Trainer = Trainer(
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            default_root_dir=str(tmp_path)
        )
        trainer.test(LinearRegressor(), dataloaders=data.loader())
        lines: list[str] = stream.getvalue().splitlines()
        assert lines[0] == "syntheticmind · test"
        assert lines[-2] == "test results"
        assert lines[-1].startswith("  test_loss  ")


class TestInteractiveOutput:
    # Interactive runs rewrite one live line with spinner and bar glyphs.
    def test_live_line_uses_control_sequences_and_glyphs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stream: TerminalStringIO = TerminalStringIO()
        monkeypatch.setattr(sys, "stderr", stream)
        data: ProbeData = ProbeData(samples=32, batch_size=8)
        trainer: Trainer = Trainer(
            accelerator="cpu",
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            callbacks=[ProgressBarCallback(refresh_interval=0.0)],
            default_root_dir=str(tmp_path)
        )
        trainer.fit(LinearRegressor(), train_dataloaders=data.loader())
        text: str = stream.getvalue()
        assert "\r\x1b[2K" in text
        assert "━━━━━━━━━━━━ 100%" in text
        assert " · lr 0.1000" in text
        assert "\x1b[3" not in text
        assert text.rstrip("\n").splitlines()[-1].startswith("✓ fit complete")


class TestConfiguration:
    # Construction validates its knobs and survives launcher pickling.
    def test_invalid_values_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            ProgressBarCallback(max_metrics=0)
        with pytest.raises(ValueError):
            ProgressBarCallback(max_metrics=True)
        with pytest.raises(TypeError):
            ProgressBarCallback(capture_diagnostics="yes")
        with pytest.raises(ValueError):
            ProgressBarCallback(redirected_interval=-0.5)

    def test_pickle_round_trip_keeps_configuration(self) -> None:
        callback: ProgressBarCallback = ProgressBarCallback(
            refresh_interval=0.5,
            redirected_interval=None,
            max_metrics=6,
            capture_diagnostics=False
        )
        restored: ProgressBarCallback = pickle.loads(pickle.dumps(callback))
        assert restored.renderer.refresh_interval == 0.5
        assert restored.renderer.redirected_interval is None
        assert not restored.renderer.diagnostics_enabled
        assert restored.is_enabled
