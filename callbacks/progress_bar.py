# This module:
# 1. Defines ProgressBarCallback, the rank-zero lifecycle renderer for fit,
#    validation, testing, prediction, epoch summaries, and failure termination
# 2. Selects bounded progress metrics and timing information from the live
#    trainer state without owning training or metric-reduction semantics
#
# Design decisions:
# - Rendering is enabled only on global zero, preventing distributed workers
#   from racing over one terminal while every rank still receives hook calls
# - ConsoleRenderer owns ANSI and stream mechanics; this callback owns ML
#   lifecycle state, stage labels, batch counts, metric selection, and timing
# - Live updates are throttled by ConsoleRenderer, while terminal batch and
#   unknown-length boundaries force a final refresh so displayed state cannot
#   lag behind completed computation
# - Epoch summaries are deferred until the next epoch or teardown because
#   fit-time validation may contribute the metric users expect in that summary
# - At most two progress metrics are rendered to preserve terminal readability;
#   the complete metric set remains available through loggers and Trainer views
# - Client print calls clear the live line before delegating to builtins.print,
#   preventing user output from corrupting the renderer's terminal state
# - Failure rendering truncates normalized exception text and marks the stage as
#   finalized, so teardown cannot overwrite the original failure diagnosis
#
# Author: Rahul Sawhney

import builtins
import time
from pathlib import Path
from typing import IO, override

import syntheticmind
import torch

from syntheticmind.callbacks.callback import Callback
from syntheticmind.callbacks.timer import Timer
from syntheticmind.data.combined_loader import CombinedLoader
from syntheticmind.core.module import Module
from syntheticmind.trainer.console import ConsoleRenderer
from syntheticmind.primitives.types import (
    Batch,
    ModelOutput,
    RunningStage,
    StepOutput,
    TrainerStage
)

__all__: list[str] = ["ProgressBarCallback"]

type _PrintValue = ModelOutput | Path | BaseException | torch.nn.Module


class ProgressBarCallback(Callback):
    # Default terminal observer installed by Trainer when progress rendering is
    # enabled. Clients may subclass it to replace presentation while retaining
    # the same callback lifecycle, or disable and re-enable it at runtime.
    def __init__(self) -> None:
        # Initializes detached rendering state; rank and stage ownership are
        # resolved only when setup receives the active Trainer.
        self._train_bar: bool = False
        self._val_bar: bool = False
        self._test_bar: bool = False
        self._predict_bar: bool = False
        self._renderer: ConsoleRenderer = ConsoleRenderer()
        self._enabled: bool = True
        self._rendering: bool = False
        self._rank_zero: bool = False
        self._stage: TrainerStage | None = None
        self._stage_finalized: bool = False
        self._pending_epoch_line: str | None = None
        self._pending_epoch_number: int | None = None
        self._setup_started_at: float = 0.0
        self._batch_counts: dict[RunningStage, int] = {
            "training": 0,
            "validating": 0,
            "testing": 0,
            "predicting": 0,
            "sanity_checking": 0
        }
        self._stage_started_at: dict[RunningStage, float] = {
            "training": 0.0,
            "validating": 0.0,
            "testing": 0.0,
            "predicting": 0.0,
            "sanity_checking": 0.0
        }
        self._live_metrics: dict[RunningStage, dict[str, float]] = {
            "training": {},
            "validating": {},
            "testing": {},
            "predicting": {},
            "sanity_checking": {}
        }

    @property
    def is_enabled(self) -> bool:
        # Reports the client-controlled rendering switch independent of rank.
        return self._enabled

    @property
    def is_disabled(self) -> bool:
        # Complements is_enabled for callback integrations and custom trainers.
        return not self._enabled

    def disable(self) -> None:
        # Stops future rendering and clears any transient terminal line.
        self._enabled: bool = False
        self._rendering: bool = False
        self._renderer.clear_live()

    def enable(self) -> None:
        # Re-enables rendering only when this process owns rank-zero output.
        self._enabled: bool = True
        self._rendering: bool = self._rank_zero

    def print(
        self,
        *args: _PrintValue,
        sep: str | None = " ",
        end: str | None = "\n",
        file: IO[str] | None = None,
        flush: bool = False
    ) -> None:
        # Prints client output without interleaving it with an active progress
        # line, preserving the ordinary builtins.print argument contract.
        self._renderer.clear_live()
        builtins.print(*args, sep=sep, end=end, file=file, flush=flush)

    @override
    def setup(self, trainer: syntheticmind.Trainer, module: Module, stage: TrainerStage) -> None:
        # Resolves rank ownership, resets all stage-local views, and emits one
        # permanent stage heading before lifecycle hooks begin.
        self._rank_zero: bool = trainer.is_global_zero
        self._rendering: bool = self._rank_zero and self._enabled
        if not self._rendering:
            return
        self._renderer.reset()
        self._stage: TrainerStage | None = stage
        self._stage_finalized: bool = False
        self._pending_epoch_line: str | None = None
        self._pending_epoch_number: int | None = None
        self._train_bar: bool = False
        self._val_bar: bool = False
        self._test_bar: bool = False
        self._predict_bar: bool = False
        self._setup_started_at: float = time.perf_counter()
        for running_stage in self._batch_counts:
            self._batch_counts[running_stage] = 0
            self._stage_started_at[running_stage] = 0.0
            self._live_metrics[running_stage] = {}
        self._renderer.write_permanent(f"syntheticmind · {stage}")

    @override
    def teardown(self, trainer: syntheticmind.Trainer, module: Module, stage: TrainerStage) -> None:
        # Flushes a pending fit epoch and writes one terminal stage summary unless
        # exception handling already finalized the stage.
        if not self._rendering or self._stage_finalized or self._stage != stage:
            return
        if stage == "fit":
            self._write_pending_epoch()
        self._finalize_stage(trainer, stage)

    @override
    def on_train_epoch_start(self, trainer: syntheticmind.Trainer, module: Module) -> None:
        # Commits the previous epoch summary, resets training counters, and opens
        # the live line for the new epoch.
        if not self._rendering:
            return
        self._write_pending_epoch()
        self._train_bar: bool = True
        self._batch_counts["training"] = 0
        self._stage_started_at["training"] = time.perf_counter()
        self._live_metrics["training"] = {}
        self._render_training(trainer, module, force=True)

    @override
    def on_train_batch_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int
    ) -> None:
        # Advances the training batch view and captures fresh on-step progress
        # metrics; known final batches force immediate rendering.
        if not self._train_bar:
            return
        self._batch_counts["training"] = batch_idx + 1
        self._live_metrics["training"] = self._capture_metrics(trainer, module)
        total: int | None = self._raw_total(trainer, "training")
        force: bool = total is not None and batch_idx + 1 >= total
        self._render_training(trainer, module, force=force)

    @override
    def on_train_epoch_end(self, trainer: syntheticmind.Trainer, module: Module) -> None:
        # Freezes one compact epoch summary but defers its permanent write until
        # validation has had the opportunity to publish final metrics.
        if self._train_bar:
            if self._raw_total(trainer, "training") is None:
                self._render_training(trainer, module, force=True)
            metrics: dict[str, float] = self._epoch_metrics(trainer)
            duration: float = self._epoch_duration(trainer)
            parts: list[str] = [
                f"✓ {self._epoch_label(trainer, module)}"
            ]
            parts.extend(self._renderer.metric_parts(metrics))
            parts.append(self._renderer.format_duration(duration))
            self._pending_epoch_line: str | None = " · ".join(parts)
            self._pending_epoch_number: int | None = module.current_epoch + 1
            self._train_bar: bool = False

    @override
    def on_validation_epoch_start(self, trainer: syntheticmind.Trainer, module: Module) -> None:
        # Opens a live validation view for real passes; sanity validation remains
        # deliberately invisible to ordinary experiment output.
        if not self._rendering or trainer.sanity_checking:
            return
        self._val_bar: bool = True
        self._batch_counts["validating"] = 0
        self._stage_started_at["validating"] = time.perf_counter()
        self._live_metrics["validating"] = {}
        self._render_evaluation(trainer, module, "validating", force=True)

    @override
    def on_validation_batch_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Counts across all validation loaders and renders their combined total.
        if self._val_bar:
            self._batch_counts["validating"] += 1
            self._live_metrics["validating"] = self._capture_metrics(trainer, module)
            total: int | None = self._raw_total(trainer, "validating")
            force: bool = total is not None and self._batch_counts["validating"] >= total
            self._render_evaluation(trainer, module, "validating", force=force)

    @override
    def on_validation_epoch_end(self, trainer: syntheticmind.Trainer, module: Module) -> None:
        # Forces a terminal refresh when loader length is unknown.
        if self._val_bar:
            if self._raw_total(trainer, "validating") is None:
                self._render_evaluation(trainer, module, "validating", force=True)

    @override
    def on_validation_end(self, trainer: syntheticmind.Trainer, module: Module) -> None:
        # Captures the reduced validation metric view and restores the training
        # line after mid-epoch validation.
        if not self._val_bar:
            return
        self._live_metrics["validating"] = trainer._get_progress_bar_metrics("validating")
        if trainer.state.stage == "fit":
            if trainer._is_mid_epoch_validation:
                self._render_training(trainer, module, force=True)
        self._val_bar: bool = False

    @override
    def on_test_epoch_start(self, trainer: syntheticmind.Trainer, module: Module) -> None:
        # Opens the standalone test progress view.
        if not self._rendering:
            return
        self._test_bar: bool = True
        self._batch_counts["testing"] = 0
        self._stage_started_at["testing"] = time.perf_counter()
        self._live_metrics["testing"] = {}
        self._render_evaluation(trainer, module, "testing", force=True)

    @override
    def on_test_batch_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Advances combined test-loader progress and metric display.
        if self._test_bar:
            self._batch_counts["testing"] += 1
            self._live_metrics["testing"] = self._capture_metrics(trainer, module)
            total: int | None = self._raw_total(trainer, "testing")
            force: bool = total is not None and self._batch_counts["testing"] >= total
            self._render_evaluation(trainer, module, "testing", force=force)

    @override
    def on_test_epoch_end(self, trainer: syntheticmind.Trainer, module: Module) -> None:
        # Forces a terminal test refresh for unknown-length loaders.
        if self._test_bar:
            if self._raw_total(trainer, "testing") is None:
                self._render_evaluation(trainer, module, "testing", force=True)

    @override
    def on_test_end(self, trainer: syntheticmind.Trainer, module: Module) -> None:
        # Retains the completed reduced metric view for the stage summary.
        if not self._test_bar:
            return
        self._live_metrics["testing"] = trainer._get_progress_bar_metrics("testing")
        self._test_bar: bool = False

    @override
    def on_predict_start(self, trainer: syntheticmind.Trainer, module: Module) -> None:
        # Opens prediction progress before its epoch and batch hooks.
        if not self._rendering:
            return
        self._predict_bar: bool = True
        self._batch_counts["predicting"] = 0
        self._stage_started_at["predicting"] = time.perf_counter()
        self._render_evaluation(trainer, module, "predicting", force=True)

    @override
    def on_predict_batch_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        outputs: ModelOutput,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Advances combined prediction-loader progress without assuming metrics.
        if not self._predict_bar:
            return
        self._batch_counts["predicting"] += 1
        total: int | None = self._raw_total(trainer, "predicting")
        force: bool = total is not None and self._batch_counts["predicting"] >= total
        self._render_evaluation(trainer, module, "predicting", force=force)

    @override
    def on_predict_end(self, trainer: syntheticmind.Trainer, module: Module) -> None:
        # Forces the final unknown-length prediction refresh and closes its view.
        if not self._predict_bar:
            return
        if self._raw_total(trainer, "predicting") is None:
            self._render_evaluation(trainer, module, "predicting", force=True)
        self._predict_bar: bool = False

    @override
    def on_exception(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        exception: BaseException
    ) -> None:
        # Replaces the live line with one permanent interruption or failure
        # record and prevents ordinary teardown from claiming success.
        if not trainer.is_global_zero or not self._enabled:
            return
        if not self._rendering:
            self._renderer.reset()
        self._rank_zero: bool = True
        self._rendering: bool = True
        stage: TrainerStage = trainer.state.stage if trainer.state.stage is not None else "fit"
        message: str = " ".join(str(exception).split())
        detail: str = (
            type(exception).__name__
            if not message
            else f"{type(exception).__name__}: {message}"
        )
        detail: str = detail[:200]
        if (
            self._pending_epoch_number is not None
            and module.current_epoch >= self._pending_epoch_number
        ):
            self._write_pending_epoch()
        else:
            self._pending_epoch_line: str | None = None
            self._pending_epoch_number: int | None = None
        if isinstance(exception, KeyboardInterrupt):
            self._renderer.write_permanent(f"✗ {stage} interrupted")
        else:
            self._renderer.write_permanent(f"✗ {stage} failed · {detail}")
        self._stage: TrainerStage | None = stage
        self._stage_finalized: bool = True

    def _render_training(self, trainer: syntheticmind.Trainer, module: Module, force: bool) -> None:
        # Assembles the live epoch, progress, estimate, and bounded metric view;
        # ConsoleRenderer decides whether a non-forced update reaches the stream.
        if not self._rendering:
            return
        completed: int = self._batch_counts["training"]
        total: int | None = self._raw_total(trainer, "training")
        parts: list[str] = [
            f"{self._renderer.next_spinner()} {self._epoch_label(trainer, module)}"
        ]
        parts.extend(self._progress_parts(trainer, "training", completed, total))
        parts.extend(self._renderer.metric_parts(self._live_metrics["training"]))
        self._renderer.write_live(" · ".join(parts), force)

    def _write_pending_epoch(self) -> None:
        # Commits the deferred epoch line exactly once.
        if self._pending_epoch_line is None:
            return
        self._renderer.write_permanent(self._pending_epoch_line)
        self._pending_epoch_line: str | None = None
        self._pending_epoch_number: int | None = None

    def _render_evaluation(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        running_stage: RunningStage,
        force: bool
    ) -> None:
        # Assembles a stage-aware evaluation line and includes the fit epoch when
        # validation is nested inside training.
        if not self._rendering:
            return
        completed: int = self._batch_counts[running_stage]
        total: int | None = self._raw_total(trainer, running_stage)
        label: str = {
            "validating": "validating",
            "testing": "testing",
            "predicting": "predicting",
            "training": "training",
            "sanity_checking": "sanity checking"
        }[running_stage]
        parts: list[str] = [f"{self._renderer.next_spinner()} {label}"]
        if running_stage == "validating" and trainer.state.stage == "fit":
            parts.append(self._epoch_label(trainer, module))
        parts.extend(self._progress_parts(trainer, running_stage, completed, total))
        parts.extend(self._renderer.metric_parts(self._live_metrics[running_stage]))
        self._renderer.write_live(" · ".join(parts), force)

    def _progress_parts(
        self,
        trainer: syntheticmind.Trainer,
        running_stage: RunningStage,
        completed: int,
        total: int | None
    ) -> list[str]:
        # Formats a count for iterable datasets and percentage plus ETA for
        # finite loaders once enough timing evidence exists.
        raw_total: int | None = self._raw_total(trainer, running_stage)
        if raw_total is None:
            return [f"{completed} batches"]
        if total is None or total <= 0:
            return []
        percentage: int = min(100, int(completed * 100 / total))
        parts: list[str] = [f"{percentage}%"]
        elapsed: float = self._stage_elapsed(trainer, running_stage)
        if completed >= 2 and completed < total and elapsed >= 0.5:
            remaining: float = elapsed / completed * (total - completed)
            parts.append(f"{self._renderer.format_duration(remaining)} left")
        return parts

    def _capture_metrics(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> dict[str, float]:
        # Selects the first two on-step Module.log records routed to the progress
        # bar and reduces them through the active Strategy.
        captured: dict[str, float] = {}
        for metric in module.logged_metrics:
            if not metric.progress_bar or not metric.on_step:
                continue
            value: torch.Tensor | None = metric.compute_step(
                module.device,
                trainer.strategy.reduce
            )
            if value is None:
                continue
            captured[metric.output_name(on_step=True)] = module.trainer._metric_to_float(value)
            if len(captured) == 2:
                break
        return captured

    def _epoch_metrics(self, trainer: syntheticmind.Trainer) -> dict[str, float]:
        # Selects a compact training-and-validation epoch view while preserving
        # insertion order from the trainer's reduced metric mappings.
        training_metrics: dict[str, float] = trainer._get_progress_bar_metrics("training")
        validation_metrics: dict[str, float] = trainer._get_progress_bar_metrics("validating")
        selected: dict[str, float] = {}
        for metrics in (training_metrics, validation_metrics):
            if metrics:
                name: str = next(iter(metrics))
                selected[name] = metrics[name]
        for metrics in (training_metrics, validation_metrics):
            for name, value in metrics.items():
                if name not in selected:
                    selected[name] = value
                if len(selected) == 2:
                    return selected
        return selected

    def _finalize_stage(self, trainer: syntheticmind.Trainer, stage: TrainerStage) -> None:
        # Writes one success record with the best available reduced metrics,
        # timer diagnosis, and end-to-end stage duration.
        if not self._rendering or self._stage_finalized:
            return
        metrics: dict[str, float]
        if stage == "fit":
            validation_metrics: dict[str, float] = trainer._get_progress_bar_metrics("validating")
            metrics: dict[str, float] = (
                validation_metrics
                or trainer._get_progress_bar_metrics("training")
            )
        elif stage == "validate":
            metrics: dict[str, float] = (
                {}
                if trainer.validate_loop.verbose
                else trainer._get_progress_bar_metrics("validating")
            )
        elif stage == "test":
            metrics: dict[str, float] = (
                {}
                if trainer.test_loop.verbose
                else trainer._get_progress_bar_metrics("testing")
            )
        else:
            metrics: dict[str, float] = {}
        parts: list[str] = [f"✓ {stage} complete"]
        timer: Timer | None = self._timer(trainer)
        if stage == "fit" and timer is not None and timer.time_limit_reached:
            parts.append("max_time reached")
        parts.extend(self._renderer.metric_parts(metrics))
        parts.append(self._renderer.format_duration(self._stage_duration(trainer, stage)))
        self._renderer.write_permanent(" · ".join(parts))
        self._stage_finalized: bool = True

    def _raw_total(self, trainer: syntheticmind.Trainer, running_stage: RunningStage) -> int | None:
        # Reads authoritative batch totals from DataConnector and CombinedLoader
        # without consuming client iterators or inventing iterable lengths.
        if running_stage == "training":
            return trainer._data_connector.total_train_batches
        if running_stage == "validating":
            combined_loader: CombinedLoader | None = (
                trainer._data_connector.val_combined_loader
            )
            return None if combined_loader is None else combined_loader._length()
        if running_stage == "testing":
            combined_loader: CombinedLoader | None = (
                trainer._data_connector.test_combined_loader
            )
            return None if combined_loader is None else combined_loader._length()
        if running_stage == "predicting":
            combined_loader: CombinedLoader | None = (
                trainer._data_connector.predict_combined_loader
            )
            return None if combined_loader is None else combined_loader._length()
        return None

    def _stage_elapsed(self, trainer: syntheticmind.Trainer, running_stage: RunningStage) -> float:
        # Prefers the shared Timer clock and otherwise falls back to the callback's
        # local monotonic stage origin.
        timer: Timer | None = self._timer(trainer)
        if running_stage == "training" and timer is not None:
            return timer._train_epoch_time_elapsed()
        started_at: float = self._stage_started_at[running_stage]
        if started_at <= 0.0:
            return 0.0
        return time.perf_counter() - started_at

    def _epoch_duration(self, trainer: syntheticmind.Trainer) -> float:
        # Uses Timer's end-to-end epoch view when available.
        timer: Timer | None = self._timer(trainer)
        if timer is not None:
            return timer._epoch_time_elapsed()
        return self._stage_elapsed(trainer, "training")

    def _stage_duration(self, trainer: syntheticmind.Trainer, stage: TrainerStage) -> float:
        # Resolves a completed shared stage duration or a local setup fallback.
        timer: Timer | None = self._timer(trainer)
        if timer is not None:
            duration: float = timer.time_elapsed(stage)
            if duration > 0.0:
                return duration
        if self._setup_started_at <= 0.0:
            return 0.0
        return time.perf_counter() - self._setup_started_at

    def _timer(self, trainer: syntheticmind.Trainer) -> Timer | None:
        # Discovers the canonical Timer callback without retaining a stale
        # cross-run reference.
        for callback in trainer.callbacks:
            if isinstance(callback, Timer):
                return callback
        return None

    def _epoch_label(self, trainer: syntheticmind.Trainer, module: Module) -> str:
        # Formats bounded and unbounded epoch plans without exposing zero-based
        # internal counters to clients.
        epoch: int = module.current_epoch + 1
        return (
            f"epoch {epoch}"
            if trainer.max_epochs == -1
            else f"epoch {epoch}/{trainer.max_epochs}"
        )
