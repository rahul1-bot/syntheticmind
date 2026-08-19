# This module:
# 1. Renders terminal progress bars for training epochs, validation passes,
#    and test passes, with live metric postfixes on the training and
#    evaluation bars
#
# Design decisions:
# - Bars are created only on rank zero, so distributed runs render exactly one
#   set of bars; on other ranks every handler observes an absent bar and
#   returns immediately
# - The training bar's postfix shows only metrics whose logging configuration
#   requested progress-bar visibility, read live from the module's logging
#   buffer before the loop drains it, which works because the loops invoke
#   callback batch-end hooks before metric dispatch
# - Bar totals come from the trainer's cached batch counts and may be None
#   for iterable datasets, which tqdm renders as an open-ended counter
# - The training and test bars persist on screen as an epoch record while the
#   validation bar clears itself, keeping epoch summaries readable
# - Evaluation bars stamp their final reduced metrics as the closing postfix,
#   leaving the last rendered state meaningful
#
# Author: Rahul Sawhney

from __future__ import annotations

import builtins
import time
from typing import IO, TYPE_CHECKING, override

import torch

from syntheticmind.callbacks.callback import Callback
from syntheticmind.callbacks.timer import Timer
from syntheticmind.core.module import Module
from syntheticmind.utilities.console import ConsoleRenderer
from syntheticmind.utilities.distributed import is_rank_zero
from syntheticmind.utilities.types import (
    Batch,
    ModelOutput,
    RunningStage,
    StepOutput,
    TrainerStage
)

if TYPE_CHECKING:
    from syntheticmind.core.trainer import Trainer

__all__: list[str] = ["ProgressBarCallback"]


class ProgressBarCallback(Callback):
    # Terminal progress rendering over the training, validation, and test
    # loops. One bar exists per active stage on rank zero; batch-end hooks
    # advance it and epoch-end hooks close it.
    def __init__(self) -> None:
        # Prepares the three bar slots; each is created at its stage's epoch
        # start and cleared again when the stage ends.
        super().__init__()
        self._train_bar: bool | None = None  # type: ignore[type-arg]
        self._val_bar: bool | None = None  # type: ignore[type-arg]
        self._test_bar: bool | None = None  # type: ignore[type-arg]
        self._predict_bar: bool | None = None
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
        return self._enabled

    @property
    def is_disabled(self) -> bool:
        return not self._enabled

    def disable(self) -> None:
        self._enabled = False
        self._rendering = False
        self._renderer.clear_live()

    def enable(self) -> None:
        self._enabled = True
        self._rendering = self._rank_zero

    def print(
        self,
        *args: object,
        sep: str | None = " ",
        end: str | None = "\n",
        file: IO[str] | None = None,
        flush: bool = False
    ) -> None:
        self._renderer.clear_live()
        builtins.print(*args, sep=sep, end=end, file=file, flush=flush)

    @override
    def setup(self, trainer: Trainer, module: Module, stage: TrainerStage) -> None:
        self._rank_zero = is_rank_zero()
        self._rendering = self._rank_zero and self._enabled
        if not self._rendering:
            return
        self._renderer.reset()
        self._stage = stage
        self._stage_finalized = False
        self._pending_epoch_line = None
        self._pending_epoch_number = None
        self._train_bar = None
        self._val_bar = None
        self._test_bar = None
        self._predict_bar = None
        self._setup_started_at = time.perf_counter()
        for running_stage in self._batch_counts:
            self._batch_counts[running_stage] = 0
            self._stage_started_at[running_stage] = 0.0
            self._live_metrics[running_stage] = {}
        self._renderer.write_permanent(f"syntheticmind · {stage}")

    @override
    def teardown(self, trainer: Trainer, module: Module, stage: TrainerStage) -> None:
        if not self._rendering or self._stage_finalized or self._stage != stage:
            return
        if stage == "fit":
            self._write_pending_epoch()
        self._finalize_stage(trainer, stage)

    @override
    def on_train_epoch_start(self, trainer: Trainer, module: Module) -> None:
        # Opens the persistent training bar for the epoch, labeled with the
        # epoch position and sized from the trainer's cached batch total.
        if not self._rendering:
            return
        self._write_pending_epoch()
        self._train_bar = True
        self._batch_counts["training"] = 0
        self._stage_started_at["training"] = time.perf_counter()
        self._live_metrics["training"] = {}
        self._render_training(trainer, module, force=True)

    @override
    def on_train_batch_end(
        self, trainer: Trainer, module: Module, outputs: StepOutput, batch: Batch, batch_idx: int
    ) -> None:
        # Advances the training bar and refreshes its postfix with the
        # progress-bar-visible metrics currently buffered on the module,
        # which the loop has not yet drained at this hook.
        if not self._train_bar:
            return
        self._batch_counts["training"] = batch_idx + 1
        self._live_metrics["training"] = self._capture_metrics(module)
        total: int | None = self._effective_total(trainer, "training")
        force: bool = total is not None and batch_idx + 1 >= total
        self._render_training(trainer, module, force=force)

    @override
    def on_train_epoch_end(self, trainer: Trainer, module: Module) -> None:
        # Closes the training bar at the epoch boundary, leaving its final
        # state on screen as the epoch record.
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
            self._pending_epoch_line = " · ".join(parts)
            self._pending_epoch_number = module.current_epoch + 1
            self._train_bar = None

    @override
    def on_validation_epoch_start(self, trainer: Trainer, module: Module) -> None:
        # Opens the transient validation bar sized from the cached validation
        # batch total; it clears itself from the terminal when closed.
        if not self._rendering or trainer.sanity_checking:
            return
        self._val_bar = True
        self._batch_counts["validating"] = 0
        self._stage_started_at["validating"] = time.perf_counter()
        self._live_metrics["validating"] = {}
        self._render_evaluation(trainer, module, "validating", force=True)

    @override
    def on_validation_batch_end(
        self, trainer: Trainer, module: Module, outputs: StepOutput, batch: Batch, batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Advances the validation bar by one batch.
        if self._val_bar:
            self._batch_counts["validating"] += 1
            self._live_metrics["validating"] = self._capture_metrics(module)
            total: int | None = self._effective_total(trainer, "validating")
            force: bool = total is not None and self._batch_counts["validating"] >= total
            self._render_evaluation(trainer, module, "validating", force=force)

    @override
    def on_validation_epoch_end(self, trainer: Trainer, module: Module) -> None:
        # Stamps the reduced validation metrics as the closing postfix and
        # closes the bar.
        if self._val_bar:
            if self._raw_total(trainer, "validating") is None:
                self._render_evaluation(trainer, module, "validating", force=True)

    @override
    def on_validation_end(self, trainer: Trainer, module: Module) -> None:
        if not self._val_bar:
            return
        self._live_metrics["validating"] = trainer._get_progress_bar_metrics("validating")
        if trainer.state.stage == "fit":
            if trainer._is_mid_epoch_validation:
                self._render_training(trainer, module, force=True)
        self._val_bar = None

    @override
    def on_test_epoch_start(self, trainer: Trainer, module: Module) -> None:
        # Opens the persistent test bar sized from the cached test batch
        # total.
        if not self._rendering:
            return
        self._test_bar = True
        self._batch_counts["testing"] = 0
        self._stage_started_at["testing"] = time.perf_counter()
        self._live_metrics["testing"] = {}
        self._render_evaluation(trainer, module, "testing", force=True)

    @override
    def on_test_batch_end(
        self, trainer: Trainer, module: Module, outputs: StepOutput, batch: Batch, batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Advances the test bar by one batch.
        if self._test_bar:
            self._batch_counts["testing"] += 1
            self._live_metrics["testing"] = self._capture_metrics(module)
            total: int | None = self._effective_total(trainer, "testing")
            force: bool = total is not None and self._batch_counts["testing"] >= total
            self._render_evaluation(trainer, module, "testing", force=force)

    @override
    def on_test_epoch_end(self, trainer: Trainer, module: Module) -> None:
        # Stamps the reduced test metrics as the closing postfix and closes
        # the bar.
        if self._test_bar:
            if self._raw_total(trainer, "testing") is None:
                self._render_evaluation(trainer, module, "testing", force=True)

    @override
    def on_test_end(self, trainer: Trainer, module: Module) -> None:
        if not self._test_bar:
            return
        self._live_metrics["testing"] = trainer._get_progress_bar_metrics("testing")
        self._test_bar = None

    @override
    def on_predict_start(self, trainer: Trainer, module: Module) -> None:
        if not self._rendering:
            return
        self._predict_bar = True
        self._batch_counts["predicting"] = 0
        self._stage_started_at["predicting"] = time.perf_counter()
        self._render_evaluation(trainer, module, "predicting", force=True)

    @override
    def on_predict_batch_end(
        self,
        trainer: Trainer,
        module: Module,
        outputs: ModelOutput,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        if not self._predict_bar:
            return
        self._batch_counts["predicting"] += 1
        total: int | None = self._effective_total(trainer, "predicting")
        force: bool = total is not None and self._batch_counts["predicting"] >= total
        self._render_evaluation(trainer, module, "predicting", force=force)

    @override
    def on_predict_end(self, trainer: Trainer, module: Module) -> None:
        if not self._predict_bar:
            return
        if self._raw_total(trainer, "predicting") is None:
            self._render_evaluation(trainer, module, "predicting", force=True)
        self._predict_bar = None

    @override
    def on_exception(self, trainer: Trainer, module: Module, exception: BaseException) -> None:
        if not is_rank_zero() or not self._enabled:
            return
        if not self._rendering:
            self._renderer.reset()
        self._rank_zero = True
        self._rendering = True
        stage: TrainerStage = trainer.state.stage if trainer.state.stage is not None else "fit"
        message: str = " ".join(str(exception).split())
        detail: str = type(exception).__name__ if not message else f"{type(exception).__name__}: {message}"
        detail = detail[:200]
        if (
            self._pending_epoch_number is not None
            and module.current_epoch >= self._pending_epoch_number
        ):
            self._write_pending_epoch()
        else:
            self._pending_epoch_line = None
            self._pending_epoch_number = None
        if isinstance(exception, KeyboardInterrupt):
            self._renderer.write_permanent(f"✗ {stage} interrupted")
        else:
            self._renderer.write_permanent(f"✗ {stage} failed · {detail}")
        self._stage = stage
        self._stage_finalized = True

    def _render_training(self, trainer: Trainer, module: Module, force: bool) -> None:
        if not self._rendering:
            return
        completed: int = self._batch_counts["training"]
        total: int | None = self._effective_total(trainer, "training")
        parts: list[str] = [
            f"{self._renderer.next_spinner()} {self._epoch_label(trainer, module)}"
        ]
        parts.extend(self._progress_parts(trainer, "training", completed, total))
        parts.extend(self._renderer.metric_parts(self._live_metrics["training"]))
        self._renderer.write_live(" · ".join(parts), force)

    def _write_pending_epoch(self) -> None:
        if self._pending_epoch_line is None:
            return
        self._renderer.write_permanent(self._pending_epoch_line)
        self._pending_epoch_line = None
        self._pending_epoch_number = None

    def _render_evaluation(
        self,
        trainer: Trainer,
        module: Module,
        running_stage: RunningStage,
        force: bool
    ) -> None:
        if not self._rendering:
            return
        completed: int = self._batch_counts[running_stage]
        total: int | None = self._effective_total(trainer, running_stage)
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
        trainer: Trainer,
        running_stage: RunningStage,
        completed: int,
        total: int | None
    ) -> list[str]:
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

    def _capture_metrics(self, module: Module) -> dict[str, float]:
        captured: dict[str, float] = {}
        for metric in module.logged_metrics:
            if not metric.progress_bar or not metric.on_step:
                continue
            value: torch.Tensor | None = metric.compute_step(module.device)
            if value is None:
                continue
            captured[metric.output_name(on_step=True)] = module.trainer._metric_to_float(value)
            if len(captured) == 2:
                break
        return captured

    def _epoch_metrics(self, trainer: Trainer) -> dict[str, float]:
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

    def _finalize_stage(self, trainer: Trainer, stage: TrainerStage) -> None:
        if not self._rendering or self._stage_finalized:
            return
        metrics: dict[str, float]
        if stage == "fit":
            validation_metrics: dict[str, float] = trainer._get_progress_bar_metrics("validating")
            metrics = validation_metrics or trainer._get_progress_bar_metrics("training")
        elif stage == "validate":
            metrics = (
                {}
                if trainer.validate_loop.verbose
                else trainer._get_progress_bar_metrics("validating")
            )
        elif stage == "test":
            metrics = (
                {}
                if trainer.test_loop.verbose
                else trainer._get_progress_bar_metrics("testing")
            )
        else:
            metrics = {}
        parts: list[str] = [f"✓ {stage} complete"]
        timer: Timer | None = self._timer(trainer)
        if stage == "fit" and timer is not None and timer.time_limit_reached:
            parts.append("max_time reached")
        parts.extend(self._renderer.metric_parts(metrics))
        parts.append(self._renderer.format_duration(self._stage_duration(trainer, stage)))
        self._renderer.write_permanent(" · ".join(parts))
        self._stage_finalized = True

    def _effective_total(self, trainer: Trainer, running_stage: RunningStage) -> int | None:
        raw_total: int | None = self._raw_total(trainer, running_stage)
        return raw_total

    def _raw_total(self, trainer: Trainer, running_stage: RunningStage) -> int | None:
        if running_stage == "training":
            return trainer._total_train_batches
        if running_stage == "validating":
            combined_loader = trainer._val_combined_loader
            return None if combined_loader is None else combined_loader._length()
        if running_stage == "testing":
            combined_loader = trainer._test_combined_loader
            return None if combined_loader is None else combined_loader._length()
        if running_stage == "predicting":
            combined_loader = trainer._predict_combined_loader
            return None if combined_loader is None else combined_loader._length()
        return None

    def _stage_elapsed(self, trainer: Trainer, running_stage: RunningStage) -> float:
        timer: Timer | None = self._timer(trainer)
        if running_stage == "training" and timer is not None:
            return timer._train_epoch_time_elapsed()
        started_at: float = self._stage_started_at[running_stage]
        if started_at <= 0.0:
            return 0.0
        return time.perf_counter() - started_at

    def _epoch_duration(self, trainer: Trainer) -> float:
        timer: Timer | None = self._timer(trainer)
        if timer is not None:
            return timer._epoch_time_elapsed()
        return self._stage_elapsed(trainer, "training")

    def _stage_duration(self, trainer: Trainer, stage: TrainerStage) -> float:
        timer: Timer | None = self._timer(trainer)
        if timer is not None:
            duration: float = timer.time_elapsed(stage)
            if duration > 0.0:
                return duration
        if self._setup_started_at <= 0.0:
            return 0.0
        return time.perf_counter() - self._setup_started_at

    def _timer(self, trainer: Trainer) -> Timer | None:
        for callback in trainer.callbacks:
            if isinstance(callback, Timer):
                return callback
        return None

    def _epoch_label(self, trainer: Trainer, module: Module) -> str:
        epoch: int = module.current_epoch + 1
        return f"epoch {epoch}" if trainer.max_epochs == -1 else f"epoch {epoch}/{trainer.max_epochs}"
