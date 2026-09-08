# This module:
# 1. Defines ProgressBarCallback, the rank-zero lifecycle renderer for the run
#    header, fit, validation, testing, prediction, epoch summaries, and failure
#    termination
# 2. Selects bounded progress metrics, throughput, learning rate, and timing
#    information from the live trainer state without owning training or
#    metric-reduction semantics
#
# Design decisions:
# - Rendering is enabled only on global zero, preventing distributed workers
#   from racing over one terminal while every rank still receives hook calls
# - ConsoleRenderer owns ANSI, stream, and diagnostics-capture mechanics; this
#   callback owns ML lifecycle state, stage labels, batch counts, metric
#   selection, timing, and the stage boundaries at which capture begins and ends
# - Live updates are throttled by ConsoleRenderer, while terminal batch and
#   unknown-length boundaries force a final refresh so displayed state cannot
#   lag behind completed computation
# - One run header records accelerator, topology, precision, seed, run ceilings,
#   resume path, and logger directory from Trainer views that require no
#   collective, so distributed setup remains symmetric across ranks
# - The epoch summary is written at on_train_epoch_end because scheduled
#   validation and epoch metric reduction both complete before that hook; it
#   prefers epoch-reduced names over on-step names and adds a fit estimate
#   derived from completed epoch durations only when the epoch ceiling is finite
# - At most max_metrics progress metrics are rendered to preserve terminal
#   readability; the complete metric set remains available through loggers and
#   Trainer views. Learning rate is read from the first optimizer parameter
#   group rather than from any monitoring callback
# - Diagnostics capture begins at setup and is released at teardown and at
#   failure, so loguru records and Python warnings emitted during a stage reach
#   the terminal through the renderer instead of splicing into the live line
# - Client print calls clear the live line before delegating to builtins.print,
#   preventing user output from corrupting the renderer's terminal state
# - Failure rendering reports epoch, global step, and active batch coordinates,
#   truncates normalized exception text, and marks the stage as finalized so
#   teardown cannot overwrite the original failure diagnosis
#
# Author: Rahul Sawhney

import builtins
import time
from pathlib import Path
from typing import TextIO, override

import syntheticmind
import torch

from syntheticmind.callbacks.callback import Callback
from syntheticmind.callbacks.timer import Timer
from syntheticmind.data.combined_loader import CombinedLoader
from syntheticmind.core.module import Module
from syntheticmind.loggers.logger import Logger
from syntheticmind.trainer.console import ConsoleRenderer
from syntheticmind.primitives.types import (
    Batch,
    ModelOutput,
    OptimizerParameterGroup,
    OptimizerStateValue,
    RunningStage,
    StepOutput,
    TrainerStage
)

__all__: list[str] = ["ProgressBarCallback"]

type _PrintValue = ModelOutput | Path | BaseException | torch.nn.Module


class ProgressBarCallback(Callback):
    # Default terminal observer installed by Trainer when progress rendering is
    # enabled. Clients may subclass it to replace presentation while retaining
    # the same callback lifecycle, configure its density and cadence, or disable
    # and re-enable it at runtime.
    #
    # Example:
    #     trainer = Trainer(
    #         callbacks=[ProgressBarCallback(max_metrics=6, redirected_interval=30.0)]
    #     )
    def __init__(
        self,
        refresh_interval: float = 0.08,
        redirected_interval: float | None = 60.0,
        max_metrics: int = 4,
        capture_diagnostics: bool = True
    ) -> None:
        # Initializes detached rendering state; rank and stage ownership are
        # resolved only when setup receives the active Trainer.
        #
        # Args:
        #     refresh_interval: Minimum seconds between interactive live
        #         rewrites. Default: ``0.08``.
        #     redirected_interval: Minimum seconds between permanent progress
        #         heartbeats when stderr is redirected, or None to emit only
        #         epoch and stage records. Default: ``60.0``.
        #     max_metrics: Maximum progress-routed metrics shown on live, epoch,
        #         and stage lines. Default: ``4``.
        #     capture_diagnostics: Whether loguru records and Python warnings
        #         route through the renderer while a stage renders.
        #         Default: ``True``.
        #
        # Raises:
        #     ValueError: If max_metrics is not a positive integer or an interval
        #         is negative.
        #     TypeError: If capture_diagnostics is not a bool.
        if type(max_metrics) is not int or max_metrics < 1:
            raise ValueError(f"`max_metrics` must be a positive int, got {max_metrics!r}.")
        self._train_bar: bool = False
        self._val_bar: bool = False
        self._test_bar: bool = False
        self._predict_bar: bool = False
        self._renderer: ConsoleRenderer = ConsoleRenderer(
            refresh_interval,
            redirected_interval,
            capture_diagnostics
        )
        self._max_metrics: int = max_metrics
        self._enabled: bool = True
        self._rendering: bool = False
        self._rank_zero: bool = False
        self._stage: TrainerStage | None = None
        self._stage_finalized: bool = False
        self._setup_started_at: float = 0.0
        self._epoch_durations: list[float] = []
        self._epoch_metric_names: set[str] = set()
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

    @property
    def renderer(self) -> ConsoleRenderer:
        # Exposes the one renderer that owns this process's live line so Trainer
        # stage markers and result records clear it instead of writing over it.
        return self._renderer

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
        file: TextIO | None = None,
        flush: bool = False
    ) -> None:
        # Prints client output without interleaving it with an active progress
        # line, preserving the ordinary builtins.print argument contract.
        self._renderer.clear_live()
        builtins.print(*args, sep=sep, end=end, file=file, flush=flush)

    @override
    def setup(self, trainer: syntheticmind.Trainer, module: Module, stage: TrainerStage) -> None:
        # Resolves rank ownership, resets all stage-local views, captures
        # diagnostics output, and emits the run header before lifecycle hooks
        # begin.
        self._rank_zero: bool = trainer.is_global_zero
        self._rendering: bool = self._rank_zero and self._enabled
        if not self._rendering:
            return
        self._renderer.reset()
        self._stage: TrainerStage | None = stage
        self._stage_finalized: bool = False
        self._train_bar: bool = False
        self._val_bar: bool = False
        self._test_bar: bool = False
        self._predict_bar: bool = False
        self._setup_started_at: float = time.perf_counter()
        self._epoch_durations: list[float] = []
        self._epoch_metric_names: set[str] = set()
        for running_stage in self._batch_counts:
            self._batch_counts[running_stage] = 0
            self._stage_started_at[running_stage] = 0.0
            self._live_metrics[running_stage] = {}
        self._renderer.capture_diagnostics()
        self._write_header(trainer, stage)

    @override
    def teardown(self, trainer: syntheticmind.Trainer, module: Module, stage: TrainerStage) -> None:
        # Restores diagnostics output and writes one terminal stage summary unless
        # exception handling already finalized the stage.
        if self._stage != stage:
            return
        self._renderer.release_diagnostics()
        if not self._rendering or self._stage_finalized:
            return
        self._finalize_stage(trainer, stage)

    @override
    def on_train_epoch_start(self, trainer: syntheticmind.Trainer, module: Module) -> None:
        # Resets training counters and opens the live line for the new epoch.
        if not self._rendering:
            return
        self._train_bar: bool = True
        self._batch_counts["training"] = 0
        self._stage_started_at["training"] = time.perf_counter()
        self._live_metrics["training"] = {}
        self._epoch_metric_names: set[str] = set()
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
        # Writes one compact epoch summary. Scheduled validation and epoch metric
        # reduction have both completed by this hook, so the record can carry
        # epoch-reduced training metrics, validation metrics, epoch duration, and
        # a remaining-fit estimate without deferral.
        if not self._train_bar:
            return
        if self._raw_total(trainer, "training") is None:
            self._render_training(trainer, module, force=True)
        metrics: dict[str, float] = self._epoch_metrics(trainer)
        duration: float = self._epoch_duration(trainer)
        self._epoch_durations.append(duration)
        parts: list[str] = [
            f"{self._renderer.style('✓', 'green')} {self._epoch_label(trainer, module)}"
        ]
        parts.extend(self._renderer.metric_parts(metrics, self._max_metrics))
        parts.append(self._renderer.format_duration(duration))
        remaining: float | None = self._fit_time_remaining(trainer, module)
        if remaining is not None and remaining >= 1.0:
            parts.append(
                self._renderer.style(
                    f"~{self._renderer.format_duration(remaining)} left",
                    "dim"
                )
            )
        self._renderer.write_permanent(" · ".join(parts))
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
        # record carrying run coordinates, restores diagnostics output, and
        # prevents ordinary teardown from claiming success.
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
        parts: list[str] = [
            f"{self._renderer.style('✗', 'red')} {stage} "
            f"{'interrupted' if isinstance(exception, KeyboardInterrupt) else 'failed'}"
        ]
        parts.extend(self._failure_coordinates(trainer, module))
        if not isinstance(exception, KeyboardInterrupt):
            parts.append(detail)
        self._renderer.write_permanent(" · ".join(parts))
        self._stage: TrainerStage | None = stage
        self._stage_finalized: bool = True
        self._renderer.release_diagnostics()

    def _write_header(self, trainer: syntheticmind.Trainer, stage: TrainerStage) -> None:
        # Emits the stage title plus dim execution and run-policy records. Every
        # value is a rank-local Trainer view; the broadcast log_dir is avoided so
        # rank zero never enters a collective the other ranks do not reach.
        self._renderer.write_permanent(
            self._renderer.style(f"syntheticmind · {stage}", "bold")
        )
        execution: list[str] = [
            f"devices {trainer.num_devices}",
            f"nodes {trainer.num_nodes}",
            f"world {trainer.world_size}",
            f"strategy {type(trainer.strategy).__name__}",
            f"precision {trainer.precision}"
        ]
        seed: int | None = trainer._randomness.seed_value
        if seed is not None:
            execution.append(f"seed {seed}")
        self._renderer.write_record(
            f"accelerator {trainer.accelerator.name()}",
            execution,
            "dim"
        )
        policy: list[str] = []
        if stage == "fit":
            policy.append(
                "max epochs unbounded"
                if trainer.max_epochs == -1
                else f"max epochs {trainer.max_epochs}"
            )
            policy.append(
                "max steps unbounded"
                if trainer.max_steps == -1
                else f"max steps {trainer.max_steps}"
            )
            if trainer.accumulate_grad_batches > 1:
                policy.append(f"accumulate {trainer.accumulate_grad_batches}")
        if trainer.ckpt_path is not None:
            policy.append(f"checkpoint {trainer.ckpt_path}")
        logger: Logger | None = trainer.logger
        if logger is not None:
            policy.append(f"logs {logger.log_dir}")
        if policy:
            self._renderer.write_record(policy[0], policy[1:], "dim")

    def _render_training(self, trainer: syntheticmind.Trainer, module: Module, force: bool) -> None:
        # Assembles the live epoch, progress, estimate, throughput, bounded metric,
        # and learning-rate view; ConsoleRenderer decides whether a non-forced
        # update reaches the stream.
        if not self._rendering:
            return
        completed: int = self._batch_counts["training"]
        total: int | None = self._raw_total(trainer, "training")
        parts: list[str] = self._live_prefix(self._epoch_label(trainer, module))
        parts.extend(self._progress_parts(trainer, "training", completed, total))
        parts.extend(
            self._renderer.metric_parts(self._live_metrics["training"], self._max_metrics)
        )
        learning_rate: float | None = self._learning_rate(trainer)
        if learning_rate is not None:
            parts.append(f"lr {self._renderer.format_metric(learning_rate)}")
        self._renderer.write_live(" · ".join(parts), force)

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
        parts: list[str] = self._live_prefix(label)
        if running_stage == "validating" and trainer.state.stage == "fit":
            parts.append(self._epoch_label(trainer, module))
        parts.extend(self._progress_parts(trainer, running_stage, completed, total))
        parts.extend(
            self._renderer.metric_parts(self._live_metrics[running_stage], self._max_metrics)
        )
        self._renderer.write_live(" · ".join(parts), force)

    def _live_prefix(self, label: str) -> list[str]:
        # Prepends the spinner only on interactive terminals; redirected
        # heartbeats stay plain so log files carry no animation residue.
        if self._renderer.interactive:
            return [f"{self._renderer.spinner_frame()} {label}"]
        return [label]

    def _progress_parts(
        self,
        trainer: syntheticmind.Trainer,
        running_stage: RunningStage,
        completed: int,
        total: int | None
    ) -> list[str]:
        # Formats a count and rate for iterable datasets, and a glyph bar,
        # percentage, ETA, and rate for finite loaders once enough timing
        # evidence exists.
        raw_total: int | None = self._raw_total(trainer, running_stage)
        elapsed: float = self._stage_elapsed(trainer, running_stage)
        rate: str | None = self._renderer.format_rate(completed, elapsed)
        if raw_total is None:
            parts: list[str] = [f"{completed} batches"]
            if rate is not None:
                parts.append(rate)
            return parts
        if total is None or total <= 0:
            return []
        percentage: int = min(100, int(completed * 100 / total))
        progress: str = f"{percentage:>3d}%"
        if self._renderer.interactive:
            progress: str = f"{self._renderer.progress_bar(completed, total)} {progress}"
        parts: list[str] = [progress]
        if completed >= 2 and completed < total and elapsed >= 0.5:
            remaining: float = elapsed / completed * (total - completed)
            parts.append(f"{self._renderer.format_duration(remaining)} left")
        if rate is not None:
            parts.append(rate)
        return parts

    def _learning_rate(self, trainer: syntheticmind.Trainer) -> float | None:
        # Reads the first parameter group of the first configured optimizer.
        # Multi-optimizer and multi-group schedules remain the LearningRateMonitor
        # callback's responsibility; the live line shows one representative value.
        if not trainer.optimizers:
            return None
        groups: list[OptimizerParameterGroup] = trainer.optimizers[0].param_groups
        if not groups or "lr" not in groups[0]:
            return None
        candidate: OptimizerStateValue | torch.nn.Parameter | list[torch.nn.Parameter] = (
            groups[0]["lr"]
        )
        if isinstance(candidate, torch.Tensor) and candidate.numel() == 1:
            return trainer._metric_to_float(candidate)
        if isinstance(candidate, float | int) and not isinstance(candidate, bool):
            return float(candidate)
        return None

    def _capture_metrics(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> dict[str, float]:
        # Selects the first max_metrics on-step Module.log records routed to the
        # progress bar and reduces them through the active Strategy. Epoch-reduced
        # output names are recorded here because the Module buffer is drained
        # before on_train_epoch_end can inspect it.
        captured: dict[str, float] = {}
        for metric in module.logged_metrics:
            if not metric.progress_bar:
                continue
            if metric.on_epoch:
                self._epoch_metric_names.add(metric.output_name(on_step=False))
            if len(captured) >= self._max_metrics or not metric.on_step:
                continue
            value: torch.Tensor | None = metric.compute_step(
                module.device,
                trainer.strategy.reduce
            )
            if value is None:
                continue
            captured[metric.output_name(on_step=True)] = module.trainer._metric_to_float(value)
        return captured

    def _epoch_metrics(self, trainer: syntheticmind.Trainer) -> dict[str, float]:
        # Selects a compact training-and-validation epoch view. The first metric
        # of each stage leads so both stages stay visible under a small limit,
        # and epoch-reduced training names take precedence over on-step names.
        training_metrics: dict[str, float] = self._epoch_view(
            trainer._get_progress_bar_metrics("training")
        )
        validation_metrics: dict[str, float] = trainer._get_progress_bar_metrics("validating")
        ordered: dict[str, float] = {}
        for metrics in (training_metrics, validation_metrics):
            if metrics:
                name: str = next(iter(metrics))
                ordered.setdefault(name, metrics[name])
        for metrics in (training_metrics, validation_metrics):
            for name, value in metrics.items():
                ordered.setdefault(name, value)
        return dict(list(ordered.items())[:self._max_metrics])

    def _epoch_view(self, metrics: dict[str, float]) -> dict[str, float]:
        # Restricts a training view to the epoch-reduced names observed during
        # the epoch because an epoch record describes the completed epoch rather
        # than its last batch. When no epoch-reduced observation exists, on-step
        # names are shown unchanged.
        selected: dict[str, float] = {
            name: value
            for name, value in metrics.items()
            if name in self._epoch_metric_names
        }
        return selected or metrics

    def _fit_time_remaining(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> float | None:
        # Estimates remaining fit time from the mean completed epoch duration.
        # Unbounded epoch ceilings and step ceilings are not estimated because
        # the horizon would be invented rather than measured.
        if trainer.max_epochs == -1 or not self._epoch_durations:
            return None
        remaining_epochs: int = trainer.max_epochs - (module.current_epoch + 1)
        if remaining_epochs <= 0:
            return None
        mean_duration: float = sum(self._epoch_durations) / len(self._epoch_durations)
        return mean_duration * remaining_epochs

    def _failure_coordinates(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> list[str]:
        # Reports the one-based epoch, managed optimizer step, and the completed
        # batch count of whichever live view was active when the failure struck.
        parts: list[str] = [
            f"epoch {module.current_epoch + 1}",
            f"step {trainer.global_step}"
        ]
        active_stage: RunningStage | None = self._active_running_stage()
        if active_stage is not None:
            parts.append(f"batch {self._batch_counts[active_stage]}")
        return parts

    def _active_running_stage(self) -> RunningStage | None:
        # Resolves the innermost open live view; validation nested inside a fit
        # epoch takes precedence over the paused training view.
        if self._val_bar:
            return "validating"
        if self._train_bar:
            return "training"
        if self._test_bar:
            return "testing"
        if self._predict_bar:
            return "predicting"
        return None

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
                or self._epoch_view(trainer._get_progress_bar_metrics("training"))
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
        parts: list[str] = [f"{self._renderer.style('✓', 'green')} {stage} complete"]
        timer: Timer | None = self._timer(trainer)
        if stage == "fit" and timer is not None and timer.time_limit_reached:
            parts.append("max_time reached")
        parts.extend(self._renderer.metric_parts(metrics, self._max_metrics))
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
