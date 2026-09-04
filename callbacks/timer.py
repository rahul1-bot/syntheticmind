# This module:
# 1. Defines Timer, the lifecycle clock that measures fit, active training,
#    validation, testing, prediction, and epoch durations
# 2. Implements a resumable training-time budget that requests a synchronized,
#    cooperative stop at step or epoch boundaries
#
# Design decisions:
# - The training budget counts active training wall time and pauses around
#   fit-time validation, so validation cadence does not consume a model's
#   configured optimization budget
# - Fit and standalone evaluation clocks remain separate from the active
#   training clock; client reporting can therefore distinguish end-to-end stage
#   latency from time spent executing training batches
# - Only elapsed training time crosses a checkpoint boundary. perf_counter
#   origins are process-local and meaningless after restart, so resumption adds
#   a validated numeric offset to a new monotonic interval
# - Global zero evaluates the budget and broadcasts the stop decision, keeping
#   distributed workers aligned without assuming synchronized host clocks
# - Failure handling closes every active interval idempotently, allowing
#   ProgressBarCallback and clients to query final durations during teardown
#
# Author: Rahul Sawhney

import math
import re
import time
from datetime import timedelta
from typing import Literal, override

import syntheticmind
from loguru import logger as log

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.primitives.exceptions import MisconfigurationError
from syntheticmind.primitives.types import Batch, StateDict, StateValue, StepOutput, TrainerStage

__all__: list[str] = ["Timer"]

type _TimerStage = Literal["fit", "train", "validate", "test", "predict"]


class Timer(Callback):
    # Stage clock and optional fit budget. Add this callback explicitly or let
    # the Trainer synthesize it from max_time; clients may query elapsed and
    # terminal durations through the public accessors after a run.
    def __init__(
        self,
        duration: str | timedelta | dict[str, int] | None = None,
        interval: Literal["step", "epoch"] = "step",
        verbose: bool = True
    ) -> None:
        # Args:
        #     duration: Positive budget as ``DD:HH:MM:SS``, timedelta, accepted
        #         timedelta keyword mapping, or None for measurement only.
        #         Default: ``None``.
        #     interval: Boundary at which the budget is checked, ``"step"`` or
        #         ``"epoch"``. Default: ``"step"``.
        #     verbose: Whether global zero logs the first budget exhaustion.
        #         Default: ``True``.
        if interval not in ("step", "epoch"):
            raise ValueError("`interval` must be 'step' or 'epoch'.")
        if type(verbose) is not bool:
            raise TypeError("`verbose` must be a bool.")
        self._duration: float | None = self._parse_duration(duration)
        self._interval: Literal["step", "epoch"] = interval
        self._verbose: bool = verbose
        self._stage_started_at: dict[_TimerStage, float | None] = self._empty_times()
        self._stage_ended_at: dict[_TimerStage, float | None] = self._empty_times()
        self._active_stage_started_at: dict[_TimerStage, float | None] = self._empty_times()
        self._last_stage_durations: dict[_TimerStage, float] = {
            stage: 0.0 for stage in self._stage_started_at
        }
        self._training_started_at: float | None = None
        self._training_ended_at: float | None = None
        self._training_time_offset: float = 0.0
        self._active_training_started_at: float | None = None
        self._current_epoch_training_seconds: float = 0.0
        self._last_train_epoch_time: float = 0.0
        self._epoch_started_at: float | None = None
        self._last_epoch_time: float = 0.0
        self._state_restored: bool = False
        self._time_limit_reached: bool = False

    @override
    def setup(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        stage: TrainerStage
    ) -> None:
        # Clears prior standalone-stage timing so repeated validate, test, or
        # predict calls report only their current transaction.
        del trainer, module
        if stage in ("validate", "test", "predict"):
            self._reset_stage(stage)

    def start_time(self, stage: _TimerStage = "train") -> float | None:
        # Returns this process's perf_counter origin for diagnostic correlation;
        # persisted clients should use elapsed durations instead.
        return self._stage_started_at[stage]

    def end_time(self, stage: _TimerStage = "train") -> float | None:
        # Returns the local perf_counter endpoint once the stage has closed.
        return self._stage_ended_at[stage]

    def time_elapsed(self, stage: _TimerStage = "train") -> float:
        # Computes live or terminal seconds. Training includes its restored
        # offset while other stages expose only the current process interval.
        if stage == "train":
            elapsed: float = self._training_time_offset
            if self._training_started_at is None:
                return elapsed
            end_time: float = (
                self._training_ended_at
                if self._training_ended_at is not None
                else time.perf_counter()
            )
            return elapsed + max(0.0, end_time - self._training_started_at)
        active_start: float | None = self._active_stage_started_at[stage]
        if active_start is not None:
            return max(0.0, time.perf_counter() - active_start)
        return self._last_stage_durations[stage]

    def time_remaining(self, stage: _TimerStage = "train") -> float | None:
        # Returns signed seconds against the configured budget, or None when the
        # callback is operating only as a clock.
        if self._duration is None:
            return None
        return self._duration - self.time_elapsed(stage)

    @override
    def on_fit_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        # Starts a fresh fit clock, retaining only a checkpoint-restored training
        # offset, and immediately rejects a budget already exhausted on resume.
        del module
        restored_offset: float = self._training_time_offset if self._state_restored else 0.0
        self._reset_all()
        self._training_time_offset: float = restored_offset
        self._state_restored: bool = False
        self._time_limit_reached: bool = False
        self._start_stage("fit")
        self._check_time_remaining(trainer)

    @override
    def on_fit_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        # Closes active training and the enclosing fit interval on normal exit.
        del trainer, module
        self._pause_active_training()
        self._stop_stage("fit")

    @override
    def on_train_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        # Opens the cumulative training clock after sanity validation.
        del trainer, module
        current_time: float = time.perf_counter()
        self._training_started_at: float | None = current_time
        self._training_ended_at: float | None = None
        self._stage_started_at["train"] = current_time
        self._stage_ended_at["train"] = None

    @override
    def on_train_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        # Closes the cumulative training clock without discarding its restored
        # offset or the final active-training duration.
        del trainer, module
        self._pause_active_training()
        if self._training_started_at is not None:
            self._training_ended_at: float | None = time.perf_counter()
            self._stage_ended_at["train"] = self._training_ended_at

    @override
    def on_train_epoch_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        # Opens both the full epoch clock and the active training portion.
        del trainer, module
        self._current_epoch_training_seconds: float = 0.0
        self._epoch_started_at: float | None = time.perf_counter()
        self._resume_active_training()

    @override
    def on_train_epoch_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        # Freezes the epoch's training-only and end-to-end durations, then checks
        # an epoch-cadence budget after all epoch work has completed.
        del module
        self._pause_active_training()
        self._last_train_epoch_time: float = self._current_epoch_training_seconds
        self._current_epoch_training_seconds: float = 0.0
        self._close_epoch()
        if self._interval == "epoch":
            self._check_time_remaining(trainer)

    @override
    def on_train_batch_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int
    ) -> None:
        # Checks a step-cadence budget after the batch and its optimizer work.
        del module, outputs, batch, batch_idx
        if self._interval == "step":
            self._check_time_remaining(trainer)

    @override
    def on_validation_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        # Opens validation timing and pauses active training for fit-time passes;
        # sanity validation is intentionally outside user timing.
        del module
        if trainer.sanity_checking:
            return
        if trainer.state.stage == "fit":
            self._pause_active_training()
        self._start_stage("validate")

    @override
    def on_validation_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        # Closes validation, resumes an interrupted training epoch, and checks
        # the budget at a durable fit boundary.
        del module
        if trainer.sanity_checking:
            return
        self._stop_stage("validate")
        if trainer.state.stage == "fit" and trainer._is_mid_epoch_validation:
            self._resume_active_training()
        if trainer.state.stage == "fit" and (
            self._interval == "step"
            or not trainer._is_mid_epoch_validation
        ):
            self._check_time_remaining(trainer)

    @override
    def on_test_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        # Opens the standalone test clock.
        del trainer, module
        self._start_stage("test")

    @override
    def on_test_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        # Closes the standalone test clock.
        del trainer, module
        self._stop_stage("test")

    @override
    def on_predict_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        # Opens the standalone prediction clock.
        del trainer, module
        self._start_stage("predict")

    @override
    def on_predict_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        # Closes the standalone prediction clock.
        del trainer, module
        self._stop_stage("predict")

    @override
    def on_exception(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        exception: BaseException
    ) -> None:
        # Freezes all active clocks before failure teardown without replacing
        # the original exception.
        del trainer, module, exception
        self._pause_active_training()
        if self._training_started_at is not None and self._training_ended_at is None:
            self._training_ended_at: float | None = time.perf_counter()
            self._stage_ended_at["train"] = self._training_ended_at
        self._close_epoch()
        self._stop_stage("validate")
        self._stop_stage("test")
        self._stop_stage("predict")
        self._stop_stage("fit")

    @property
    def last_train_epoch_time(self) -> float:
        # Returns active training seconds for the most recently closed epoch.
        return self._last_train_epoch_time

    @property
    def last_validation_time(self) -> float:
        # Returns seconds for the most recently closed validation pass.
        return self._last_stage_durations["validate"]

    @property
    def last_test_time(self) -> float:
        # Returns seconds for the most recently closed test pass.
        return self._last_stage_durations["test"]

    @property
    def last_predict_time(self) -> float:
        # Returns seconds for the most recently closed prediction pass.
        return self._last_stage_durations["predict"]

    @property
    def time_limit_reached(self) -> bool:
        # Reports whether this run has crossed its configured training budget.
        return self._time_limit_reached

    @override
    def state_dict(self) -> StateDict:
        # Persists semantic configuration and accumulated training seconds;
        # process-local timestamps are deliberately excluded.
        return {
            "duration": self._duration,
            "interval": self._interval,
            "time_elapsed": {"train": self.time_elapsed("train")},
            "time_limit_reached": self._time_limit_reached
        }

    @override
    def load_state_dict(self, state_dict: StateDict) -> None:
        # Restores elapsed training only when duration and check cadence exactly
        # match the active client configuration.
        duration_value: StateValue | None = state_dict.get("duration")
        interval_value: StateValue | None = state_dict.get("interval")
        elapsed_value: StateValue | None = state_dict.get("time_elapsed")
        limit_value: StateValue | None = state_dict.get("time_limit_reached")
        if duration_value != self._duration:
            raise MisconfigurationError(
                "Timer checkpoint duration does not match the configured duration."
            )
        if interval_value != self._interval:
            raise MisconfigurationError(
                "Timer checkpoint interval does not match the configured interval."
            )
        if not isinstance(elapsed_value, dict):
            raise TypeError("Timer checkpoint state requires a `time_elapsed` mapping.")
        training_value: StateValue | None = elapsed_value.get("train")
        if (
            type(training_value) not in (int, float)
            or not math.isfinite(float(training_value))
            or training_value < 0
        ):
            raise TypeError("Timer training elapsed state must be a finite non-negative number.")
        if type(limit_value) is not bool:
            raise TypeError("Timer time_limit_reached state must be a bool.")
        self._training_time_offset: float = float(training_value)
        self._time_limit_reached: bool = limit_value
        self._state_restored: bool = True

    def _train_epoch_time_elapsed(self) -> float:
        # Computes the live active-training portion of the current epoch.
        elapsed: float = self._current_epoch_training_seconds
        if self._active_training_started_at is not None:
            elapsed += max(0.0, time.perf_counter() - self._active_training_started_at)
        return elapsed

    def _epoch_time_elapsed(self) -> float:
        # Computes end-to-end epoch time, including scheduled validation.
        if self._epoch_started_at is None:
            return self._last_epoch_time
        return max(0.0, time.perf_counter() - self._epoch_started_at)

    def _start_stage(self, stage: _TimerStage) -> None:
        # Opens or replaces one non-training stage interval.
        current_time: float = time.perf_counter()
        self._stage_started_at[stage] = current_time
        self._stage_ended_at[stage] = None
        self._active_stage_started_at[stage] = current_time

    def _stop_stage(self, stage: _TimerStage) -> None:
        # Idempotently freezes one open stage interval.
        active_start: float | None = self._active_stage_started_at[stage]
        if active_start is None:
            return
        current_time: float = time.perf_counter()
        self._last_stage_durations[stage] = max(0.0, current_time - active_start)
        self._stage_ended_at[stage] = current_time
        self._active_stage_started_at[stage] = None

    def _resume_active_training(self) -> None:
        # Opens the current epoch's training-only interval if it is paused.
        if self._active_training_started_at is None:
            self._active_training_started_at: float | None = time.perf_counter()

    def _pause_active_training(self) -> None:
        # Accumulates and closes the current training-only interval.
        if self._active_training_started_at is None:
            return
        self._current_epoch_training_seconds += max(
            0.0,
            time.perf_counter() - self._active_training_started_at
        )
        self._active_training_started_at: float | None = None

    def _close_epoch(self) -> None:
        # Freezes the current end-to-end epoch duration once.
        if self._epoch_started_at is None:
            return
        self._last_epoch_time: float = max(
            0.0,
            time.perf_counter() - self._epoch_started_at
        )
        self._epoch_started_at: float | None = None

    def _check_time_remaining(self, trainer: syntheticmind.Trainer) -> None:
        # Evaluates on global zero, broadcasts the result, and raises only the
        # cooperative stop flag so loops finish their current durable boundary.
        if self._duration is None:
            return
        rank_zero_stop: bool = (
            self.time_elapsed("train") >= self._duration
            if trainer.is_global_zero
            else False
        )
        should_stop: bool = trainer.strategy.broadcast(rank_zero_stop)
        trainer.should_stop: bool = trainer.should_stop or should_stop
        if not should_stop or self._time_limit_reached:
            return
        self._time_limit_reached: bool = True
        if self._verbose and trainer.is_global_zero:
            elapsed: timedelta = timedelta(seconds=int(self.time_elapsed("train")))
            log.info(f"Time limit reached after {elapsed}; requesting training stop.")

    def _reset_all(self) -> None:
        # Clears process-local clocks for a new fit while the caller separately
        # preserves any checkpoint-restored training offset.
        self._stage_started_at: dict[_TimerStage, float | None] = self._empty_times()
        self._stage_ended_at: dict[_TimerStage, float | None] = self._empty_times()
        self._active_stage_started_at: dict[
            _TimerStage,
            float | None
        ] = self._empty_times()
        self._last_stage_durations: dict[_TimerStage, float] = {
            stage: 0.0 for stage in self._stage_started_at
        }
        self._training_started_at: float | None = None
        self._training_ended_at: float | None = None
        self._active_training_started_at: float | None = None
        self._current_epoch_training_seconds: float = 0.0
        self._last_train_epoch_time: float = 0.0
        self._epoch_started_at: float | None = None
        self._last_epoch_time: float = 0.0

    def _reset_stage(self, stage: _TimerStage) -> None:
        # Clears one repeatable standalone stage without perturbing fit state.
        self._stage_started_at[stage] = None
        self._stage_ended_at[stage] = None
        self._active_stage_started_at[stage] = None
        self._last_stage_durations[stage] = 0.0

    def _empty_times(self) -> dict[_TimerStage, float | None]:
        # Creates independent mutable stage-clock mappings for each timing view.
        return {
            "fit": None,
            "train": None,
            "validate": None,
            "test": None,
            "predict": None
        }

    def _parse_duration(
        self,
        duration: str | timedelta | dict[str, int] | None
    ) -> float | None:
        # Normalizes supported client duration forms to positive finite seconds
        # and rejects ambiguous strings or timedelta mappings immediately.
        if duration is None:
            return None
        parsed_duration: timedelta
        if isinstance(duration, str):
            duration_match: re.Match[str] | None = re.fullmatch(
                r"(\d+):(\d\d):(\d\d):(\d\d)",
                duration.strip()
            )
            if duration_match is None:
                raise MisconfigurationError(
                    f"Invalid Timer duration {duration!r}; expected DD:HH:MM:SS."
                )
            hours: int = int(duration_match.group(2))
            minutes: int = int(duration_match.group(3))
            seconds: int = int(duration_match.group(4))
            if hours >= 24 or minutes >= 60 or seconds >= 60:
                raise MisconfigurationError(
                    f"Invalid Timer duration {duration!r}; hours must be below 24 "
                    "and minutes or seconds below 60."
                )
            parsed_duration: timedelta = timedelta(
                days=int(duration_match.group(1)),
                hours=hours,
                minutes=minutes,
                seconds=seconds
            )
        elif isinstance(duration, dict):
            allowed_units: set[str] = {
                "weeks",
                "days",
                "hours",
                "minutes",
                "seconds",
                "milliseconds",
                "microseconds"
            }
            if not duration or set(duration) - allowed_units:
                raise MisconfigurationError(
                    f"Invalid Timer duration mapping keys: {sorted(duration)}."
                )
            if any(type(value) is not int for value in duration.values()):
                raise TypeError("Timer duration mapping values must be integers.")
            if any(value < 0 for value in duration.values()):
                raise ValueError("Timer duration mapping values must be non-negative.")
            try:
                parsed_duration: timedelta = timedelta(**duration)
            except OverflowError as exception:
                raise MisconfigurationError(
                    f"Invalid Timer duration mapping {duration!r}."
                ) from exception
        elif isinstance(duration, timedelta):
            parsed_duration: timedelta = duration
        else:
            raise TypeError(
                "Timer duration must be DD:HH:MM:SS, timedelta, an integer mapping, or None."
            )
        duration_seconds: float = parsed_duration.total_seconds()
        if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
            raise ValueError("Timer duration must resolve to a positive finite interval.")
        return duration_seconds

    @override
    def __repr__(self) -> str:
        # Presents the normalized budget and cadence used by the callback.
        return (
            f"Timer(duration={self._duration!r}, interval={self._interval!r}, "
            f"verbose={self._verbose})"
        )
