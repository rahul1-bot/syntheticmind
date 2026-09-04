from __future__ import annotations

import math
import re
import time
from datetime import timedelta
from typing import Literal, override

import syntheticmind
from loguru import logger as log

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.exceptions import MisconfigurationError
from syntheticmind.types import Batch, StateDict, StateValue, StepOutput, TrainerStage

__all__: list[str] = ["Timer"]

type _TimerStage = Literal["fit", "train", "validate", "test", "predict"]


class Timer(Callback):
    def __init__(
        self,
        duration: str | timedelta | dict[str, int] | None = None,
        interval: Literal["step", "epoch"] = "step",
        verbose: bool = True
    ) -> None:
        super().__init__()
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
        del trainer, module
        if stage in ("validate", "test", "predict"):
            self._reset_stage(stage)

    def start_time(self, stage: _TimerStage = "train") -> float | None:
        return self._stage_started_at[stage]

    def end_time(self, stage: _TimerStage = "train") -> float | None:
        return self._stage_ended_at[stage]

    def time_elapsed(self, stage: _TimerStage = "train") -> float:
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
        if self._duration is None:
            return None
        return self._duration - self.time_elapsed(stage)

    @override
    def on_fit_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
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
        del trainer, module
        self._pause_active_training()
        self._stop_stage("fit")

    @override
    def on_train_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
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
        del module, outputs, batch, batch_idx
        if self._interval == "step":
            self._check_time_remaining(trainer)

    @override
    def on_validation_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
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
        del trainer, module
        self._start_stage("test")

    @override
    def on_test_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        del trainer, module
        self._stop_stage("test")

    @override
    def on_predict_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        del trainer, module
        self._start_stage("predict")

    @override
    def on_predict_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        del trainer, module
        self._stop_stage("predict")

    @override
    def on_exception(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        exception: BaseException
    ) -> None:
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
        return self._last_train_epoch_time

    @property
    def last_validation_time(self) -> float:
        return self._last_stage_durations["validate"]

    @property
    def last_test_time(self) -> float:
        return self._last_stage_durations["test"]

    @property
    def last_predict_time(self) -> float:
        return self._last_stage_durations["predict"]

    @property
    def time_limit_reached(self) -> bool:
        return self._time_limit_reached

    @override
    def state_dict(self) -> StateDict:
        return {
            "time_elapsed": {"train": self.time_elapsed("train")},
            "time_limit_reached": self._time_limit_reached
        }

    @override
    def load_state_dict(self, state_dict: StateDict) -> None:
        elapsed_value: StateValue | None = state_dict.get("time_elapsed")
        limit_value: StateValue | None = state_dict.get("time_limit_reached")
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
        elapsed: float = self._current_epoch_training_seconds
        if self._active_training_started_at is not None:
            elapsed += max(0.0, time.perf_counter() - self._active_training_started_at)
        return elapsed

    def _epoch_time_elapsed(self) -> float:
        if self._epoch_started_at is None:
            return self._last_epoch_time
        return max(0.0, time.perf_counter() - self._epoch_started_at)

    def _start_stage(self, stage: _TimerStage) -> None:
        current_time: float = time.perf_counter()
        self._stage_started_at[stage] = current_time
        self._stage_ended_at[stage] = None
        self._active_stage_started_at[stage] = current_time

    def _stop_stage(self, stage: _TimerStage) -> None:
        active_start: float | None = self._active_stage_started_at[stage]
        if active_start is None:
            return
        current_time: float = time.perf_counter()
        self._last_stage_durations[stage] = max(0.0, current_time - active_start)
        self._stage_ended_at[stage] = current_time
        self._active_stage_started_at[stage] = None

    def _resume_active_training(self) -> None:
        if self._active_training_started_at is None:
            self._active_training_started_at: float | None = time.perf_counter()

    def _pause_active_training(self) -> None:
        if self._active_training_started_at is None:
            return
        self._current_epoch_training_seconds += max(
            0.0,
            time.perf_counter() - self._active_training_started_at
        )
        self._active_training_started_at: float | None = None

    def _close_epoch(self) -> None:
        if self._epoch_started_at is None:
            return
        self._last_epoch_time: float = max(
            0.0,
            time.perf_counter() - self._epoch_started_at
        )
        self._epoch_started_at: float | None = None

    def _check_time_remaining(self, trainer: syntheticmind.Trainer) -> None:
        if self._duration is None:
            return
        rank_zero_stop: bool = (
            self.time_elapsed("train") >= self._duration
            if trainer.is_global_zero
            else False
        )
        should_stop: bool = trainer.strategy.broadcast(rank_zero_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if not should_stop or self._time_limit_reached:
            return
        self._time_limit_reached: bool = True
        if self._verbose and trainer.is_global_zero:
            elapsed: timedelta = timedelta(seconds=int(self.time_elapsed("train")))
            log.info(f"Time limit reached after {elapsed}; requesting training stop.")

    def _reset_all(self) -> None:
        self._stage_started_at = self._empty_times()
        self._stage_ended_at = self._empty_times()
        self._active_stage_started_at = self._empty_times()
        self._last_stage_durations = {
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
        self._stage_started_at[stage] = None
        self._stage_ended_at[stage] = None
        self._active_stage_started_at[stage] = None
        self._last_stage_durations[stage] = 0.0

    @staticmethod
    def _empty_times() -> dict[_TimerStage, float | None]:
        return {
            "fit": None,
            "train": None,
            "validate": None,
            "test": None,
            "predict": None
        }

    @staticmethod
    def _parse_duration(
        duration: str | timedelta | dict[str, int] | None
    ) -> float | None:
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
            parsed_duration = timedelta(
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
                parsed_duration = timedelta(**duration)
            except OverflowError as exception:
                raise MisconfigurationError(
                    f"Invalid Timer duration mapping {duration!r}."
                ) from exception
        elif isinstance(duration, timedelta):
            parsed_duration = duration
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
        return (
            f"Timer(duration={self._duration!r}, interval={self._interval!r}, "
            f"verbose={self._verbose})"
        )
