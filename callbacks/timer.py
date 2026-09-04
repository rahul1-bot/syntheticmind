# This module:
# 1. Measures wall-clock durations for the whole fit, each training epoch, and
#    each validation pass, and reports them through the log at rank zero
#
# Design decisions:
# - Durations are measured with the monotonic performance counter, so system
#   clock adjustments cannot corrupt them
# - One epoch-start timestamp slot serves both training epochs and validation
#   passes, which is sound because the harness never interleaves the start of
#   one with the end of the other on a single process
# - Timestamps are recorded on every rank so accumulated totals stay
#   meaningful everywhere, while log emission is restricted to rank zero to
#   keep distributed output uncluttered
#
# Author: Rahul Sawhney

from __future__ import annotations

import re
import time
from datetime import timedelta
from typing import TYPE_CHECKING, Literal, override

from loguru import logger as log

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.utilities.distributed import is_rank_zero
from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.types import Batch, StateDict, StepOutput, TrainerStage

if TYPE_CHECKING:
    from syntheticmind.trainer.trainer import Trainer

__all__: list[str] = ["Timer"]

type _TimerStage = Literal[
    "fit",
    "train",
    "sanity_check",
    "validate",
    "test",
    "predict"
]


class Timer(Callback):
    # Wall-clock timing over the fit lifecycle: per-epoch training durations,
    # per-pass validation durations, and the final summary splitting total
    # time into its training and validation components.
    def __init__(
        self,
        duration: str | timedelta | dict[str, int] | None = None,
        interval: Literal["step", "epoch"] = "step",
        verbose: bool = True
    ) -> None:
        # Zeroes the fit-start and epoch-start timestamps and the accumulated
        # training and validation totals.
        super().__init__()
        self._duration: float | None = self._parse_duration(duration)
        if interval not in {"step", "epoch"}:
            raise MisconfigurationError(
                f"Unsupported Timer interval {interval!r}. Expected 'step' or 'epoch'."
            )
        self._interval: Literal["step", "epoch"] = interval
        self._verbose: bool = verbose
        self._start_times: dict[_TimerStage, float | None] = {
            "fit": None,
            "train": None,
            "sanity_check": None,
            "validate": None,
            "test": None,
            "predict": None
        }
        self._elapsed_times: dict[_TimerStage, float] = {
            "fit": 0.0,
            "train": 0.0,
            "sanity_check": 0.0,
            "validate": 0.0,
            "test": 0.0,
            "predict": 0.0
        }
        self._last_stage_durations: dict[_TimerStage, float] = {
            "fit": 0.0,
            "train": 0.0,
            "sanity_check": 0.0,
            "validate": 0.0,
            "test": 0.0,
            "predict": 0.0
        }
        self._last_train_epoch_time: float = 0.0
        self._current_train_epoch_time: float = 0.0
        self._epoch_start_time: float | None = None
        self._last_epoch_time: float = 0.0
        self._stage_start_times: dict[_TimerStage, float | None] = {
            "fit": None,
            "train": None,
            "sanity_check": None,
            "validate": None,
            "test": None,
            "predict": None
        }
        self._end_times: dict[_TimerStage, float | None] = {
            "fit": None,
            "train": None,
            "sanity_check": None,
            "validate": None,
            "test": None,
            "predict": None
        }
        self._training_wall_start: float | None = None
        self._training_wall_end: float | None = None
        self._training_time_offset: float = 0.0
        self._restored: bool = False
        self._time_limit_reached: bool = False

    @override
    def setup(self, trainer: Trainer, module: Module, stage: TrainerStage) -> None:
        if stage == "validate":
            self._reset_stage("validate")
        elif stage == "test":
            self._reset_stage("test")
        elif stage == "predict":
            self._reset_stage("predict")

    def start_time(self, stage: _TimerStage = "train") -> float | None:
        return self._stage_start_times[stage]

    def end_time(self, stage: _TimerStage = "train") -> float | None:
        return self._end_times[stage]

    def time_elapsed(self, stage: _TimerStage = "train") -> float:
        if stage == "train":
            elapsed: float = self._training_time_offset
            if self._training_wall_start is not None:
                training_end_time: float = (
                    self._training_wall_end
                    if self._training_wall_end is not None
                    else time.perf_counter()
                )
                elapsed: float = elapsed + training_end_time - self._training_wall_start
            return elapsed
        stage_start_time: float | None = self.start_time(stage)
        if stage_start_time is None:
            return 0.0
        stage_end_time: float | None = self.end_time(stage)
        if stage_end_time is None:
            return time.perf_counter() - stage_start_time
        return stage_end_time - stage_start_time

    def time_remaining(self, stage: _TimerStage = "train") -> float | None:
        if self._duration is None:
            return None
        return self._duration - self.time_elapsed(stage)

    @override
    def on_fit_start(self, trainer: Trainer, module: Module) -> None:
        # Marks the beginning of the fit for the end-of-run total.
        for stage in self._elapsed_times:
            self._elapsed_times[stage] = 0.0
            self._start_times[stage] = None
            self._last_stage_durations[stage] = 0.0
            self._stage_start_times[stage] = None
            self._end_times[stage] = None
        if not self._restored:
            self._training_time_offset = 0.0
        self._restored = False
        self._training_wall_start: float | None = None
        self._training_wall_end: float | None = None
        self._time_limit_reached: bool = False
        self._last_train_epoch_time: float = 0.0
        self._current_train_epoch_time: float = 0.0
        self._epoch_start_time: float | None = None
        self._last_epoch_time: float = 0.0
        self._start_stage("fit")
        self._check_time_remaining(trainer)

    @override
    def on_fit_end(self, trainer: Trainer, module: Module) -> None:
        # Reports the total fit duration together with the accumulated
        # training and validation components, from rank zero only.
        self._pause_training()
        self._stop_stage("fit")

    @override
    def on_train_start(self, trainer: Trainer, module: Module) -> None:
        current_time: float = time.perf_counter()
        self._training_wall_start: float | None = current_time
        self._training_wall_end: float | None = None
        self._stage_start_times["train"] = current_time
        self._end_times["train"] = None

    @override
    def on_train_end(self, trainer: Trainer, module: Module) -> None:
        self._pause_training()
        if self._training_wall_start is not None:
            self._training_wall_end: float | None = time.perf_counter()
            self._end_times["train"] = self._training_wall_end

    @override
    def on_train_epoch_start(self, trainer: Trainer, module: Module) -> None:
        # Marks the beginning of the training epoch.
        self._current_train_epoch_time: float = 0.0
        self._epoch_start_time: float | None = time.perf_counter()
        self._resume_training()

    @override
    def on_train_epoch_end(self, trainer: Trainer, module: Module) -> None:
        # Accumulates the epoch duration into the training total and reports
        # it from rank zero.
        self._pause_training()
        self._last_train_epoch_time: float = self._current_train_epoch_time
        self._current_train_epoch_time: float = 0.0
        self._close_epoch()
        if self._interval == "epoch":
            self._check_time_remaining(trainer)

    @override
    def on_train_batch_end(
        self,
        trainer: Trainer,
        module: Module,
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int
    ) -> None:
        if self._interval == "step":
            self._check_time_remaining(trainer)

    @override
    def on_validation_start(self, trainer: Trainer, module: Module) -> None:
        # Marks the beginning of the validation pass.
        if trainer.sanity_checking:
            return
        if trainer.state.stage == "validate":
            self._elapsed_times["validate"] = 0.0
        if trainer.state.stage == "fit":
            self._pause_training()
        self._start_stage("validate")

    @override
    def on_validation_end(self, trainer: Trainer, module: Module) -> None:
        # Accumulates the pass duration into the validation total and reports
        # it from rank zero.
        if trainer.sanity_checking:
            return
        self._stop_stage("validate")
        if trainer.state.stage == "fit" and trainer._is_mid_epoch_validation:
            self._resume_training()

    @override
    def on_test_start(self, trainer: Trainer, module: Module) -> None:
        self._elapsed_times["test"] = 0.0
        self._start_stage("test")

    @override
    def on_test_end(self, trainer: Trainer, module: Module) -> None:
        self._stop_stage("test")

    @override
    def on_predict_start(self, trainer: Trainer, module: Module) -> None:
        self._elapsed_times["predict"] = 0.0
        self._start_stage("predict")

    @override
    def on_predict_end(self, trainer: Trainer, module: Module) -> None:
        self._stop_stage("predict")

    @override
    def on_exception(self, trainer: Trainer, module: Module, exception: BaseException) -> None:
        self._pause_training()
        if self._training_wall_start is not None and self._training_wall_end is None:
            self._training_wall_end: float | None = time.perf_counter()
            self._end_times["train"] = self._training_wall_end
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
        return self._last_stage_duration("validate")

    @property
    def last_test_time(self) -> float:
        return self._last_stage_duration("test")

    @property
    def last_predict_time(self) -> float:
        return self._last_stage_duration("predict")

    @property
    def time_limit_reached(self) -> bool:
        return self._time_limit_reached

    @override
    def state_dict(self) -> StateDict:
        return {
            "time_elapsed": {
                stage: self.time_elapsed(stage)
                for stage in self._elapsed_times
            }
        }

    @override
    def load_state_dict(self, state_dict: StateDict) -> None:
        time_elapsed: object = state_dict.get("time_elapsed", {})
        if isinstance(time_elapsed, dict):
            training_elapsed: object = time_elapsed.get("train", 0.0)
            if isinstance(training_elapsed, (int, float)):
                self._training_time_offset: float = float(training_elapsed)
        self._restored: bool = True

    def _start_stage(self, stage: _TimerStage) -> None:
        if self._start_times[stage] is None:
            current_time: float = time.perf_counter()
            self._start_times[stage] = current_time
            self._stage_start_times[stage] = current_time
            self._end_times[stage] = None

    def _stop_stage(self, stage: _TimerStage) -> None:
        start_time: float | None = self._start_times[stage]
        if start_time is None:
            return
        duration: float = time.perf_counter() - start_time
        self._elapsed_times[stage] += duration
        self._last_stage_durations[stage] = duration
        self._end_times[stage] = start_time + duration
        self._start_times[stage] = None

    def _last_stage_duration(self, stage: _TimerStage) -> float:
        return self._last_stage_durations[stage]

    def _train_epoch_time_elapsed(self) -> float:
        elapsed: float = self._current_train_epoch_time
        start_time: float | None = self._start_times["train"]
        if start_time is not None:
            elapsed: float = elapsed + time.perf_counter() - start_time
        return elapsed

    def _epoch_time_elapsed(self) -> float:
        if self._epoch_start_time is not None:
            return time.perf_counter() - self._epoch_start_time
        return self._last_epoch_time

    def _reset_stage(self, stage: _TimerStage) -> None:
        self._start_times[stage] = None
        self._elapsed_times[stage] = 0.0
        self._last_stage_durations[stage] = 0.0
        self._stage_start_times[stage] = None
        self._end_times[stage] = None

    def _resume_training(self) -> None:
        self._start_stage("train")

    def _pause_training(self) -> None:
        start_time: float | None = self._start_times["train"]
        if start_time is None:
            return
        duration: float = time.perf_counter() - start_time
        self._elapsed_times["train"] += duration
        self._current_train_epoch_time += duration
        self._start_times["train"] = None

    def _close_epoch(self) -> None:
        if self._epoch_start_time is None:
            return
        self._last_epoch_time: float = time.perf_counter() - self._epoch_start_time
        self._epoch_start_time: float | None = None

    def _check_time_remaining(self, trainer: Trainer) -> None:
        if self._duration is None:
            return
        should_stop: bool = self.time_elapsed("train") >= self._duration
        should_stop: bool = trainer.strategy.reduce_boolean_decision(should_stop)
        trainer_should_stop: bool = trainer.should_stop or should_stop
        trainer.should_stop = trainer_should_stop
        if should_stop and not self._time_limit_reached:
            self._time_limit_reached: bool = True
            if self._verbose and is_rank_zero():
                elapsed: timedelta = timedelta(seconds=int(self.time_elapsed("train")))
                log.info(f"Time limit reached after {elapsed}; requesting training stop.")

    def _parse_duration(
        self,
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
                    f"Invalid Timer duration {duration!r}. Expected 'DD:HH:MM:SS'."
                )
            parsed_duration: timedelta = timedelta(
                days=int(duration_match.group(1)),
                hours=int(duration_match.group(2)),
                minutes=int(duration_match.group(3)),
                seconds=int(duration_match.group(4))
            )
        elif isinstance(duration, dict):
            try:
                parsed_duration: timedelta = timedelta(**duration)
            except (TypeError, OverflowError) as exception:
                raise MisconfigurationError(
                    f"Invalid Timer duration mapping {duration!r}."
                ) from exception
        elif isinstance(duration, timedelta):
            parsed_duration: timedelta = duration
        else:
            raise TypeError(
                "Timer duration must be a string, datetime.timedelta, dictionary, or None."
            )
        return parsed_duration.total_seconds()
