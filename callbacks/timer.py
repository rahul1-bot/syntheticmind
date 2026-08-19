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

import time
from typing import TYPE_CHECKING, Literal, override

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.utilities.types import TrainerStage

if TYPE_CHECKING:
    from syntheticmind.core.trainer import Trainer

__all__: list[str] = ["TimerCallback"]

type _TimerStage = Literal["fit", "train", "validate", "test", "predict"]


class TimerCallback(Callback):
    # Wall-clock timing over the fit lifecycle: per-epoch training durations,
    # per-pass validation durations, and the final summary splitting total
    # time into its training and validation components.
    def __init__(self) -> None:
        # Zeroes the fit-start and epoch-start timestamps and the accumulated
        # training and validation totals.
        super().__init__()
        self._start_times: dict[_TimerStage, float | None] = {
            "fit": None,
            "train": None,
            "validate": None,
            "test": None,
            "predict": None
        }
        self._elapsed_times: dict[_TimerStage, float] = {
            "fit": 0.0,
            "train": 0.0,
            "validate": 0.0,
            "test": 0.0,
            "predict": 0.0
        }
        self._last_stage_durations: dict[_TimerStage, float] = {
            "fit": 0.0,
            "train": 0.0,
            "validate": 0.0,
            "test": 0.0,
            "predict": 0.0
        }
        self._last_train_epoch_time: float = 0.0
        self._current_train_epoch_time: float = 0.0

    @override
    def setup(self, trainer: Trainer, module: Module, stage: TrainerStage) -> None:
        if stage == "validate":
            self._reset_stage("validate")
        elif stage == "test":
            self._reset_stage("test")
        elif stage == "predict":
            self._reset_stage("predict")

    def time_elapsed(self, stage: _TimerStage = "fit") -> float:
        elapsed: float = self._elapsed_times[stage]
        start_time: float | None = self._start_times[stage]
        if start_time is not None:
            elapsed += time.perf_counter() - start_time
        return elapsed

    @override
    def on_fit_start(self, trainer: Trainer, module: Module) -> None:
        # Marks the beginning of the fit for the end-of-run total.
        for stage in self._elapsed_times:
            self._elapsed_times[stage] = 0.0
            self._start_times[stage] = None
            self._last_stage_durations[stage] = 0.0
        self._last_train_epoch_time: float = 0.0
        self._current_train_epoch_time: float = 0.0
        self._start_stage("fit")

    @override
    def on_fit_end(self, trainer: Trainer, module: Module) -> None:
        # Reports the total fit duration together with the accumulated
        # training and validation components, from rank zero only.
        self._pause_training()
        self._stop_stage("fit")

    @override
    def on_train_epoch_start(self, trainer: Trainer, module: Module) -> None:
        # Marks the beginning of the training epoch.
        self._current_train_epoch_time: float = 0.0
        self._resume_training()

    @override
    def on_train_epoch_end(self, trainer: Trainer, module: Module) -> None:
        # Accumulates the epoch duration into the training total and reports
        # it from rank zero.
        self._pause_training()
        self._last_train_epoch_time: float = self._current_train_epoch_time
        self._current_train_epoch_time: float = 0.0

    @override
    def on_validation_start(self, trainer: Trainer, module: Module) -> None:
        # Marks the beginning of the validation pass.
        if trainer._sanity_checking:
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
        if trainer._sanity_checking:
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

    def _start_stage(self, stage: _TimerStage) -> None:
        if self._start_times[stage] is None:
            self._start_times[stage] = time.perf_counter()

    def _stop_stage(self, stage: _TimerStage) -> None:
        start_time: float | None = self._start_times[stage]
        if start_time is None:
            return
        duration: float = time.perf_counter() - start_time
        self._elapsed_times[stage] += duration
        self._last_stage_durations[stage] = duration
        self._start_times[stage] = None

    def _last_stage_duration(self, stage: _TimerStage) -> float:
        return self._last_stage_durations[stage]

    def _reset_stage(self, stage: _TimerStage) -> None:
        self._start_times[stage] = None
        self._elapsed_times[stage] = 0.0
        self._last_stage_durations[stage] = 0.0

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
