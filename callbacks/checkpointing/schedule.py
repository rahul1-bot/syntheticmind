# This module:
# 1. Defines the checkpoint trigger schedule across optimizer steps, elapsed
#    training time, completed epochs, validation availability, and resumed runs
# 2. Snapshots and persists scheduling coordinates independently of ranking and
#    storage state
#
# Design decisions:
# - should_skip centralizes contexts in which persistence has no valid training
#   meaning: fast development, non-fit stages, sanity checks, and a global step
#   already saved
# - Wall-time eligibility is evaluated on global zero and broadcast because host
#   monotonic clocks are not a distributed synchronization primitive
# - A monitored step or time trigger may defer until validation supplies its
#   metric; the deferred flag makes that dependency explicit across hook events
# - Default epoch-end placement depends on validation cadence, ensuring a save
#   sees the latest validation metrics instead of persisting too early
# - Checkpoints store elapsed time since the last check, never an absolute
#   monotonic timestamp that cannot survive process restart
#
# Author: Rahul Sawhney

import math
import time
from datetime import timedelta

import syntheticmind

from syntheticmind.primitives.types import StateDict, StateValue

__all__: list[str] = []

type _ScheduleSnapshot = tuple[int, float | None, float, bool]


class _CheckpointSchedule:
    # Pure trigger state machine composed by ModelCheckpoint and its transaction.
    def __init__(
        self,
        every_n_train_steps: int,
        train_time_interval: timedelta | None,
        every_n_epochs: int,
        save_on_train_epoch_end: bool | None
    ) -> None:
        # Binds one periodic trigger family and initializes fresh-run coordinates.
        self.every_n_train_steps: int = every_n_train_steps
        self.train_time_interval: timedelta | None = train_time_interval
        self.every_n_epochs: int = every_n_epochs
        self.save_on_train_epoch_end: bool | None = save_on_train_epoch_end
        self.last_global_step_saved: int = -1
        self.last_time_checked: float | None = None
        self.restored_time_since_last_check: float = 0.0
        self.deferred_until_validation: bool = False

    def start(self) -> None:
        # Reconstructs a process-local monotonic origin from a persisted elapsed
        # interval when training begins or resumes.
        self.last_time_checked: float | None = (
            time.monotonic() - self.restored_time_since_last_check
        )
        self.restored_time_since_last_check: float = 0.0

    def should_skip(self, trainer: syntheticmind.Trainer) -> bool:
        # Rejects invalid lifecycle contexts and duplicate global-step saves.
        return (
            bool(trainer.fast_dev_run)
            or trainer.state.stage != "fit"
            or trainer.sanity_checking
            or self.last_global_step_saved == trainer.global_step
        )

    def batch_trigger(
        self,
        trainer: syntheticmind.Trainer
    ) -> tuple[bool, float]:
        # Evaluates step and wall-time triggers at a completed training batch and
        # returns the shared decision plus the timestamp to commit on success.
        step_due: bool = (
            self.every_n_train_steps > 0
            and trainer.global_step > 0
            and trainer.global_step % self.every_n_train_steps == 0
        )
        trigger_time: float = time.monotonic()
        time_due: bool = False
        if (
            trainer.is_global_zero
            and self.train_time_interval is not None
            and self.last_time_checked is not None
        ):
            time_due: bool = (
                trigger_time - self.last_time_checked
                >= self.train_time_interval.total_seconds()
            )
        time_due: bool = trainer.strategy.broadcast(time_due)
        return step_due or time_due, trigger_time

    def validation_trigger(self, trainer: syntheticmind.Trainer) -> bool:
        # Resolves a deferred monitored save or a validation-positioned epoch
        # trigger after validation metrics are available.
        if self.should_save_on_train_epoch_end(trainer) and not self.deferred_until_validation:
            return False
        if self.deferred_until_validation:
            return True
        return (
            self.every_n_epochs > 0
            and (trainer.current_epoch + 1) % self.every_n_epochs == 0
        )

    def epoch_trigger(self, trainer: syntheticmind.Trainer) -> bool:
        # Evaluates checkpoint cadence for policies placed at train epoch end.
        return (
            self.should_save_on_train_epoch_end(trainer)
            and self.every_n_epochs > 0
            and (trainer.current_epoch + 1) % self.every_n_epochs == 0
        )

    def should_save_on_train_epoch_end(
        self,
        trainer: syntheticmind.Trainer
    ) -> bool:
        # Chooses train-end versus post-validation placement from explicit client
        # policy or the trainer's actual validation schedule.
        if self.save_on_train_epoch_end is not None:
            return self.save_on_train_epoch_end
        if trainer.check_val_every_n_epoch != 1:
            return False
        if not trainer.enable_validation or not any(trainer.num_val_batches):
            return True
        return trainer.val_check_interval == 1.0

    def mark_saved(self, global_step: int, trigger_time: float) -> None:
        # Commits duplicate-suppression and wall-time coordinates after success.
        self.last_global_step_saved: int = global_step
        self.last_time_checked: float | None = trigger_time

    def mark_time_checked(self, trigger_time: float) -> None:
        # Advances only the time clock when a due check produces no artifact.
        self.last_time_checked: float | None = trigger_time

    def defer(self) -> None:
        # Records that the next validation result must resolve a pending save.
        self.deferred_until_validation: bool = True

    def clear_deferred(self) -> None:
        # Clears monitored deferral once a validation-aware transaction begins.
        self.deferred_until_validation: bool = False

    def snapshot(self) -> _ScheduleSnapshot:
        # Captures an in-process rollback point before logical save state mutates.
        return (
            self.last_global_step_saved,
            self.last_time_checked,
            self.restored_time_since_last_check,
            self.deferred_until_validation
        )

    def restore_snapshot(self, snapshot: _ScheduleSnapshot) -> None:
        # Reinstates scheduling state when persistence fails.
        self.last_global_step_saved: int = snapshot[0]
        self.last_time_checked: float | None = snapshot[1]
        self.restored_time_since_last_check: float = snapshot[2]
        self.deferred_until_validation: bool = snapshot[3]

    def state_dict(self) -> StateDict:
        # Serializes portable cadence coordinates and derived elapsed time.
        elapsed_since_last_check: float = (
            0.0
            if self.last_time_checked is None
            else max(0.0, time.monotonic() - self.last_time_checked)
        )
        return {
            "last_global_step_saved": self.last_global_step_saved,
            "defer_save_until_validation": self.deferred_until_validation,
            "time_since_last_check": elapsed_since_last_check
        }

    def load_state_dict(self, state_dict: StateDict) -> None:
        # Restores validated schedule state while deferring monotonic-origin
        # reconstruction until start is called in the new process.
        step_value: StateValue | None = state_dict.get("last_global_step_saved")
        deferred_value: StateValue | None = state_dict.get(
            "defer_save_until_validation"
        )
        elapsed_value: StateValue | None = state_dict.get("time_since_last_check")
        if type(step_value) is not int or step_value < -1:
            raise TypeError("Checkpoint schedule step state must be an integer >= -1.")
        if type(deferred_value) is not bool:
            raise TypeError("Checkpoint deferred-save state must be a bool.")
        if (
            type(elapsed_value) not in (int, float)
            or not math.isfinite(float(elapsed_value))
            or elapsed_value < 0
        ):
            raise TypeError("Checkpoint schedule time state must be finite and non-negative.")
        self.last_global_step_saved: int = step_value
        self.deferred_until_validation: bool = deferred_value
        self.restored_time_since_last_check: float = float(elapsed_value)
        self.last_time_checked: float | None = None
