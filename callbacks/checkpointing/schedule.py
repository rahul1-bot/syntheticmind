import math
import time
from datetime import timedelta

import syntheticmind

from syntheticmind.primitives.types import StateDict, StateValue

__all__: list[str] = []

type _ScheduleSnapshot = tuple[int, float | None, float, bool]


class _CheckpointSchedule:
    def __init__(
        self,
        every_n_train_steps: int,
        train_time_interval: timedelta | None,
        every_n_epochs: int,
        save_on_train_epoch_end: bool | None
    ) -> None:
        self.every_n_train_steps: int = every_n_train_steps
        self.train_time_interval: timedelta | None = train_time_interval
        self.every_n_epochs: int = every_n_epochs
        self.save_on_train_epoch_end: bool | None = save_on_train_epoch_end
        self.last_global_step_saved: int = -1
        self.last_time_checked: float | None = None
        self.restored_time_since_last_check: float = 0.0
        self.deferred_until_validation: bool = False

    def start(self) -> None:
        self.last_time_checked: float | None = (
            time.monotonic() - self.restored_time_since_last_check
        )
        self.restored_time_since_last_check: float = 0.0

    def should_skip(self, trainer: syntheticmind.Trainer) -> bool:
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
        if self.should_save_on_train_epoch_end(trainer) and not self.deferred_until_validation:
            return False
        if self.deferred_until_validation:
            return True
        return (
            self.every_n_epochs > 0
            and (trainer.current_epoch + 1) % self.every_n_epochs == 0
        )

    def epoch_trigger(self, trainer: syntheticmind.Trainer) -> bool:
        return (
            self.should_save_on_train_epoch_end(trainer)
            and self.every_n_epochs > 0
            and (trainer.current_epoch + 1) % self.every_n_epochs == 0
        )

    def should_save_on_train_epoch_end(
        self,
        trainer: syntheticmind.Trainer
    ) -> bool:
        if self.save_on_train_epoch_end is not None:
            return self.save_on_train_epoch_end
        if trainer.check_val_every_n_epoch != 1:
            return False
        if not trainer.enable_validation or not any(trainer.num_val_batches):
            return True
        return trainer.val_check_interval == 1.0

    def mark_saved(self, global_step: int, trigger_time: float) -> None:
        self.last_global_step_saved: int = global_step
        self.last_time_checked: float | None = trigger_time

    def mark_time_checked(self, trigger_time: float) -> None:
        self.last_time_checked: float | None = trigger_time

    def defer(self) -> None:
        self.deferred_until_validation: bool = True

    def clear_deferred(self) -> None:
        self.deferred_until_validation: bool = False

    def snapshot(self) -> _ScheduleSnapshot:
        return (
            self.last_global_step_saved,
            self.last_time_checked,
            self.restored_time_since_last_check,
            self.deferred_until_validation
        )

    def restore_snapshot(self, snapshot: _ScheduleSnapshot) -> None:
        self.last_global_step_saved: int = snapshot[0]
        self.last_time_checked: float | None = snapshot[1]
        self.restored_time_since_last_check: float = snapshot[2]
        self.deferred_until_validation: bool = snapshot[3]

    def state_dict(self) -> StateDict:
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
