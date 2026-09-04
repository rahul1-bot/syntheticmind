# This module:
# 1. Defines ValidationSchedule, the stateful policy resolving epoch eligibility
#    and batch-, cross-epoch-, fraction-, final-batch-, stop-, or time-based
#    validation boundaries
#
# Design decisions:
# - check_val_every_n_epoch=None gives integer intervals cross-epoch batch
#   semantics; a finite epoch cadence resets integer counting within each epoch
# - Fractional intervals require a sized loader because a proportion cannot be
#   resolved from an unknown horizon
# - Unsized loaders accept integer, time-based, or final-exhaustion validation;
#   the last form waits for lookahead-confirmed natural epoch completion
# - Cooperative stopping triggers validation before exit when minimum epoch and
#   step conditions allow the stop, ensuring monitors observe the final state
# - Time cadence uses monotonic time and advances only after validation ends, so
#   a long validation pass does not immediately trigger another pass
# - String durations use explicit DD:HH:MM:SS clock fields and mappings delegate
#   only to timedelta-supported integer units
#
# Author: Rahul Sawhney

import time
from datetime import timedelta

from syntheticmind.primitives.exceptions import MisconfigurationError

__all__: list[str] = ["ValidationSchedule"]

type _ValidationInterval = int | float | str | timedelta | dict[str, int]


class ValidationSchedule:
    # Configured validation policy reused across fit epochs. Trainer supplies the
    # resolved train-loader horizon and whether a validation source exists.
    def __init__(
        self,
        val_check_interval: _ValidationInterval | None,
        check_val_every_n_epoch: int | None
    ) -> None:
        # Validates public cadence forms, parses optional duration, and starts
        # disabled until Trainer attaches actual data.
        if check_val_every_n_epoch is not None and (
            isinstance(check_val_every_n_epoch, bool)
            or not isinstance(check_val_every_n_epoch, int)
            or check_val_every_n_epoch < 1
        ):
            raise MisconfigurationError(
                "`check_val_every_n_epoch` must be a positive integer or None, "
                f"got {check_val_every_n_epoch!r}."
            )
        if check_val_every_n_epoch is None and isinstance(val_check_interval, float):
            raise MisconfigurationError(
                "`val_check_interval` must be an integer or a time-based duration when "
                "`check_val_every_n_epoch=None`."
            )
        self.check_val_every_n_epoch: int | None = check_val_every_n_epoch
        self.val_check_interval: _ValidationInterval = (
            1.0 if val_check_interval is None else val_check_interval
        )
        self.time_interval_seconds: float | None = None
        if isinstance(self.val_check_interval, bool):
            raise MisconfigurationError(
                f"Invalid `val_check_interval`: {self.val_check_interval!r}."
            )
        if isinstance(self.val_check_interval, int):
            if self.val_check_interval < 1:
                raise MisconfigurationError(
                    "Integer `val_check_interval` must be at least 1."
                )
        elif isinstance(self.val_check_interval, float):
            if not 0.0 <= self.val_check_interval <= 1.0:
                raise MisconfigurationError(
                    "Float `val_check_interval` must be in [0.0, 1.0]."
                )
        elif isinstance(self.val_check_interval, (str, timedelta, dict)):
            self.time_interval_seconds: float | None = self._parse_time_interval(
                self.val_check_interval
            )
        else:
            raise MisconfigurationError(
                f"Unsupported type for `val_check_interval`: {type(self.val_check_interval)!r}."
            )
        self.batch_interval: int | None = None
        self._validation_enabled: bool = False
        self._last_validation_time: float = 0.0

    def configure(
        self,
        max_batches: int | None,
        validation_enabled: bool,
        reset_time: bool
    ) -> None:
        # Resolves a concrete batch interval from loader size and public cadence,
        # validates sized and unsized constraints, and optionally resets time.
        self._validation_enabled: bool = validation_enabled
        if self.time_interval_seconds is not None:
            self.batch_interval: int | None = None
        elif isinstance(self.val_check_interval, int):
            self.batch_interval: int | None = self.val_check_interval
            if (
                validation_enabled
                and max_batches is not None
                and max_batches > 0
                and self.check_val_every_n_epoch is not None
                and self.batch_interval > max_batches
            ):
                raise ValueError(
                    f"`val_check_interval` ({self.batch_interval}) must be less than or equal "
                    f"to the number of training batches ({max_batches}). Set "
                    "`check_val_every_n_epoch=None` for a cross-epoch batch interval."
                )
        elif max_batches is None:
            if self.val_check_interval != 1.0:
                raise MisconfigurationError(
                    "An unsized training dataloader requires a time-based interval, an integer "
                    "interval, or `val_check_interval=1.0`."
                )
            self.batch_interval: int | None = None
        else:
            assert isinstance(self.val_check_interval, float)
            self.batch_interval: int | None = max(
                1,
                int(max_batches * self.val_check_interval)
            )
        if reset_time:
            self._last_validation_time: float = time.monotonic()

    def is_epoch_eligible(self, current_epoch: int) -> bool:
        # Applies one-based epoch cadence or permits every cross-epoch interval.
        return (
            self.check_val_every_n_epoch is None
            or (current_epoch + 1) % self.check_val_every_n_epoch == 0
        )

    def should_validate(
        self,
        current_epoch: int,
        batch_idx: int,
        total_batch_idx: int,
        is_last_batch: bool,
        should_stop: bool,
        can_stop_early: bool
    ) -> bool:
        # Evaluates enabled data, epoch cadence, cooperative stop, monotonic time,
        # final exhaustion, and local or cross-epoch batch coordinates in order.
        if not self._validation_enabled or not self.is_epoch_eligible(current_epoch):
            return False
        if should_stop and can_stop_early:
            return True
        if self.time_interval_seconds is not None:
            return time.monotonic() - self._last_validation_time >= self.time_interval_seconds
        if self.batch_interval is None:
            return is_last_batch
        current_iteration: int = (
            total_batch_idx
            if self.check_val_every_n_epoch is None
            else batch_idx
        )
        return (current_iteration + 1) % self.batch_interval == 0

    def mark_validation_end(self) -> None:
        # Starts the next time interval after validation work completes.
        if self.time_interval_seconds is not None:
            self._last_validation_time: float = time.monotonic()

    @property
    def enabled(self) -> bool:
        return self._validation_enabled

    @property
    def requires_unsized_epoch_end(self) -> bool:
        # Whether an unsized loader requires lookahead-confirmed exhaustion to
        # schedule its final validation pass.
        return (
            self._validation_enabled
            and self.time_interval_seconds is None
            and self.batch_interval is None
        )

    def _parse_time_interval(
        self,
        interval: str | timedelta | dict[str, int]
    ) -> float:
        # Converts timedelta, mapping, or DD:HH:MM:SS input to non-negative
        # seconds with strict field and type validation.
        if isinstance(interval, timedelta):
            seconds: float = interval.total_seconds()
        elif isinstance(interval, dict):
            try:
                seconds: float = timedelta(**interval).total_seconds()
            except (TypeError, OverflowError) as exception:
                raise MisconfigurationError(
                    f"Invalid time-based `val_check_interval`: {interval!r}."
                ) from exception
        else:
            parts: list[str] = interval.split(":")
            if len(parts) != 4:
                raise MisconfigurationError(
                    f"Invalid time format for `val_check_interval`: {interval!r}. "
                    "Expected 'DD:HH:MM:SS'."
                )
            try:
                components: list[int] = [int(component) for component in parts]
            except ValueError as exception:
                raise MisconfigurationError(
                    f"Non-integer component in `val_check_interval`: {interval!r}. "
                    "Use 'DD:HH:MM:SS'."
                ) from exception
            days: int = components[0]
            hours: int = components[1]
            minutes: int = components[2]
            seconds_component: int = components[3]
            if (
                days < 0
                or not 0 <= hours <= 23
                or not 0 <= minutes <= 59
                or not 0 <= seconds_component <= 59
            ):
                raise MisconfigurationError(
                    f"Invalid `val_check_interval` clock fields: {interval!r}."
                )
            seconds: float = timedelta(
                days=days,
                hours=hours,
                minutes=minutes,
                seconds=seconds_component
            ).total_seconds()
        if seconds < 0.0:
            raise MisconfigurationError(
                f"Time-based `val_check_interval` must be non-negative, got {seconds}."
            )
        return seconds
