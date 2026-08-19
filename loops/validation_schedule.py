import time
from datetime import timedelta

from syntheticmind.utilities.exceptions import MisconfigurationError

__all__: list[str] = ["ValidationSchedule"]

type _ValidationInterval = int | float | str | timedelta | dict[str, int]


def _parse_time_interval(interval: str | timedelta | dict[str, int]) -> float:
    if isinstance(interval, timedelta):
        seconds: float = interval.total_seconds()
    elif isinstance(interval, dict):
        try:
            seconds = timedelta(**interval).total_seconds()
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
            days, hours, minutes, seconds_component = (
                int(component) for component in parts
            )
        except ValueError as exception:
            raise MisconfigurationError(
                f"Non-integer component in `val_check_interval`: {interval!r}. "
                "Use 'DD:HH:MM:SS'."
            ) from exception
        seconds = timedelta(
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


class ValidationSchedule:
    def __init__(
        self,
        val_check_interval: _ValidationInterval | None,
        check_val_every_n_epoch: int | None
    ) -> None:
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
            self.time_interval_seconds = _parse_time_interval(self.val_check_interval)
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
        self._validation_enabled = validation_enabled
        if self.time_interval_seconds is not None:
            self.batch_interval = None
        elif isinstance(self.val_check_interval, int):
            self.batch_interval = self.val_check_interval
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
            self.batch_interval = None
        else:
            self.batch_interval = max(1, int(max_batches * self.val_check_interval))
        if reset_time:
            self._last_validation_time = time.monotonic()

    def is_epoch_eligible(self, current_epoch: int) -> bool:
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
        if self.time_interval_seconds is not None:
            self._last_validation_time = time.monotonic()

    @property
    def enabled(self) -> bool:
        return self._validation_enabled

    @property
    def requires_unsized_epoch_end(self) -> bool:
        return (
            self._validation_enabled
            and self.time_interval_seconds is None
            and self.batch_interval is None
        )
