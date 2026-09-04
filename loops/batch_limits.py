from typing import Literal

from syntheticmind.exceptions import MisconfigurationError

__all__: list[str] = [
    "normalize_batch_limit",
    "normalize_fast_dev_run",
    "normalize_sanity_steps",
    "resolve_batch_limit"
]

type BatchLimit = int | float
type BatchStage = Literal["train", "val", "test", "predict"]


def normalize_fast_dev_run(value: int | bool) -> int | bool:
    if not isinstance(value, (bool, int)):
        raise TypeError(
            f"`fast_dev_run` must be a bool or an int, got {value!r}."
        )
    if value < 0:
        raise MisconfigurationError(
            f"`fast_dev_run={value!r}` is invalid. Expected a non-negative integer or bool."
        )
    return True if value == 1 else value


def normalize_batch_limit(
    value: int | float | None,
    name: str
) -> BatchLimit:
    if value is None:
        return 1.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be an int, a float, or None, got {value!r}.")
    if 0 <= value <= 1:
        return value
    if value > 1:
        if isinstance(value, int):
            return value
        if value.is_integer():
            return int(value)
    raise MisconfigurationError(
        f"Invalid value {value!r} for `{name}`. Expected a float in [0.0, 1.0] "
        "or a non-negative integer."
    )


def normalize_sanity_steps(value: int | None) -> int | float:
    if value is None:
        return 2
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"`num_sanity_val_steps` must be an int or None, got {value!r}."
        )
    if value < -1:
        raise MisconfigurationError(
            "`num_sanity_val_steps` must be a non-negative integer, -1, or None."
        )
    return float("inf") if value == -1 else value


def resolve_batch_limit(
    total_batches: int | None,
    limit_batches: BatchLimit,
    stage: BatchStage
) -> int | None:
    if total_batches == 0 or limit_batches == 0:
        return 0
    if total_batches is None:
        if isinstance(limit_batches, int):
            return limit_batches
        if limit_batches == 1.0:
            return None
        raise MisconfigurationError(
            f"An unsized `{stage}_dataloader` requires `limit_{stage}_batches` "
            "to be 1.0 or an integer."
        )
    if isinstance(limit_batches, int):
        return min(total_batches, limit_batches)
    resolved_batches: int = int(total_batches * limit_batches)
    if resolved_batches == 0 and limit_batches > 0.0:
        minimum_fraction: float = 1.0 / total_batches
        raise MisconfigurationError(
            f"`limit_{stage}_batches={limit_batches}` selects fewer than one batch "
            f"from a dataloader with {total_batches} batches. Use at least "
            f"{minimum_fraction}."
        )
    return resolved_batches
