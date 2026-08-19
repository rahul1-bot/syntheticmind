# This module:
# 1. Accumulates scalar metric observations across steps and reduces them to a
#    single value per logging window
# 2. Supports mean, sum, min, and max reductions, with the mean optionally
#    weighted by per-step batch size
# 3. Provides reset semantics so one instance serves consecutive epochs or
#    evaluation windows without reallocation
#
# Design decisions:
# - The mean prefers batch-size weighting whenever any weights were supplied,
#   because averaging per-batch means directly would let a smaller final batch
#   bias the window mean
# - compute() on an empty window returns 0.0 instead of raising, so summary
#   emission never fails on a metric that logged nothing in the current window
# - Min and max trackers start at plus and minus infinity so the first
#   observation always establishes the initial extreme
# - State lives in plain Python floats; tensors are detached and converted at
#   the logging boundary, keeping the accumulator free of autograd references
# - __slots__ is declared because one accumulator exists per logged metric name
#   and long runs hold many instances simultaneously
#
# Author: Rahul Sawhney

from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass
from typing import Literal, Protocol, cast

import torch
import torch.distributed as dist

from syntheticmind.utilities.distributed import DistributedUtils
from syntheticmind.utilities.exceptions import MisconfigurationError

__all__: list[str] = ["MetricAccumulator"]

type ReductionMode = Literal["mean", "sum", "min", "max"]


class _Metric(Protocol):
    def compute(self) -> object: ...

    def reset(self) -> None: ...


class MetricAccumulator:
    # Streaming reducer for one named metric. Callers push observations with
    # update() during a window, read the reduced value with compute() at the
    # window boundary, and clear with reset() before the next window. The
    # reduction mode is fixed at construction so a metric cannot silently
    # change semantics mid-run.
    __slots__: tuple[str, ...] = (
        "_reduction",
        "_sum",
        "_count",
        "_min",
        "_max",
        "_weighted_sum",
        "_total_weight",
        "_sync_dist",
        "_sync_dist_group",
        "_rank_zero_only",
        "_metric"
    )

    def __init__(
        self,
        reduction: ReductionMode = "mean",
        sync_dist: bool = False,
        sync_dist_group: dist.ProcessGroup | None = None,
        rank_zero_only: bool = False
    ) -> None:
        # Fixes the reduction mode and zeroes every accumulator: plain sum and
        # count for unweighted means, running extremes for min and max, and the
        # weighted pair used when callers report batch sizes.
        self._reduction: ReductionMode = reduction
        self._sum: float = 0.0
        self._count: int = 0
        self._min: float = float("inf")
        self._max: float = float("-inf")
        self._weighted_sum: float = 0.0
        self._total_weight: float = 0.0
        self._sync_dist: bool = sync_dist
        self._sync_dist_group: dist.ProcessGroup | None = sync_dist_group
        self._rank_zero_only: bool = rank_zero_only
        self._metric: _Metric | None = None

    def update(self, value: float, batch_size: int | None = None) -> None:
        # Adds one observation: always advances the unweighted sum, count, and
        # extremes; additionally advances the weighted pair when a batch size is
        # given, which arms batch-size-weighted averaging in compute().
        self._sum += value
        self._count += 1
        if value < self._min:
            self._min: float = value
        if value > self._max:
            self._max: float = value
        if batch_size is not None:
            self._weighted_sum += value * batch_size
            self._total_weight += batch_size

    def update_metric(self, metric: object) -> None:
        if not callable(getattr(metric, "compute", None)) or not callable(
            getattr(metric, "reset", None)
        ):
            raise TypeError("A logged metric must define compute() and reset().")
        resolved_metric: _Metric = cast(_Metric, metric)
        if self._metric is not None and self._metric is not resolved_metric:
            raise MisconfigurationError(
                "A logged metric name cannot refer to different Metric instances in one epoch."
            )
        self._metric = resolved_metric

    def compute(self, device: torch.device | None = None) -> float:
        # Reduces the window: sum, min, and max read their accumulators
        # directly; the mean divides the weighted sum by total weight when any
        # weights were recorded and otherwise falls back to the unweighted mean
        # over update calls. An empty window reduces to 0.0.
        if self._metric is not None:
            value: object = self._metric.compute()
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                raise ValueError(
                    "The compute() result of a logged Metric must be a single-element Tensor."
                )
            return float(value.detach().item())
        if self._count == 0:
            return 0.0
        resolved_device: torch.device = device or torch.device("cpu")
        should_sync: bool = (
            self._sync_dist
            and not self._rank_zero_only
            and DistributedUtils.is_distributed()
        )
        if should_sync:
            if self._reduction == "mean":
                weighted: bool = self._total_weight > 0
                numerator: float = self._weighted_sum if weighted else self._sum
                denominator: float = self._total_weight if weighted else self._count
                state: torch.Tensor = torch.tensor(
                    [numerator, denominator],
                    dtype=torch.float64,
                    device=resolved_device
                )
                DistributedUtils.all_reduce(
                    state,
                    op=dist.ReduceOp.SUM,
                    group=self._sync_dist_group
                )
                return float((state[0] / state[1]).item())
            operation: dist.ReduceOp.RedOpType = {
                "sum": dist.ReduceOp.SUM,
                "min": dist.ReduceOp.MIN,
                "max": dist.ReduceOp.MAX
            }[self._reduction]
            state = torch.tensor(
                {
                    "sum": self._sum,
                    "min": self._min,
                    "max": self._max
                }[self._reduction],
                dtype=torch.float64,
                device=resolved_device
            )
            DistributedUtils.all_reduce(
                state,
                op=operation,
                group=self._sync_dist_group
            )
            return float(state.item())
        match self._reduction:
            case "sum":
                return self._sum
            case "min":
                return self._min
            case "max":
                return self._max
            case _:
                if self._total_weight > 0:
                    return self._weighted_sum / self._total_weight
                return self._sum / self._count

    def reset(self) -> None:
        # Returns every accumulator to its initial state so the instance can be
        # reused for the next window.
        self._sum: float = 0.0
        self._count: int = 0
        self._min: float = float("inf")
        self._max: float = float("-inf")
        self._weighted_sum: float = 0.0
        self._total_weight: float = 0.0
        if self._metric is not None:
            self._metric.reset()
            self._metric = None

    @property
    def reduction(self) -> ReductionMode:
        # Reduction mode fixed at construction.
        return self._reduction

    @property
    def count(self) -> int:
        # Number of observations recorded in the current window.
        return self._count


def _extract_batch_sizes(batch: object) -> Iterable[int | None]:
    if isinstance(batch, torch.Tensor):
        yield 1 if batch.ndim == 0 else batch.size(0)
        return
    if isinstance(batch, Mapping):
        for value in batch.values():
            yield from _extract_batch_sizes(value)
        return
    if isinstance(batch, Iterable) and not isinstance(batch, (str, bytes)):
        for value in batch:
            yield from _extract_batch_sizes(value)
        return
    if is_dataclass(batch) and not isinstance(batch, type):
        for field in fields(batch):
            yield from _extract_batch_sizes(getattr(batch, field.name))
        return
    yield None


def extract_batch_size(batch: object) -> int:
    inferred_batch_size: int | None = None
    try:
        for candidate in _extract_batch_sizes(batch):
            if candidate is None:
                continue
            if inferred_batch_size is None:
                inferred_batch_size = candidate
                continue
            if candidate != inferred_batch_size:
                break
    except RecursionError as exception:
        raise MisconfigurationError(
            "The batch size could not be inferred. Pass batch_size to self.log()."
        ) from exception
    if inferred_batch_size is None:
        raise MisconfigurationError(
            "The batch size could not be inferred. Pass batch_size to self.log()."
        )
    return cast(int, inferred_batch_size)
