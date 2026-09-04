from collections.abc import Iterable, Iterator, Mapping
from dataclasses import Field, fields, is_dataclass
from typing import Annotated, ClassVar, Literal, cast

import torch
import torch.distributed as dist
from pydantic import BaseModel, ConfigDict, Field as ModelField, ValidationError
from torchmetrics import Metric

from syntheticmind.exceptions import MisconfigurationError
from syntheticmind.types import Batch, ModelOutput, Reduction, StateDict, TensorReducer

__all__: list[str] = ["MetricAccumulator"]

type _MetricCandidate = torch.nn.Module | ModelOutput


class _MetricAccumulatorCheckpoint(BaseModel):
    schema_version: Literal[1] = 1
    reduction: Reduction
    count: Annotated[int, ModelField(ge=0)]
    sum_value: torch.Tensor | None
    minimum: torch.Tensor | None
    maximum: torch.Tensor | None
    weighted_sum: torch.Tensor | None
    total_weight: Annotated[float, ModelField(ge=0.0)]
    is_metric: bool
    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
        strict=True,
        validate_default=True,
        allow_inf_nan=False
    )


class MetricAccumulator:
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
        "_enable_graph",
        "_metric"
    )

    def __init__(
        self,
        reduction: Reduction = "mean",
        sync_dist: bool = False,
        sync_dist_group: dist.ProcessGroup | None = None,
        rank_zero_only: bool = False,
        enable_graph: bool = False
    ) -> None:
        if reduction not in ("mean", "sum", "min", "max"):
            raise ValueError(f"Unsupported metric reduction {reduction!r}.")
        for name, value in (
            ("sync_dist", sync_dist),
            ("rank_zero_only", rank_zero_only),
            ("enable_graph", enable_graph)
        ):
            if type(value) is not bool:
                raise TypeError(f"MetricAccumulator {name} must be a bool.")
        self._reduction: Reduction = reduction
        self._sum: torch.Tensor | None = None
        self._count: int = 0
        self._min: torch.Tensor | None = None
        self._max: torch.Tensor | None = None
        self._weighted_sum: torch.Tensor | None = None
        self._total_weight: float = 0.0
        self._sync_dist: bool = sync_dist
        self._sync_dist_group: dist.ProcessGroup | None = sync_dist_group
        self._rank_zero_only: bool = rank_zero_only
        self._enable_graph: bool = enable_graph
        self._metric: Metric | None = None

    def update(
        self,
        value: torch.Tensor,
        batch_size: int | None = None,
        batch: Batch | None = None
    ) -> None:
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise ValueError("A metric observation must be a single-element tensor.")
        if batch_size is not None and (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("A metric batch size must be a positive integer.")
        resolved_batch_size: int | None = batch_size
        if (
            resolved_batch_size is None
            and batch is not None
            and self._reduction == "mean"
        ):
            resolved_batch_size: int | None = self._infer_batch_size(batch)
        scalar: torch.Tensor = value.squeeze()
        if self._enable_graph:
            with torch.enable_grad():
                self._update(scalar, resolved_batch_size)
            return
        with torch.no_grad():
            self._update(scalar.detach(), resolved_batch_size)

    def _update(self, value: torch.Tensor, batch_size: int | None) -> None:
        self._sum: torch.Tensor | None = (
            value if self._sum is None else self._sum + value
        )
        self._count: int = self._count + 1
        self._min: torch.Tensor | None = (
            value if self._min is None else torch.minimum(self._min, value)
        )
        self._max: torch.Tensor | None = (
            value if self._max is None else torch.maximum(self._max, value)
        )
        if batch_size is not None:
            self._weighted_sum: torch.Tensor | None = (
                value * batch_size
                if self._weighted_sum is None
                else self._weighted_sum + value * batch_size
            )
            self._total_weight: float = self._total_weight + batch_size

    def update_metric(self, metric: _MetricCandidate) -> None:
        if not isinstance(metric, Metric):
            raise TypeError("A logged metric must be a torchmetrics.Metric.")
        resolved_metric: Metric = metric
        if self._metric is not None and self._metric is not resolved_metric:
            raise MisconfigurationError(
                "A logged metric name cannot refer to different Metric instances "
                "in one epoch."
            )
        self._metric: Metric | None = resolved_metric

    def compute(
        self,
        device: torch.device | None = None,
        reduce_tensor: TensorReducer | None = None
    ) -> torch.Tensor:
        if self._enable_graph:
            with torch.enable_grad():
                return self._compute(device, reduce_tensor)
        with torch.no_grad():
            return self._compute(device, reduce_tensor).detach()

    def _compute(
        self,
        device: torch.device | None,
        reduce_tensor: TensorReducer | None
    ) -> torch.Tensor:
        if self._metric is not None:
            computed: ModelOutput = cast(ModelOutput, self._metric.compute())
            if not isinstance(computed, torch.Tensor) or computed.numel() != 1:
                raise ValueError(
                    "The compute() result of a logged Metric must be a "
                    "single-element tensor."
                )
            return computed.squeeze()
        if self._count == 0:
            return torch.tensor(0.0, device=device)
        should_sync: bool = self._sync_dist and not self._rank_zero_only
        assert self._sum is not None
        assert self._min is not None
        assert self._max is not None
        if should_sync:
            if reduce_tensor is None:
                raise RuntimeError(
                    "Distributed metric reduction requires the active Strategy reducer."
                )
            if self._reduction == "mean":
                weighted: bool = self._total_weight > 0
                numerator: torch.Tensor | None = (
                    self._weighted_sum if weighted else self._sum
                )
                assert numerator is not None
                denominator: float = (
                    self._total_weight if weighted else self._count
                )
                state: torch.Tensor = torch.stack(
                    (numerator, numerator.new_tensor(denominator))
                )
                state: torch.Tensor = reduce_tensor(
                    state,
                    self._sync_dist_group,
                    "sum"
                )
                return state[0] / state[1]
            state: torch.Tensor = {
                "sum": self._sum,
                "min": self._min,
                "max": self._max
            }[self._reduction].clone()
            return reduce_tensor(
                state,
                self._sync_dist_group,
                self._reduction
            )
        match self._reduction:
            case "sum":
                return self._sum
            case "min":
                return self._min
            case "max":
                return self._max
            case _:
                if self._total_weight > 0 and self._weighted_sum is not None:
                    return self._weighted_sum / self._total_weight
                return self._sum / self._count

    def reset(self) -> None:
        self._sum: torch.Tensor | None = None
        self._count: int = 0
        self._min: torch.Tensor | None = None
        self._max: torch.Tensor | None = None
        self._weighted_sum: torch.Tensor | None = None
        self._total_weight: float = 0.0
        if self._metric is not None:
            self._metric.reset()
            self._metric: Metric | None = None

    def _checkpoint_state(self) -> StateDict:
        state: _MetricAccumulatorCheckpoint = _MetricAccumulatorCheckpoint(
            reduction=self._reduction,
            count=self._count,
            sum_value=self._checkpoint_tensor(self._sum),
            minimum=self._checkpoint_tensor(self._min),
            maximum=self._checkpoint_tensor(self._max),
            weighted_sum=self._checkpoint_tensor(self._weighted_sum),
            total_weight=self._total_weight,
            is_metric=self._metric is not None
        )
        return cast(StateDict, state.model_dump(mode="python"))

    def _load_checkpoint_state(
        self,
        state: StateDict,
        device: torch.device,
        metric: Metric | None
    ) -> None:
        checkpoint: _MetricAccumulatorCheckpoint = self._parse_checkpoint_state(
            state
        )
        if checkpoint.reduction != self._reduction:
            raise MisconfigurationError(
                "Checkpoint epoch metric reduction does not match its identity."
            )
        if checkpoint.is_metric != (metric is not None):
            raise TypeError("Checkpoint epoch metric accumulator has the wrong kind.")
        self._validate_checkpoint_values(checkpoint)
        self._sum: torch.Tensor | None = self._restore_tensor(
            checkpoint.sum_value,
            device
        )
        self._count: int = checkpoint.count
        self._min: torch.Tensor | None = self._restore_tensor(
            checkpoint.minimum,
            device
        )
        self._max: torch.Tensor | None = self._restore_tensor(
            checkpoint.maximum,
            device
        )
        self._weighted_sum: torch.Tensor | None = self._restore_tensor(
            checkpoint.weighted_sum,
            device
        )
        self._total_weight: float = checkpoint.total_weight
        self._metric: Metric | None = metric

    @classmethod
    def _parse_checkpoint_state(
        cls,
        state: StateDict
    ) -> _MetricAccumulatorCheckpoint:
        del cls
        try:
            return _MetricAccumulatorCheckpoint.model_validate(state, strict=True)
        except ValidationError as exception:
            raise ValueError("Checkpoint epoch metric accumulator is invalid.") from exception

    @classmethod
    def _validate_checkpoint_values(
        cls,
        checkpoint: _MetricAccumulatorCheckpoint
    ) -> None:
        del cls
        scalar_values: tuple[torch.Tensor | None, ...] = (
            checkpoint.sum_value,
            checkpoint.minimum,
            checkpoint.maximum
        )
        checkpoint_values: tuple[torch.Tensor | None, ...] = (
            *scalar_values,
            checkpoint.weighted_sum
        )
        scalar_value: torch.Tensor | None
        for scalar_value in checkpoint_values:
            if scalar_value is not None and scalar_value.numel() != 1:
                raise ValueError("Checkpoint epoch accumulator tensors must be scalar.")
        if checkpoint.count == 0 and any(value is not None for value in scalar_values):
            raise ValueError("An empty epoch accumulator cannot contain scalar state.")
        if checkpoint.count > 0 and any(value is None for value in scalar_values):
            raise ValueError("A populated epoch accumulator requires complete scalar state.")
        if (checkpoint.total_weight > 0.0) != (checkpoint.weighted_sum is not None):
            raise ValueError("Epoch accumulator weight state is inconsistent.")

    def _checkpoint_tensor(self, value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return value.detach().cpu().clone()

    def _restore_tensor(
        self,
        value: torch.Tensor | None,
        device: torch.device
    ) -> torch.Tensor | None:
        if value is None:
            return None
        if value.numel() != 1:
            raise ValueError("Checkpoint epoch accumulator tensors must be scalar.")
        return value.detach().to(device).clone()

    @property
    def reduction(self) -> Reduction:
        return self._reduction

    @property
    def count(self) -> int:
        return self._count

    def _infer_batch_size(self, batch: Batch) -> int:
        inferred_batch_size: int | None = None
        try:
            for candidate in self._extract_batch_sizes(batch):
                if candidate is None:
                    continue
                if inferred_batch_size is None:
                    inferred_batch_size: int | None = candidate
                    continue
                if candidate != inferred_batch_size:
                    raise MisconfigurationError(
                        "Nested batch tensors have inconsistent leading dimensions. "
                        "Pass batch_size to self.log()."
                    )
        except RecursionError as exception:
            raise MisconfigurationError(
                "The batch size could not be inferred. Pass batch_size to self.log()."
            ) from exception
        if inferred_batch_size is None:
            raise MisconfigurationError(
                "The batch size could not be inferred. Pass batch_size to self.log()."
            )
        return inferred_batch_size

    def _extract_batch_sizes(self, batch: Batch) -> Iterator[int | None]:
        if isinstance(batch, torch.Tensor):
            yield 1 if batch.ndim == 0 else batch.size(0)
            return
        if isinstance(batch, Mapping):
            mapping_value: Batch
            for mapping_value in batch.values():
                yield from self._extract_batch_sizes(mapping_value)
            return
        if is_dataclass(batch) and not isinstance(batch, type):
            dataclass_field: Field[Batch]
            for dataclass_field in fields(batch):
                field_value: Batch = cast(
                    Batch,
                    getattr(batch, dataclass_field.name)
                )
                yield from self._extract_batch_sizes(field_value)
            return
        if isinstance(batch, Iterable) and not isinstance(batch, str | bytes):
            iterable_value: Batch
            for iterable_value in batch:
                yield from self._extract_batch_sizes(iterable_value)
            return
        yield None
