# This module:
# 1. Defines LoggedMetric, the internal observation produced by Module.log and
#    consumed by loop reduction, runtime views, callbacks, progress output, and
#    experiment loggers
# 2. Defines the strict checkpoint identity used to continue scalar and
#    torchmetrics epoch windows across a partial training epoch
#
# Design decisions:
# - A logged observation carries its originating hook and dataloader identity in
#   addition to its public name, preventing collisions between stages and loader
#   leaves before routing creates consumer-facing names
# - Scalar values and torchmetrics.Metric instances remain distinct: scalar
#   observations use SyntheticMind reduction, while Metric instances retain
#   their own state, compute operation, and synchronization semantics
# - Step computation is cached because progress, callback, logger, and epoch
#   consumers may inspect the same observation; the cached tensor preserves the
#   requested graph only when enable_graph=True
# - Distributed scalar reduction passes through the selected Strategy operation
#   supplied by the caller, while rank-zero-only observations never enter a
#   collective that other ranks cannot match
# - Partial-epoch checkpointing persists immutable observation identity plus
#   non-persistent torchmetrics tensor or tensor-list state and update count;
#   custom process groups and live autograd graphs are rejected because they
#   cannot cross process or persistence ownership safely
# - Metric persistence is enabled only while state is copied and its original
#   flags are restored in finally, so checkpoint inspection cannot permanently
#   change a client Metric
#
# Author: Rahul Sawhney

import numbers
from typing import Annotated, ClassVar, Literal, cast

import torch
import torch.distributed as dist
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from torchmetrics import Metric

from syntheticmind.primitives.exceptions import MisconfigurationError
from syntheticmind.primitives.types import (
    ModelOutput,
    Reduction,
    StateDict,
    StateValue,
    TensorReducer
)

__all__: list[str] = ["LoggedMetric"]

type _MetricMetadata = tuple[
    bool,
    bool,
    bool,
    bool,
    Reduction,
    bool,
    bool,
    dist.ProcessGroup | None,
    bool,
    str | None,
    bool
]
type _TorchMetricState = torch.Tensor | list[_TorchMetricState]


class _LoggedMetricCheckpoint(BaseModel):
    # Strict metadata schema identifying one resumable epoch observation. The
    # live scalar or Metric value travels separately because identity must be
    # validated before state is applied to a runtime object.
    schema_version: Literal[1] = 1
    name: Annotated[str, Field(min_length=1)]
    hook_name: Annotated[str, Field(min_length=1)]
    progress_bar: bool
    logger: bool
    on_step: bool
    on_epoch: bool
    reduction: Reduction
    sync_dist: bool
    add_dataloader_idx: bool
    batch_size: Annotated[int, Field(ge=1)] | None
    metric_attribute: str | None
    rank_zero_only: bool
    dataloader_idx: Annotated[int, Field(ge=0)] | None
    is_metric: bool
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        validate_default=True
    )


class LoggedMetric:
    # One Module.log observation with complete routing, reduction, distributed,
    # dataloader, graph, and checkpoint semantics. Clients create these only
    # through Module.log or Module.log_dict; loops and the logger connector own
    # consumption and publication.
    __slots__: tuple[str, ...] = (
        "name",
        "hook_name",
        "value",
        "progress_bar",
        "logger",
        "on_step",
        "on_epoch",
        "reduction",
        "enable_graph",
        "sync_dist",
        "sync_dist_group",
        "add_dataloader_idx",
        "batch_size",
        "metric_attribute",
        "rank_zero_only",
        "dataloader_idx",
        "storage_key",
        "metadata",
        "_computed_step_value"
    )

    def __init__(
        self,
        name: str,
        hook_name: str,
        value: torch.Tensor | Metric,
        progress_bar: bool,
        logger: bool,
        on_step: bool,
        on_epoch: bool,
        reduction: Reduction,
        enable_graph: bool,
        sync_dist: bool,
        sync_dist_group: dist.ProcessGroup | None,
        add_dataloader_idx: bool,
        batch_size: int | None,
        metric_attribute: str | None,
        rank_zero_only: bool,
        dataloader_idx: int | None
    ) -> None:
        # Binds immutable routing metadata to the current tensor or Metric value
        # and derives an internal storage key from hook, public name, and loader
        # identity. The step cache starts empty for lazy consumer-driven compute.
        self.name: str = name
        self.hook_name: str = hook_name
        self.value: torch.Tensor | Metric = value
        self.progress_bar: bool = progress_bar
        self.logger: bool = logger
        self.on_step: bool = on_step
        self.on_epoch: bool = on_epoch
        self.reduction: Reduction = reduction
        self.enable_graph: bool = enable_graph
        self.sync_dist: bool = sync_dist
        self.sync_dist_group: dist.ProcessGroup | None = sync_dist_group
        self.add_dataloader_idx: bool = add_dataloader_idx
        self.batch_size: int | None = batch_size
        self.metric_attribute: str | None = metric_attribute
        self.rank_zero_only: bool = rank_zero_only
        self.dataloader_idx: int | None = dataloader_idx
        dataloader_key: str = (
            f".{dataloader_idx}"
            if add_dataloader_idx and dataloader_idx is not None
            else ""
        )
        self.storage_key: str = f"{hook_name}.{name}{dataloader_key}"
        self.metadata: _MetricMetadata = (
            progress_bar,
            logger,
            on_step,
            on_epoch,
            reduction,
            enable_graph,
            sync_dist,
            sync_dist_group,
            add_dataloader_idx,
            metric_attribute,
            rank_zero_only
        )
        self._computed_step_value: torch.Tensor | None = None

    @property
    def is_metric(self) -> bool:
        # Whether reduction and epoch state belong to a torchmetrics.Metric
        # instance rather than SyntheticMind's scalar accumulator.
        return isinstance(self.value, Metric)

    def output_name(self, on_step: bool) -> str:
        # Builds the consumer-facing name, adding step or epoch suffixes only
        # when both cadences were requested and appending loader identity when
        # configured.
        name: str = self.name
        if self.on_step and self.on_epoch:
            name: str = f"{name}_{'step' if on_step else 'epoch'}"
        if self.add_dataloader_idx and self.dataloader_idx is not None:
            name: str = f"{name}/dataloader_idx_{self.dataloader_idx}"
        return name

    def base_name(self) -> str:
        # Builds the unsuffixed public identity used by callback and evaluation
        # views while retaining optional dataloader qualification.
        name: str = self.name
        if self.add_dataloader_idx and self.dataloader_idx is not None:
            name: str = f"{name}/dataloader_idx_{self.dataloader_idx}"
        return name

    def local_value(self) -> torch.Tensor:
        # Returns the rank-local scalar tensor for accumulator input. Stateful
        # Metric objects have no equivalent per-call epoch scalar and fail here.
        if self.is_metric:
            raise RuntimeError(
                "A torchmetrics.Metric has no per-call scalar epoch value."
            )
        if not isinstance(self.value, torch.Tensor):
            raise TypeError("A scalar logged value must be a tensor.")
        return self.value

    def compute_step(
        self,
        device: torch.device,
        reduce_tensor: TensorReducer
    ) -> torch.Tensor | None:
        # Computes the step observation under the requested graph policy. Metric
        # objects may return None before their first forward cache exists.
        if self.enable_graph:
            with torch.enable_grad():
                return self._compute_step(device, reduce_tensor)
        with torch.no_grad():
            return self._compute_step(device, reduce_tensor)

    def _compute_step(
        self,
        device: torch.device,
        reduce_tensor: TensorReducer
    ) -> torch.Tensor | None:
        # Normalizes the current scalar, applies optional Strategy reduction,
        # detaches when requested, and caches the squeezed result once.
        if self._computed_step_value is not None:
            return self._computed_step_value
        value: numbers.Number | torch.Tensor | None = self.value
        if self.is_metric:
            value: numbers.Number | torch.Tensor | None = getattr(
                self.value,
                "_forward_cache"
            )
            if value is None:
                return None
        if isinstance(value, numbers.Number):
            tensor: torch.Tensor = torch.tensor(value, device=device)
        elif isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"The logged value for {self.name!r} must contain one element."
                )
            tensor: torch.Tensor = value.clone()
        else:
            raise ValueError(
                f"The logged value for {self.name!r} could not be converted to a scalar."
            )
        if self.sync_dist and not self.rank_zero_only:
            tensor: torch.Tensor = reduce_tensor(
                tensor,
                self.sync_dist_group,
                self.reduction
            )
        if not self.enable_graph:
            tensor: torch.Tensor = tensor.detach()
        self._computed_step_value: torch.Tensor | None = tensor.squeeze()
        return self._computed_step_value

    def compute(
        self,
        device: torch.device,
        reduce_tensor: TensorReducer
    ) -> torch.Tensor:
        # Produces a scalar for immediate consumers, using the step cache when
        # available and falling back to Metric.compute for stateful metrics.
        if self.enable_graph:
            with torch.enable_grad():
                return self._compute(device, reduce_tensor)
        with torch.no_grad():
            return self._compute(device, reduce_tensor)

    def _compute(
        self,
        device: torch.device,
        reduce_tensor: TensorReducer
    ) -> torch.Tensor:
        # Resolves an immediate scalar observation first and falls back to the
        # registered stateful Metric when no scalar step cache exists.
        value: torch.Tensor | None = self.compute_step(device, reduce_tensor)
        if value is not None:
            return value
        return self._metric_value()

    def compute_epoch(
        self,
        device: torch.device,
        reduce_tensor: TensorReducer
    ) -> torch.Tensor:
        # Produces an epoch value from the local scalar observation or the
        # complete stateful Metric window under the configured graph policy.
        if self.enable_graph:
            with torch.enable_grad():
                return self._compute_epoch(device, reduce_tensor)
        with torch.no_grad():
            return self._compute_epoch(device, reduce_tensor)

    def _compute_epoch(
        self,
        device: torch.device,
        reduce_tensor: TensorReducer
    ) -> torch.Tensor:
        # Reuses the scalar accumulator's epoch result or delegates the complete
        # stateful window to Metric.compute without mixing the two state models.
        if not self.is_metric:
            value: torch.Tensor | None = self.compute_step(device, reduce_tensor)
            if value is None:
                raise RuntimeError("A scalar metric produced no step value.")
            return value
        return self._metric_value()

    def _metric_value(self) -> torch.Tensor:
        # Executes Metric.compute and enforces SyntheticMind's scalar publication
        # contract before optional detachment.
        if not isinstance(self.value, Metric):
            raise TypeError("The logged value is not a torchmetrics.Metric.")
        computed: ModelOutput = self.value.compute()
        if not isinstance(computed, torch.Tensor) or computed.numel() != 1:
            raise ValueError(
                f"The compute() result for logged Metric {self.name!r} must be "
                "a single-element tensor."
            )
        if not self.enable_graph:
            computed: torch.Tensor = computed.detach()
        return computed.squeeze()

    def _checkpoint_state(self) -> StateDict:
        # Serializes observation identity after rejecting graph-retaining or
        # custom-process-group state that cannot be resumed faithfully.
        if self.enable_graph:
            raise MisconfigurationError(
                "A graph-retaining epoch metric cannot cross a checkpoint boundary."
            )
        if self.sync_dist_group is not None:
            raise MisconfigurationError(
                "An epoch metric using a custom process group cannot cross a "
                "checkpoint boundary."
            )
        state: _LoggedMetricCheckpoint = _LoggedMetricCheckpoint(
            name=self.name,
            hook_name=self.hook_name,
            progress_bar=self.progress_bar,
            logger=self.logger,
            on_step=self.on_step,
            on_epoch=self.on_epoch,
            reduction=self.reduction,
            sync_dist=self.sync_dist,
            add_dataloader_idx=self.add_dataloader_idx,
            batch_size=self.batch_size,
            metric_attribute=self.metric_attribute,
            rank_zero_only=self.rank_zero_only,
            dataloader_idx=self.dataloader_idx,
            is_metric=self.is_metric
        )
        return cast(StateDict, state.model_dump(mode="python"))

    def _metric_checkpoint_state(self) -> tuple[StateDict, int]:
        # Captures torchmetrics state on CPU together with its update count while
        # restoring every original persistence flag afterward.
        if not isinstance(self.value, Metric):
            return {}, 0
        metric: Metric = self.value
        persistence: dict[str, bool] = dict(
            cast(dict[str, bool], getattr(metric, "_persistent"))
        )
        try:
            metric.persistent(True)
            state: StateDict = {
                name: self._clone_metric_state(
                    cast(StateValue, value),
                    torch.device("cpu")
                )
                for name, value in metric.state_dict().items()
            }
        finally:
            cast(
                dict[str, bool],
                getattr(metric, "_persistent")
            ).update(persistence)
        return state, metric.update_count

    def _load_metric_checkpoint_state(
        self,
        state: StateDict,
        update_count: int
    ) -> None:
        # Restores strict Metric state onto its live device, reinstates the update
        # count, and invalidates computed caches so continuation recomputes from
        # the restored window.
        if not isinstance(self.value, Metric):
            if state or update_count != 0:
                raise ValueError("A scalar epoch metric cannot contain Metric state.")
            return
        if isinstance(update_count, bool) or not isinstance(update_count, int):
            raise TypeError("Checkpoint Metric update count must be an integer.")
        if update_count < 0:
            raise ValueError("Checkpoint Metric update count must be non-negative.")
        metric: Metric = self.value
        persistence: dict[str, bool] = dict(
            cast(dict[str, bool], getattr(metric, "_persistent"))
        )
        try:
            metric.persistent(True)
            restored_state: StateDict = {
                name: self._clone_metric_state(value, metric.device)
                for name, value in state.items()
            }
            metric.load_state_dict(restored_state, strict=True)
        finally:
            cast(
                dict[str, bool],
                getattr(metric, "_persistent")
            ).update(persistence)
        setattr(metric, "_update_count", update_count)
        setattr(metric, "_computed", None)
        setattr(metric, "_forward_cache", None)

    @classmethod
    def _validate_metric_checkpoint_state(cls, state: StateDict) -> None:
        # Validates every state leaf before a live Metric is mutated.
        value: StateValue
        for value in state.values():
            cls._validate_metric_state_value(value)

    @classmethod
    def _validate_metric_state_value(
        cls,
        value: StateValue
    ) -> None:
        # Restricts resumable Metric state to tensors and recursively nested
        # tensor lists, the forms verified by current TorchMetrics support.
        if isinstance(value, torch.Tensor):
            return
        if isinstance(value, list):
            item: StateValue
            for item in value:
                cls._validate_metric_state_value(item)
            return
        raise TypeError("Checkpoint Metric state supports tensors and tensor lists only.")

    def _clone_metric_state(
        self,
        value: StateValue,
        device: torch.device
    ) -> _TorchMetricState:
        # Detaches, clones, and moves supported Metric state without retaining
        # checkpoint storage or autograd aliases.
        if isinstance(value, torch.Tensor):
            return value.detach().to(device).clone()
        if isinstance(value, list):
            cloned_items: list[_TorchMetricState] = []
            item: StateValue
            for item in value:
                cloned_items.append(self._clone_metric_state(item, device))
            return cloned_items
        raise TypeError("Metric state supports tensors and tensor lists only.")

    @classmethod
    def _parse_checkpoint_state(
        cls,
        state: StateDict
    ) -> _LoggedMetricCheckpoint:
        # Validates strict schema and rejects unknown, missing, or mistyped
        # identity fields before reconstruction.
        del cls
        try:
            return _LoggedMetricCheckpoint.model_validate(state, strict=True)
        except ValidationError as exception:
            raise ValueError("Checkpoint epoch metric identity is invalid.") from exception

    @classmethod
    def _from_checkpoint_state(
        cls,
        state: _LoggedMetricCheckpoint,
        value: torch.Tensor | Metric
    ) -> LoggedMetric:
        # Reconstructs the observation around a separately supplied scalar or
        # Metric object and verifies that its persisted kind still matches.
        if state.is_metric != isinstance(value, Metric):
            raise TypeError("Checkpoint epoch metric value has the wrong kind.")
        return cls(
            name=state.name,
            hook_name=state.hook_name,
            value=value,
            progress_bar=state.progress_bar,
            logger=state.logger,
            on_step=state.on_step,
            on_epoch=state.on_epoch,
            reduction=state.reduction,
            enable_graph=False,
            sync_dist=state.sync_dist,
            sync_dist_group=None,
            add_dataloader_idx=state.add_dataloader_idx,
            batch_size=state.batch_size,
            metric_attribute=state.metric_attribute,
            rank_zero_only=state.rank_zero_only,
            dataloader_idx=state.dataloader_idx
        )
