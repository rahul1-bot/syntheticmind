import numbers

import torch
import torch.distributed as dist
from torchmetrics import Metric

from syntheticmind.types import ModelOutput, Reduction, TensorReducer

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


class LoggedMetric:
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
        return isinstance(self.value, Metric)

    def output_name(self, on_step: bool) -> str:
        name: str = self.name
        if self.on_step and self.on_epoch:
            name: str = f"{name}_{'step' if on_step else 'epoch'}"
        if self.add_dataloader_idx and self.dataloader_idx is not None:
            name: str = f"{name}/dataloader_idx_{self.dataloader_idx}"
        return name

    def base_name(self) -> str:
        name: str = self.name
        if self.add_dataloader_idx and self.dataloader_idx is not None:
            name: str = f"{name}/dataloader_idx_{self.dataloader_idx}"
        return name

    def local_value(self) -> torch.Tensor:
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
        value: torch.Tensor | None = self.compute_step(device, reduce_tensor)
        if value is not None:
            return value
        return self._metric_value()

    def compute_epoch(
        self,
        device: torch.device,
        reduce_tensor: TensorReducer
    ) -> torch.Tensor:
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
        if not self.is_metric:
            value: torch.Tensor | None = self.compute_step(device, reduce_tensor)
            if value is None:
                raise RuntimeError("A scalar metric produced no step value.")
            return value
        return self._metric_value()

    def _metric_value(self) -> torch.Tensor:
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
