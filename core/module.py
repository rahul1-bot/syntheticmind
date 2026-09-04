
from __future__ import annotations

import builtins
import numbers
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import IO, ClassVar, Literal, Self, cast, overload, override

import numpy as np
import syntheticmind
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torchmetrics import Metric, MetricCollection

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.hooks import CheckpointHooks, DataHooks, ModelHooks
from syntheticmind.core.hyperparameters import HyperparametersMixin
from syntheticmind.core.optimizer import (
    LRSchedulerTypeUnion,
    StatefulScheduler,
    SyntheticOptimizer,
    OptimizerLRScheduler
)
from syntheticmind.core.saving import _CheckpointLoader
from syntheticmind.loggers.logger import Logger
from syntheticmind.utilities.distributed import DistributedUtils
from syntheticmind.exceptions import MisconfigurationError
from syntheticmind.utilities.model_helpers import is_parameter_in_hook_signature
from syntheticmind.types import (
    Batch,
    BackwardInputs,
    GatherData,
    GatherResult,
    HyperparameterValue,
    MapLocation,
    ModelOutput,
    OptimizerClosure,
    OptimizerParameterGroup,
    Reduction,
    RunningStage,
    StepOutput,
    TrainingStepOutput
)

__all__: list[str] = ["LoggedMetric", "Module"]

type _MetricCandidate = nn.Module | ModelOutput
type _MetricReduction = (
    str
    | Callable[[torch.Tensor], torch.Tensor]
    | Callable[[Sequence[torch.Tensor]], torch.Tensor | int]
)
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
type _ModuleConversionArgument = (
    bool
    | str
    | torch.device
    | torch.dtype
    | torch.Tensor
    | torch.memory_format
    | None
)
type _Printable = ModelOutput | nn.Module | Path | BaseException
type _ModuleStateValue = (
    HyperparameterValue
    | LoggedMetric
    | syntheticmind.Trainer
    | DeviceMesh
    | dist.ProcessGroup
    | Logger
    | Callback
    | Callable[..., ModelOutput]
    | dict[str | int, _ModuleStateValue]
    | list[_ModuleStateValue]
    | tuple[_ModuleStateValue, ...]
    | set[_ModuleStateValue]
)


class Module(HyperparametersMixin, ModelHooks, DataHooks, CheckpointHooks, nn.Module):
    _checkpoint_hyperparameters_key: ClassVar[str] = "hyper_parameters"
    _checkpoint_hyperparameters_name: ClassVar[str] = "hparams_name"
    _checkpoint_hyperparameters_type: ClassVar[str] = "hparams_type"
    @override
    def __init__(self) -> None:
        super().__init__()
        self._trainer: syntheticmind.Trainer | None = None
        self._example_input_array: Batch | None = None
        self._automatic_optimization: bool = True
        self._strict_loading: bool | None = None
        self._current_epoch: int = 0
        self._global_step: int = 0
        self._logged_metrics: list[LoggedMetric] = []
        self._logged_metric_metadata: dict[str, _MetricMetadata] = {}
        self._metric_attributes: dict[int, str] | None = None
        self._logging_warning_keys: set[str] = set()
        self._running_stage: RunningStage | None = None
        self._device: torch.device = torch.device("cpu")
        self._device_mesh: DeviceMesh | None = None
        self._dtype: torch.dtype = torch.float32
        self._current_fx_name: str | None = None
        self._current_dataloader_idx: int | None = None
        self._param_requires_grad_state: dict[torch.nn.Parameter, bool] = {}

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | IO[bytes],
        map_location: MapLocation = None,
        hparams_file: str | Path | None = None,
        strict: bool | None = None,
        weights_only: bool | None = None,
        **kwargs: HyperparameterValue
    ) -> Self:
        return _CheckpointLoader.load(
            cls,
            checkpoint_path,
            map_location,
            hparams_file,
            strict,
            weights_only,
            kwargs
        )

    @contextmanager
    def _hook_context(self, hook_name: str) -> Iterator[None]:
        previous_hook_name: str | None = self._current_fx_name
        self._current_fx_name: str | None = hook_name
        try:
            yield
        finally:
            self._current_fx_name: str | None = previous_hook_name

    @override
    def forward(self, *args: Batch, **kwargs: Batch) -> ModelOutput:
        return super().forward(*args, **kwargs)

    def training_step(self, batch: Batch, batch_idx: int) -> TrainingStepOutput:
        raise NotImplementedError

    def validation_step(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> StepOutput:
        raise NotImplementedError

    def test_step(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> StepOutput:
        validation_step: Callable[..., StepOutput] = cast(
            Callable[..., StepOutput],
            self.validation_step
        )
        if is_parameter_in_hook_signature(
            validation_step,
            "dataloader_idx",
            explicit=True
        ):
            return validation_step(batch, batch_idx, dataloader_idx)
        return validation_step(batch, batch_idx)

    def predict_step(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> ModelOutput:
        del batch_idx, dataloader_idx
        return self(batch)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        warnings.warn(
            "configure_optimizers must be implemented to use optimization during Trainer.fit.",
            UserWarning,
            stacklevel=2
        )
        return None

    def configure_callbacks(self) -> Callback | Sequence[Callback]:
        return []

    @torch.compiler.disable
    def log(
        self,
        name: str,
        value: numbers.Number | torch.Tensor | Metric,
        prog_bar: bool = False,
        logger: bool | None = None,
        on_step: bool | None = None,
        on_epoch: bool | None = None,
        reduce_fx: _MetricReduction = "mean",
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_group: dist.ProcessGroup | None = None,
        add_dataloader_idx: bool = True,
        batch_size: int | None = None,
        metric_attribute: str | None = None,
        rank_zero_only: bool = False
    ) -> None:
        if not isinstance(name, str):
            raise TypeError(f"self.log() metric names must be strings, got {type(name).__name__}.")
        for parameter_name, parameter_value in (
            ("prog_bar", prog_bar),
            ("enable_graph", enable_graph),
            ("sync_dist", sync_dist),
            ("add_dataloader_idx", add_dataloader_idx),
            ("rank_zero_only", rank_zero_only)
        ):
            if type(parameter_value) is not bool:
                raise TypeError(f"self.log({parameter_name}=...) requires a bool.")
        for parameter_name, parameter_value in (
            ("logger", logger),
            ("on_step", on_step),
            ("on_epoch", on_epoch)
        ):
            if parameter_value is not None and type(parameter_value) is not bool:
                raise TypeError(f"self.log({parameter_name}=...) requires a bool or None.")
        if metric_attribute is not None and not isinstance(metric_attribute, str):
            raise TypeError("self.log(metric_attribute=...) requires a string or None.")
        if isinstance(value, Mapping):
            raise ValueError(f"self.log({name!r}, ...) does not support nested mappings.")
        is_metric_value: bool = self._is_metric(value)
        if not isinstance(value, (numbers.Number, torch.Tensor)) and not is_metric_value:
            raise ValueError(
                f"self.log({name!r}, ...) does not support values of type {type(value).__name__}."
            )
        if self._trainer is None:
            warning_key: str = "detached_module"
            if warning_key not in self._logging_warning_keys:
                warnings.warn(
                    "self.log() was called before the module was attached to a Trainer.",
                    RuntimeWarning,
                    stacklevel=2
                )
                self._logging_warning_keys.add(warning_key)
            return
        if self._trainer.barebones:
            warning_key = "barebones"
            if self._trainer.is_global_zero and warning_key not in self._logging_warning_keys:
                warnings.warn(
                    "self.log() is disabled because Trainer(barebones=True) is configured.",
                    RuntimeWarning,
                    stacklevel=2
                )
                self._logging_warning_keys.add(warning_key)
            return
        current_fx_name: str | None = self._current_fx_name
        if current_fx_name is None:
            raise MisconfigurationError(
                "self.log() must be called inside a trainer-managed hook or step method."
            )
        resolved_on_step: bool
        resolved_on_epoch: bool
        resolved_on_step, resolved_on_epoch = self._resolve_logging_schedule(
            hook_name=current_fx_name,
            on_step=on_step,
            on_epoch=on_epoch
        )
        if self._running_stage is None:
            raise MisconfigurationError(
                "self.log() can only be called from within a hook or step method "
                "during trainer.fit(), trainer.validate(), or trainer.test()."
            )
        if self._running_stage == "predicting":
            raise MisconfigurationError(
                "self.log() is not supported in predict_step or predict hooks. "
                "Use the return value of predict_step() instead."
            )
        if add_dataloader_idx and "/dataloader_idx_" in name:
            raise MisconfigurationError(
                f"self.log() received metric name {name!r} containing "
                "`/dataloader_idx_` while add_dataloader_idx=True."
            )
        if not resolved_on_step and not resolved_on_epoch:
            raise MisconfigurationError(
                "self.log(on_step=False, on_epoch=False) does not publish a metric."
            )
        reduction: Reduction
        if isinstance(reduce_fx, str):
            reduction_name: str = reduce_fx.lower()
            reduction_name = "mean" if reduction_name == "avg" else reduction_name
            if reduction_name not in {"mean", "sum", "min", "max"}:
                raise MisconfigurationError(
                    "self.log() supports only mean, sum, min, and max reductions. "
                    "Use a torchmetrics.Metric for custom reductions."
                )
            reduction = cast(Reduction, reduction_name)
        elif callable(reduce_fx):
            if reduce_fx is torch.mean:
                reduction = "mean"
            elif reduce_fx is torch.sum or reduce_fx is builtins.sum:
                reduction = "sum"
            elif reduce_fx is torch.min or reduce_fx is builtins.min:
                reduction = "min"
            elif reduce_fx is torch.max or reduce_fx is builtins.max:
                reduction = "max"
            else:
                raise MisconfigurationError(
                    "self.log() supports only mean, sum, min, and max reductions. "
                    "Use a torchmetrics.Metric for custom reductions."
                )
        else:
            raise MisconfigurationError(
                "self.log() supports only mean, sum, min, and max reductions. "
                "Use a torchmetrics.Metric for custom reductions."
            )
        if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
            raise MisconfigurationError("self.log(batch_size=...) requires a positive integer.")
        resolved_logger: bool = True if logger is None else logger
        if logger is True and not self._trainer.loggers and self._trainer.is_global_zero:
            warning_key = f"missing_logger:{name}"
            if warning_key not in self._logging_warning_keys:
                warnings.warn(
                    f"self.log({name!r}, ..., logger=True) was called without a configured logger.",
                    RuntimeWarning,
                    stacklevel=2
                )
                self._logging_warning_keys.add(warning_key)
        resolved_value: torch.Tensor | Metric
        if is_metric_value:
            if metric_attribute is None:
                if self._metric_attributes is None:
                    self._metric_attributes = {
                        id(metric): attribute
                        for attribute, metric in self.named_modules()
                        if self._is_metric(metric)
                    }
                metric_attribute = self._metric_attributes.get(id(value))
                if metric_attribute is None:
                    raise MisconfigurationError(
                        "A logged torchmetrics.Metric must be assigned to a Module attribute or "
                        "supplied with metric_attribute."
                    )
            else:
                try:
                    registered_metric: nn.Module = self.get_submodule(metric_attribute)
                except AttributeError as exception:
                    raise MisconfigurationError(
                        f"metric_attribute={metric_attribute!r} is not a registered Module attribute."
                    ) from exception
                if registered_metric is not value:
                    raise MisconfigurationError(
                        f"metric_attribute={metric_attribute!r} does not reference the logged Metric."
                    )
            resolved_value = cast(Metric, value)
        else:
            default_dtype: torch.dtype = torch.get_default_dtype()
            if default_dtype not in {torch.float32, torch.float64}:
                default_dtype = torch.float32
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError(
                        f"self.log({name!r}, ...) requires a single-element Tensor. "
                        f"Received shape {tuple(value.shape)}."
                    )
                resolved_value = value if enable_graph else value.detach().clone()
                resolved_value = resolved_value.squeeze()
                if not torch.is_floating_point(resolved_value):
                    warning_key = f"non_floating_metric:{current_fx_name}:{name}"
                    if (
                        self._trainer.is_global_zero
                        and warning_key not in self._logging_warning_keys
                    ):
                        warnings.warn(
                            f"self.log({name!r}, ...) converted a non-floating Tensor "
                            f"to {default_dtype} for reduction.",
                            RuntimeWarning,
                            stacklevel=2
                        )
                        self._logging_warning_keys.add(warning_key)
                if not torch.is_floating_point(resolved_value) or resolved_value.dtype not in {
                    torch.float32,
                    torch.float64
                }:
                    resolved_value = resolved_value.to(default_dtype)
            else:
                resolved_value = torch.tensor(
                    value,
                    device=self.device,
                    dtype=default_dtype
                ).squeeze()

        metric: LoggedMetric = LoggedMetric(
            name=name,
            hook_name=current_fx_name,
            value=resolved_value,
            progress_bar=prog_bar,
            logger=resolved_logger,
            on_step=resolved_on_step,
            on_epoch=resolved_on_epoch,
            reduction=reduction,
            enable_graph=enable_graph,
            sync_dist=sync_dist,
            sync_dist_group=sync_dist_group,
            add_dataloader_idx=add_dataloader_idx,
            batch_size=batch_size,
            metric_attribute=metric_attribute,
            rank_zero_only=rank_zero_only,
            dataloader_idx=self._current_dataloader_idx
        )
        previous_metadata: _MetricMetadata | None = self._logged_metric_metadata.get(
            metric.storage_key
        )
        if previous_metadata is not None and previous_metadata != metric.metadata:
            raise MisconfigurationError(
                f"self.log({name!r}, ...) was called twice in {current_fx_name} "
                "with different arguments."
            )
        self._logged_metric_metadata[metric.storage_key] = metric.metadata
        self._logged_metrics.append(metric)

    @torch.compiler.disable
    def log_dict(
        self,
        dictionary: Mapping[str, numbers.Number | torch.Tensor | Metric] | MetricCollection,
        prog_bar: bool = False,
        logger: bool | None = None,
        on_step: bool | None = None,
        on_epoch: bool | None = None,
        reduce_fx: _MetricReduction = "mean",
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_group: dist.ProcessGroup | None = None,
        add_dataloader_idx: bool = True,
        batch_size: int | None = None,
        rank_zero_only: bool = False
    ) -> None:
        if not isinstance(dictionary, Mapping) and not self._is_metric_collection(dictionary):
            raise TypeError(
                "self.log_dict() requires a mapping or torchmetrics.MetricCollection."
            )
        items: Iterator[tuple[str, numbers.Number | torch.Tensor | Metric]] = iter(
            dictionary.items()
        )
        log: Callable[..., None] = cast(Callable[..., None], self.log)
        name: str
        value: numbers.Number | torch.Tensor | Metric
        for name, value in items:
            log(
                name=name,
                value=value,
                prog_bar=prog_bar,
                logger=logger,
                on_step=on_step,
                on_epoch=on_epoch,
                reduce_fx=reduce_fx,
                enable_graph=enable_graph,
                sync_dist=sync_dist,
                sync_dist_group=sync_dist_group,
                add_dataloader_idx=add_dataloader_idx,
                batch_size=batch_size,
                rank_zero_only=rank_zero_only
            )

    @override
    def to(
        self,
        *args: _ModuleConversionArgument,
        **kwargs: _ModuleConversionArgument
    ) -> Module:
        result: Module = super().to(*args, **kwargs)
        param: torch.nn.Parameter | None = next(iter(result.parameters()), None)
        if param is not None:
            result._device: torch.device = param.device
            result._dtype: torch.dtype = param.dtype
        else:
            buf: torch.Tensor | None = next(iter(result.buffers()), None)
            if buf is not None:
                result._device: torch.device = buf.device
                result._dtype: torch.dtype = buf.dtype
            else:
                if args and isinstance(args[0], (torch.device, str)):
                    result._device: torch.device = torch.device(args[0])
                elif isinstance(kwargs.get("device"), (torch.device, str)):
                    device_argument: torch.device | str = cast(
                        torch.device | str,
                        kwargs["device"]
                    )
                    result._device: torch.device = torch.device(device_argument)
                if args and isinstance(args[0], torch.dtype):
                    result._dtype: torch.dtype = args[0]
                elif "dtype" in kwargs and isinstance(kwargs["dtype"], torch.dtype):
                    result._dtype: torch.dtype = kwargs["dtype"]
        return result

    @override
    def on_validation_model_zero_grad(self) -> None:
        self.zero_grad()

    @override
    def on_validation_model_eval(self) -> None:
        self.eval()

    @override
    def on_validation_model_train(self) -> None:
        self.train()

    @override
    def on_test_model_eval(self) -> None:
        self.eval()

    @override
    def on_test_model_train(self) -> None:
        self.train()

    @override
    def on_predict_model_eval(self) -> None:
        self.eval()

    def lr_scheduler_step(
        self,
        scheduler: StatefulScheduler,
        metric: torch.Tensor | float | None
    ) -> None:
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer | SyntheticOptimizer,
        optimizer_closure: OptimizerClosure | None = None
    ) -> None:
        del epoch, batch_idx
        optimizer.step(closure=optimizer_closure)

    def optimizer_zero_grad(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer
    ) -> None:
        del epoch, batch_idx
        optimizer.zero_grad()

    def manual_backward(
        self,
        loss: torch.Tensor,
        gradient: torch.Tensor | None = None,
        retain_graph: bool | None = None,
        create_graph: bool = False,
        inputs: BackwardInputs = None
    ) -> None:
        if self.automatic_optimization:
            raise MisconfigurationError(
                "manual_backward() requires automatic_optimization = False."
            )
        trainer: syntheticmind.Trainer = self.trainer
        if trainer.fit_loop.epoch_loop is None:
            raise RuntimeError("Trainer fit loop is not initialized.")
        trainer.fit_loop.epoch_loop.manual_backward(
            trainer,
            self,
            loss,
            gradient,
            retain_graph,
            create_graph,
            inputs
        )

    def backward(
        self,
        loss: torch.Tensor,
        gradient: torch.Tensor | None = None,
        retain_graph: bool | None = None,
        create_graph: bool = False,
        inputs: BackwardInputs = None
    ) -> None:
        loss.backward(
            gradient=gradient,
            retain_graph=retain_graph,
            create_graph=create_graph,
            inputs=inputs
        )

    def clip_gradients(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: int | builtins.float | None = None,
        gradient_clip_algorithm: str | None = None
    ) -> None:
        trainer: syntheticmind.Trainer = self.trainer
        if trainer.fit_loop.epoch_loop is None:
            raise RuntimeError("Trainer fit loop is not initialized.")
        raw_optimizer: torch.optim.Optimizer = (
            optimizer.optimizer if isinstance(optimizer, SyntheticOptimizer) else optimizer
        )
        resolved_gradient_clip_val: int | builtins.float
        if gradient_clip_val is None:
            resolved_gradient_clip_val: int | builtins.float = trainer.gradient_clip_val or 0.0
        elif (
            trainer.gradient_clip_val is not None
            and trainer.gradient_clip_val != gradient_clip_val
        ):
            raise MisconfigurationError(
                f"You have set `Trainer(gradient_clip_val={trainer.gradient_clip_val!r})`"
                f" and have passed `clip_gradients(gradient_clip_val={gradient_clip_val!r})`."
                " Please use only one of them."
            )
        else:
            resolved_gradient_clip_val: int | builtins.float = gradient_clip_val
        resolved_gradient_clip_algorithm: str
        if gradient_clip_algorithm is None:
            resolved_gradient_clip_algorithm: str = trainer.gradient_clip_algorithm or "norm"
        else:
            resolved_gradient_clip_algorithm: str = gradient_clip_algorithm.lower()
            if (
                trainer.gradient_clip_algorithm is not None
                and trainer.gradient_clip_algorithm != resolved_gradient_clip_algorithm
            ):
                raise MisconfigurationError(
                    "You have set "
                    f"`Trainer(gradient_clip_algorithm={trainer.gradient_clip_algorithm!r})`"
                    " and have passed "
                    "`clip_gradients(gradient_clip_algorithm="
                    f"{resolved_gradient_clip_algorithm!r})` Please use only one of them."
                )
        if not isinstance(resolved_gradient_clip_val, (int, float)):
            raise TypeError(
                "`gradient_clip_val` should be an int or a float. "
                f"Got {resolved_gradient_clip_val}."
            )
        if resolved_gradient_clip_algorithm not in {"norm", "value"}:
            raise MisconfigurationError(
                f"`gradient_clip_algorithm` {resolved_gradient_clip_algorithm} is invalid. "
                "Allowed algorithms: ('norm', 'value')."
            )
        trainer.fit_loop.epoch_loop.clip_gradients(
            optimizer=raw_optimizer,
            gradient_clip_val=resolved_gradient_clip_val,
            gradient_clip_algorithm=resolved_gradient_clip_algorithm
        )

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: int | builtins.float | None = None,
        gradient_clip_algorithm: str | None = None
    ) -> None:
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm
        )

    @torch.compiler.disable
    def toggle_optimizer(self, optimizer: torch.optim.Optimizer | SyntheticOptimizer) -> None:
        param_requires_grad_state: dict[torch.nn.Parameter, bool] = {}
        trainer_optimizer: torch.optim.Optimizer
        parameter_group: OptimizerParameterGroup
        parameter: torch.nn.Parameter
        for trainer_optimizer in self.trainer.optimizers:
            for parameter_group in trainer_optimizer.param_groups:
                for parameter in parameter_group["params"]:
                    if parameter in param_requires_grad_state:
                        continue
                    param_requires_grad_state[parameter] = parameter.requires_grad
                    parameter.requires_grad: bool = False
        for parameter_group in optimizer.param_groups:
            for parameter in parameter_group["params"]:
                parameter.requires_grad: bool = param_requires_grad_state[parameter]
        self._param_requires_grad_state: dict[
            torch.nn.Parameter,
            bool
        ] = param_requires_grad_state

    @torch.compiler.disable
    def untoggle_optimizer(self, optimizer: torch.optim.Optimizer | SyntheticOptimizer) -> None:
        raw_optimizer: torch.optim.Optimizer = (
            optimizer.optimizer if isinstance(optimizer, SyntheticOptimizer) else optimizer
        )
        trainer_optimizer: torch.optim.Optimizer
        parameter_group: OptimizerParameterGroup
        parameter: torch.nn.Parameter
        for trainer_optimizer in self.trainer.optimizers:
            if trainer_optimizer is raw_optimizer:
                continue
            for parameter_group in trainer_optimizer.param_groups:
                for parameter in parameter_group["params"]:
                    if parameter in self._param_requires_grad_state:
                        parameter.requires_grad: bool = self._param_requires_grad_state[parameter]
        self._param_requires_grad_state: dict[torch.nn.Parameter, bool] = {}

    @contextmanager
    def toggled_optimizer(
        self,
        optimizer: torch.optim.Optimizer | SyntheticOptimizer
    ) -> Iterator[None]:
        self.toggle_optimizer(optimizer)
        try:
            yield
        finally:
            self.untoggle_optimizer(optimizer)

    @overload
    def optimizers(
        self,
        use_synthetic_optimizer: Literal[True] = True
    ) -> SyntheticOptimizer | list[SyntheticOptimizer]: ...

    @overload
    def optimizers(
        self,
        use_synthetic_optimizer: Literal[False]
    ) -> torch.optim.Optimizer | list[torch.optim.Optimizer]: ...

    @overload
    def optimizers(
        self,
        use_synthetic_optimizer: bool
    ) -> (
        SyntheticOptimizer
        | list[SyntheticOptimizer]
        | torch.optim.Optimizer
        | list[torch.optim.Optimizer]
    ): ...

    def optimizers(
        self,
        use_synthetic_optimizer: bool = True
    ) -> (
        SyntheticOptimizer
        | list[SyntheticOptimizer]
        | torch.optim.Optimizer
        | list[torch.optim.Optimizer]
    ):
        trainer: syntheticmind.Trainer = self.trainer
        optimizers: list[SyntheticOptimizer] | list[torch.optim.Optimizer] = (
            trainer._synthetic_optimizers if use_synthetic_optimizer else trainer.optimizers
        )
        if len(optimizers) == 1:
            return optimizers[0]
        return optimizers

    def lr_schedulers(
        self
    ) -> LRSchedulerTypeUnion | list[LRSchedulerTypeUnion] | None:
        trainer: syntheticmind.Trainer = self.trainer
        if not trainer.lr_scheduler_configs:
            return None
        schedulers: list[LRSchedulerTypeUnion] = [
            configuration.scheduler for configuration in trainer.lr_scheduler_configs
        ]
        if len(schedulers) == 1:
            return schedulers[0]
        return schedulers

    def freeze(self) -> Self:
        for param in self.parameters():
            param.requires_grad: bool = False
        return self.eval()

    def unfreeze(self) -> Self:
        for param in self.parameters():
            param.requires_grad: bool = True
        return self.train()

    @override
    def cuda(self, device: int | torch.device | None = None) -> Module:
        result: Module = super().cuda(device)
        param: torch.nn.Parameter | None = next(iter(result.parameters()), None)
        if param is not None:
            result._device: torch.device = param.device
        else:
            buf: torch.Tensor | None = next(iter(result.buffers()), None)
            if buf is not None:
                result._device: torch.device = buf.device
            else:
                cuda_index: int = device if isinstance(device, int) else 0
                result._device: torch.device = torch.device("cuda", cuda_index)
        return result

    @override
    def cpu(self) -> Module:
        result: Module = super().cpu()
        result._device: torch.device = torch.device("cpu")
        return result

    @override
    def type(self, dst_type: str | torch.dtype) -> Module:
        result: Module = super().type(dst_type)
        param: torch.nn.Parameter | None = next(iter(result.parameters()), None)
        if param is not None:
            result._dtype: torch.dtype = param.dtype
        else:
            buf: torch.Tensor | None = next(iter(result.buffers()), None)
            if buf is not None:
                result._dtype: torch.dtype = buf.dtype
            elif isinstance(dst_type, torch.dtype):
                result._dtype: torch.dtype = dst_type
        return result

    @override
    def float(self) -> Module:
        result: Module = super().float()
        result._dtype: torch.dtype = torch.float32
        return result

    @override
    def half(self) -> Module:
        result: Module = super().half()
        result._dtype: torch.dtype = torch.float16
        return result

    @override
    def double(self) -> Module:
        result: Module = super().double()
        result._dtype = torch.float64
        return result

    @override
    def bfloat16(self) -> Module:
        result: Module = super().bfloat16()
        result._dtype = torch.bfloat16
        return result

    @property
    def trainer(self) -> syntheticmind.Trainer:
        if self._trainer is None:
            raise RuntimeError("Module is not attached to a Trainer. Call trainer.fit(module, ...) first")
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: syntheticmind.Trainer | None) -> None:
        for child in self.children():
            if isinstance(child, Module):
                child.trainer = trainer
        self._trainer = trainer

    @property
    def example_input_array(self) -> Batch | None:
        return self._example_input_array

    @example_input_array.setter
    def example_input_array(
        self,
        example_input_array: Batch | None
    ) -> None:
        self._example_input_array: Batch | None = example_input_array

    @property
    def automatic_optimization(self) -> bool:
        return self._automatic_optimization

    @automatic_optimization.setter
    def automatic_optimization(self, automatic_optimization: bool) -> None:
        self._automatic_optimization: bool = automatic_optimization

    @property
    def strict_loading(self) -> bool:
        return self._strict_loading in (None, True)

    @strict_loading.setter
    def strict_loading(self, strict_loading: bool) -> None:
        self._strict_loading: bool | None = strict_loading

    @property
    def device(self) -> torch.device:
        param: torch.nn.Parameter | None = next(iter(self.parameters()), None)
        if param is not None:
            return param.device
        buf: torch.Tensor | None = next(iter(self.buffers()), None)
        if buf is not None:
            return buf.device
        return self._device

    @property
    def device_mesh(self) -> DeviceMesh | None:
        return self._device_mesh

    @property
    def dtype(self) -> torch.dtype:
        param: torch.nn.Parameter | None = next(iter(self.parameters()), None)
        if param is not None:
            return param.dtype
        buf: torch.Tensor | None = next(iter(self.buffers()), None)
        if buf is not None:
            return buf.dtype
        return self._dtype

    @property
    def logger(self) -> Logger | None:
        if self._trainer is None:
            return None
        return self._trainer.logger

    @property
    def loggers(self) -> list[Logger]:
        if self._trainer is None:
            return []
        return self._trainer.loggers

    def print(
        self,
        *args: _Printable,
        sep: str | None = " ",
        end: str | None = "\n",
        file: IO[str] | None = None,
        flush: bool = False
    ) -> None:
        trainer: syntheticmind.Trainer = self.trainer
        if not trainer.is_global_zero:
            return
        progress_bar = trainer.progress_bar_callback
        if progress_bar is not None and progress_bar.is_enabled:
            progress_bar.print(*args, sep=sep, end=end, file=file, flush=flush)
            return
        builtins.print(*args, sep=sep, end=end, file=file, flush=flush)

    def all_gather(
        self,
        data: GatherData,
        group: dist.ProcessGroup | None = None,
        sync_grads: bool = False
    ) -> GatherResult:
        if type(sync_grads) is not bool:
            raise TypeError(
                f"`sync_grads` must be a bool, got {type(sync_grads).__name__}."
            )
        resolved_group: dist.ProcessGroup | None = (
            group if group is not None else dist.group.WORLD
        )

        gather_tensor: Callable[[torch.Tensor], torch.Tensor] = partial(
            self._gather_tensor,
            group=resolved_group,
            sync_grads=sync_grads
        )
        return self._transform_gather_data(data, self.device, gather_tensor)

    @property
    def current_epoch(self) -> int:
        return self.trainer.current_epoch if self._trainer is not None else 0

    @property
    def global_step(self) -> int:
        return self.trainer.global_step if self._trainer is not None else 0

    @property
    def global_rank(self) -> int:
        return self.trainer.global_rank if self._trainer is not None else 0

    @property
    def local_rank(self) -> int:
        return self.trainer.local_rank if self._trainer is not None else 0

    @property
    def on_gpu(self) -> bool:
        return self.device.type == "cuda"

    @property
    def logged_metrics(self) -> list[LoggedMetric]:
        return list(self._logged_metrics)

    @override
    def __getstate__(self) -> dict[str, _ModuleStateValue]:
        state: dict[str, _ModuleStateValue] = cast(
            dict[str, _ModuleStateValue],
            dict(self.__dict__)
        )
        state["_trainer"] = None
        return state

    def _reset_logged_metrics(self) -> None:
        self._logged_metrics.clear()
        self._current_dataloader_idx = None

    def _reset_logging_state(self) -> None:
        self._reset_logged_metrics()
        self._logged_metric_metadata.clear()

    def _gather_tensor(
        self,
        tensor: torch.Tensor,
        group: dist.ProcessGroup | None,
        sync_grads: bool
    ) -> torch.Tensor:
        return self.trainer.strategy.all_gather(
            tensor,
            group=group,
            sync_grads=sync_grads
        )

    def _transform_gather_data(
        self,
        data: GatherData,
        device: torch.device,
        transform: Callable[[torch.Tensor], torch.Tensor]
    ) -> GatherResult:
        if isinstance(data, torch.Tensor):
            tensor: torch.Tensor = data.to(device)
            return transform(tensor)
        if isinstance(data, np.ndarray):
            tensor: torch.Tensor = torch.from_numpy(data).to(device)
            return transform(tensor)
        if type(data) is bool:
            tensor: torch.Tensor = torch.tensor(data, device=device, dtype=torch.uint8)
            return transform(tensor)
        if type(data) is int:
            tensor: torch.Tensor = torch.tensor(data, device=device, dtype=torch.int)
            return transform(tensor)
        if type(data) is float:
            tensor: torch.Tensor = torch.tensor(data, device=device, dtype=torch.float)
            return transform(tensor)
        if isinstance(data, dict):
            transformed_mapping: dict[str | int, GatherResult] = {
                key: self._transform_gather_data(value, device, transform)
                for key, value in data.items()
            }
            return transformed_mapping
        if isinstance(data, list):
            transformed_list: list[GatherResult] = [
                self._transform_gather_data(value, device, transform)
                for value in data
            ]
            return transformed_list
        if isinstance(data, tuple):
            transformed_tuple: tuple[GatherResult, ...] = tuple(
                self._transform_gather_data(value, device, transform)
                for value in data
            )
            if hasattr(data, "_fields"):
                tuple_factory: Callable[
                    [tuple[GatherResult, ...]],
                    tuple[GatherResult, ...]
                ] = cast(
                    Callable[
                        [tuple[GatherResult, ...]],
                        tuple[GatherResult, ...]
                    ],
                    getattr(type(data), "_make")
                )
                return tuple_factory(transformed_tuple)
            return transformed_tuple
        raise TypeError(
            "all_gather data must be a tensor, NumPy array, bool, int, float, "
            "or a nested dict, list, or tuple containing those values, got "
            f"{type(data).__name__}."
        )


    @classmethod
    def _is_metric(cls, value: _MetricCandidate) -> bool:
        return isinstance(value, Metric)

    @classmethod
    def _is_metric_collection(cls, value: _MetricCandidate) -> bool:
        return isinstance(value, MetricCollection)


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
        return Module._is_metric(self.value)

    def output_name(self, on_step: bool) -> str:
        name: str = self.name
        if self.on_step and self.on_epoch:
            name = f"{name}_{'step' if on_step else 'epoch'}"
        if self.add_dataloader_idx and self.dataloader_idx is not None:
            name = f"{name}/dataloader_idx_{self.dataloader_idx}"
        return name

    def base_name(self) -> str:
        name: str = self.name
        if self.add_dataloader_idx and self.dataloader_idx is not None:
            name = f"{name}/dataloader_idx_{self.dataloader_idx}"
        return name

    def local_value(self) -> torch.Tensor:
        if self.is_metric:
            raise RuntimeError("A torchmetrics.Metric has no per-call scalar epoch value.")
        return cast(torch.Tensor, self.value)

    def compute_step(self, device: torch.device) -> torch.Tensor | None:
        if self.enable_graph:
            with torch.enable_grad():
                return self._compute_step(device)
        with torch.no_grad():
            return self._compute_step(device)

    def _compute_step(self, device: torch.device) -> torch.Tensor | None:
        if self._computed_step_value is not None:
            return self._computed_step_value
        value: numbers.Number | torch.Tensor | None = cast(
            numbers.Number | torch.Tensor | None,
            self.value
        )
        if self.is_metric:
            value: numbers.Number | torch.Tensor | None = cast(
                numbers.Number | torch.Tensor | None,
                getattr(self.value, "_forward_cache")
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
            tensor = value.clone()
        else:
            raise ValueError(
                f"The logged value for {self.name!r} could not be converted to a scalar."
            )
        if self.sync_dist and not self.rank_zero_only and DistributedUtils.is_distributed():
            if self.reduction == "mean":
                tensor = DistributedUtils.all_reduce_mean(
                    tensor,
                    group=self.sync_dist_group
                )
            else:
                operation: dist.ReduceOp.RedOpType = {
                    "sum": dist.ReduceOp.SUM,
                    "min": dist.ReduceOp.MIN,
                    "max": dist.ReduceOp.MAX
                }[self.reduction]
                tensor = DistributedUtils.all_reduce(
                    tensor,
                    op=operation,
                    group=self.sync_dist_group
                )
        if not self.enable_graph:
            tensor = tensor.detach()
        self._computed_step_value = tensor.squeeze()
        return self._computed_step_value

    def compute(self, device: torch.device) -> torch.Tensor:
        if self.enable_graph:
            with torch.enable_grad():
                return self._compute(device)
        with torch.no_grad():
            return self._compute(device)

    def _compute(self, device: torch.device) -> torch.Tensor:
        value: torch.Tensor | None = self.compute_step(device)
        if value is not None:
            return value
        metric: Metric = cast(Metric, self.value)
        computed: ModelOutput = cast(ModelOutput, metric.compute())
        if not isinstance(computed, torch.Tensor) or computed.numel() != 1:
            raise ValueError(
                f"The compute() result for logged Metric {self.name!r} must be a single-element Tensor."
            )
        if not self.enable_graph:
            computed = computed.detach()
        return computed.squeeze()

    def compute_epoch(self, device: torch.device) -> torch.Tensor:
        if self.enable_graph:
            with torch.enable_grad():
                return self._compute_epoch(device)
        with torch.no_grad():
            return self._compute_epoch(device)

    def _compute_epoch(self, device: torch.device) -> torch.Tensor:
        if not self.is_metric:
            value: torch.Tensor | None = self.compute_step(device)
            assert value is not None
            return value
        metric: Metric = cast(Metric, self.value)
        computed: ModelOutput = cast(ModelOutput, metric.compute())
        if not isinstance(computed, torch.Tensor) or computed.numel() != 1:
            raise ValueError(
                f"The compute() result for logged Metric {self.name!r} must be a single-element Tensor."
            )
        if not self.enable_graph:
            computed = computed.detach()
        return computed.squeeze()
