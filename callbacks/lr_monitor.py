# This module:
# 1. Defines LearningRateMonitor, the callback that publishes learning rate and
#    optional momentum and weight-decay values for every optimizer parameter
#    group
# 2. Derives stable metric identities from scheduler names, optimizer classes,
#    and client-supplied parameter-group names while preserving emission cadence
#    across checkpoint resumption
#
# Design decisions:
# - A client may force one global logging_interval or defer to each configured
#   scheduler interval; optimizers without schedulers default to step logging
# - Scheduler names identify their optimizer rather than the scheduler object,
#   so several schedulers attached to one optimizer must agree on one public
#   metric prefix
# - Repeated optimizer classes receive deterministic numeric suffixes, and
#   multi-group optimizers receive explicit group suffixes; duplicate final
#   names fail instead of overwriting telemetry
# - Step emission occurs at the before-optimizer-step boundary and suppresses
#   duplicates when several optimizer callbacks share a global step
# - Callback state stores only cadence coordinates. Optimizer topology and names
#   are rebuilt from the live resumed configuration and validated again
#
# Author: Rahul Sawhney

from typing import Literal, cast, override

import syntheticmind
import torch

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.core.optimizer import LRSchedulerConfig
from syntheticmind.primitives.exceptions import MisconfigurationError
from syntheticmind.primitives.types import (
    OptimizerParameterGroup,
    OptimizerStateValue,
    SchedulerInterval,
    StateDict,
    StateValue
)

__all__: list[str] = ["LearningRateMonitor"]


class LearningRateMonitor(Callback):
    # Optimizer-hyperparameter observer for client runs with configured loggers.
    # `lrs` retains the emitted learning-rate history per resolved metric name
    # for direct experiment inspection after fitting.
    def __init__(
        self,
        logging_interval: Literal["step", "epoch"] | None = None,
        log_momentum: bool = False,
        log_weight_decay: bool = False,
        log_key_prefix: str | None = None
    ) -> None:
        # Args:
        #     logging_interval: Forced ``"step"`` or ``"epoch"`` cadence, or
        #         None to follow scheduler intervals. Default: ``None``.
        #     log_momentum: Whether supported momentum or first-beta values are
        #         emitted beside learning rates. Default: ``False``.
        #     log_weight_decay: Whether configured weight decay is emitted.
        #         Default: ``False``.
        #     log_key_prefix: Optional prefix prepended to every resolved metric
        #         name. Default: ``None``.
        if logging_interval not in (None, "step", "epoch"):
            raise ValueError("`logging_interval` must be 'step', 'epoch', or None.")
        if type(log_momentum) is not bool:
            raise TypeError("`log_momentum` must be a bool.")
        if type(log_weight_decay) is not bool:
            raise TypeError("`log_weight_decay` must be a bool.")
        if log_key_prefix is not None and not isinstance(log_key_prefix, str):
            raise TypeError("`log_key_prefix` must be a string or None.")
        self.logging_interval: Literal["step", "epoch"] | None = logging_interval
        self.log_momentum: bool = log_momentum
        self.log_weight_decay: bool = log_weight_decay
        self.log_key_prefix: str = log_key_prefix or ""
        self.lrs: dict[str, list[float]] = {}
        self._optimizer_names: dict[int, str] = {}
        self._optimizer_intervals: dict[int, set[SchedulerInterval]] = {}
        self._last_logged_step: int = -1
        self._last_logged_epoch: int = -1
        self._state_restored: bool = False

    @property
    @override
    def state_key(self) -> str:
        # Qualifies state by every field that changes metric identity or cadence.
        return self._generate_state_key(
            logging_interval=self.logging_interval,
            log_momentum=self.log_momentum,
            log_weight_decay=self.log_weight_decay,
            log_key_prefix=self.log_key_prefix
        )

    @override
    def on_train_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        # Resolves optimizer identities only after configure_optimizers has been
        # materialized, retaining restored cadence coordinates for the first
        # resumed epoch.
        del module
        if not trainer.loggers:
            raise MisconfigurationError(
                "LearningRateMonitor requires at least one configured logger."
            )
        self._configure_optimizers(
            trainer.optimizers,
            trainer.lr_scheduler_configs
        )
        if not self._state_restored:
            self._last_logged_step: int = -1
            self._last_logged_epoch: int = -1
        self._state_restored: bool = False

    @override
    def on_before_optimizer_step(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        optimizer: torch.optim.Optimizer
    ) -> None:
        # Publishes step-cadence values immediately before the selected optimizer
        # steps, when its current parameter-group values are authoritative.
        del module
        if not self._should_log(optimizer, "step"):
            return
        if self._last_logged_step == trainer.global_step:
            return
        self._emit(trainer, [optimizer], trainer.global_step)
        self._last_logged_step: int = trainer.global_step

    @override
    def on_train_epoch_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        # Publishes one epoch-cadence snapshot before training batches begin.
        del module
        if self._last_logged_epoch == trainer.current_epoch:
            return
        optimizers: list[torch.optim.Optimizer] = [
            optimizer
            for optimizer in trainer.optimizers
            if self._should_log(optimizer, "epoch")
        ]
        if optimizers:
            self._emit(trainer, optimizers, trainer.global_step)
        self._last_logged_epoch: int = trainer.current_epoch

    @override
    def state_dict(self) -> StateDict:
        # Persists duplicate-suppression coordinates across resumption.
        return {
            "last_logged_step": self._last_logged_step,
            "last_logged_epoch": self._last_logged_epoch
        }

    @override
    def load_state_dict(self, state_dict: StateDict) -> None:
        # Restores cadence state strictly and marks it for preservation through
        # the next on_train_start configuration rebuild.
        step_value: StateValue | None = state_dict.get("last_logged_step")
        epoch_value: StateValue | None = state_dict.get("last_logged_epoch")
        if type(step_value) is not int or step_value < -1:
            raise TypeError("Learning-rate monitor step state must be an integer >= -1.")
        if type(epoch_value) is not int or epoch_value < -1:
            raise TypeError("Learning-rate monitor epoch state must be an integer >= -1.")
        self._last_logged_step: int = step_value
        self._last_logged_epoch: int = epoch_value
        self._state_restored: bool = True

    def _configure_optimizers(
        self,
        optimizers: list[torch.optim.Optimizer],
        scheduler_configs: list[LRSchedulerConfig]
    ) -> None:
        # Rebuilds stable public names and scheduler-derived interval sets from
        # the live optimizer graph, rejecting ambiguous client configurations.
        self._optimizer_names.clear()
        self._optimizer_intervals.clear()
        self.lrs.clear()
        type_counts: dict[str, int] = {}
        used_names: set[str] = set()
        optimizer: torch.optim.Optimizer
        for optimizer in optimizers:
            matching_configs: list[LRSchedulerConfig] = [
                configuration
                for configuration in scheduler_configs
                if configuration.scheduler.optimizer is optimizer
            ]
            configured_names: set[str] = {
                configuration.name
                for configuration in matching_configs
                if configuration.name is not None
            }
            if len(configured_names) > 1:
                raise MisconfigurationError(
                    f"Optimizer {type(optimizer).__name__} has conflicting scheduler "
                    f"logging names {sorted(configured_names)}."
                )
            optimizer_type: str = type(optimizer).__name__
            type_index: int = type_counts.get(optimizer_type, 0)
            type_counts[optimizer_type] = type_index + 1
            configured_name: str | None = next(iter(configured_names), None)
            base_name: str = configured_name or f"lr-{optimizer_type}"
            if configured_name is None and type_index > 0:
                base_name: str = f"{base_name}-{type_index}"
            base_name: str = f"{self.log_key_prefix}{base_name}"
            if base_name in used_names:
                raise MisconfigurationError(
                    f"Learning-rate logging name {base_name!r} is duplicated."
                )
            used_names.add(base_name)
            optimizer_identity: int = id(optimizer)
            self._optimizer_names[optimizer_identity] = base_name
            self._optimizer_intervals[optimizer_identity] = {
                configuration.interval
                for configuration in matching_configs
            }
            for metric_name in self._group_names(optimizer, base_name):
                if metric_name in self.lrs:
                    raise MisconfigurationError(
                        f"Learning-rate parameter-group name {metric_name!r} is duplicated."
                    )
                self.lrs[metric_name] = []

    def _group_names(
        self,
        optimizer: torch.optim.Optimizer,
        base_name: str
    ) -> list[str]:
        # Resolves parameter-group suffixes, preferring explicit client names and
        # falling back to deterministic one-based pg identifiers.
        parameter_groups: list[OptimizerParameterGroup] = cast(
            list[OptimizerParameterGroup],
            optimizer.param_groups
        )
        names: list[str] = []
        group_index: int
        group: OptimizerParameterGroup
        for group_index, group in enumerate(parameter_groups):
            group_name_value: OptimizerStateValue | None = group.get("name")
            if group_name_value is not None and not isinstance(group_name_value, str):
                raise MisconfigurationError(
                    "Optimizer parameter-group `name` must be a string when provided."
                )
            group_name: str | None = cast(str | None, group_name_value)
            if len(parameter_groups) > 1:
                suffix: str = group_name or f"pg{group_index + 1}"
                names.append(f"{base_name}/{suffix}")
            elif group_name is not None:
                names.append(f"{base_name}/{group_name}")
            else:
                names.append(base_name)
        return names

    def _should_log(
        self,
        optimizer: torch.optim.Optimizer,
        interval: SchedulerInterval
    ) -> bool:
        # Applies an explicit monitor-wide cadence first, then the optimizer's
        # scheduler intervals, then the step default for unscheduled optimizers.
        if self.logging_interval is not None:
            return self.logging_interval == interval
        configured_intervals: set[SchedulerInterval] = self._optimizer_intervals[
            id(optimizer)
        ]
        if not configured_intervals:
            return interval == "step"
        return interval in configured_intervals

    def _emit(
        self,
        trainer: syntheticmind.Trainer,
        optimizers: list[torch.optim.Optimizer],
        step: int
    ) -> None:
        # Merges values from the selected optimizers into one callback snapshot;
        # experiment loggers emit only on global zero.
        metrics: dict[str, float] = {}
        optimizer: torch.optim.Optimizer
        for optimizer in optimizers:
            metrics.update(self._optimizer_metrics(optimizer))
        if not metrics:
            return
        trainer._logger_connector.callback_metrics.update(
            {
                name: torch.tensor(value, device=trainer.strategy.root_device)
                for name, value in metrics.items()
            }
        )
        if trainer.is_global_zero:
            trainer._log_metrics(metrics, step=step)

    def _optimizer_metrics(
        self,
        optimizer: torch.optim.Optimizer
    ) -> dict[str, float]:
        # Reads live scalar hyperparameters and appends learning-rate history
        # under the same names sent to configured loggers.
        base_name: str = self._optimizer_names[id(optimizer)]
        group_names: list[str] = self._group_names(optimizer, base_name)
        parameter_groups: list[OptimizerParameterGroup] = cast(
            list[OptimizerParameterGroup],
            optimizer.param_groups
        )
        metrics: dict[str, float] = {}
        group: OptimizerParameterGroup
        metric_name: str
        for group, metric_name in zip(parameter_groups, group_names, strict=True):
            learning_rate: float = self._scalar(group.get("lr"), "learning rate")
            metrics[metric_name] = learning_rate
            if metric_name not in self.lrs:
                previous_name: str = base_name
                self.lrs[metric_name] = (
                    self.lrs.pop(previous_name)
                    if metric_name == f"{base_name}/pg1" and previous_name in self.lrs
                    else []
                )
            self.lrs[metric_name].append(learning_rate)
            if self.log_momentum:
                momentum: float | None = self._momentum(group)
                if momentum is not None:
                    metrics[f"{metric_name}-momentum"] = momentum
            if self.log_weight_decay:
                weight_decay: OptimizerStateValue | None = group.get("weight_decay")
                if weight_decay is not None:
                    metrics[f"{metric_name}-weight_decay"] = self._scalar(
                        weight_decay,
                        "weight decay"
                    )
        return metrics

    def _momentum(self, group: OptimizerParameterGroup) -> float | None:
        # Uses the first adaptive beta when present, otherwise conventional
        # momentum, matching the optimizer vocabulary exposed to clients.
        betas: OptimizerStateValue | None = group.get("betas")
        if isinstance(betas, list | tuple) and betas:
            return self._scalar(betas[0], "momentum beta")
        momentum: OptimizerStateValue | None = group.get("momentum")
        if momentum is None:
            return None
        return self._scalar(momentum, "momentum")

    def _scalar(
        self,
        value: OptimizerStateValue | None,
        name: str
    ) -> float:
        # Normalizes Python and scalar-tensor optimizer fields while rejecting
        # vector or foreign values that cannot form one metric sample.
        if type(value) is int or type(value) is float:
            return float(value)
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return float(value.detach().cpu().item())
        raise TypeError(f"Optimizer {name} must be a scalar, got {value!r}.")

    @override
    def __repr__(self) -> str:
        # Reports the full naming and cadence policy for run diagnostics.
        return (
            f"LearningRateMonitor(logging_interval={self.logging_interval!r}, "
            f"log_momentum={self.log_momentum}, "
            f"log_weight_decay={self.log_weight_decay}, "
            f"log_key_prefix={self.log_key_prefix!r})"
        )
