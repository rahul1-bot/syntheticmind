# This module:
# 1. Defines the validated optimizer and scheduler configuration models consumed
#    by the optimizer and scheduler builders
# 2. Defines the definition unions that additionally admit pre-constructed
#    torch optimizer and scheduler objects
# 3. Defines OptimizationConfiguration, the single return contract of
#    Module.configure_optimizers, with cross-field validation of the
#    optimizer-to-scheduler structure
#
# Design decisions:
# - Both configuration models are frozen value objects, so a configuration
#   cannot be mutated after validation and every consumer observes the same
#   validated values
# - Each configuration carries the superset of fields across its families
#   (betas for the Adam family, momentum and nesterov for SGD, and the
#   per-schedule fields for the eight schedule families); the builders project
#   only the fields relevant to the selected name, which keeps one model
#   sufficient for every family
# - The plateau schedule is validated at construction to require an epoch
#   interval and a monitor metric, because it can only advance on a monitored
#   validation value produced at epoch boundaries
# - The definition unions accept already-constructed torch objects so advanced
#   callers can supply instances the builders cannot express, at the cost of
#   bypassing configuration validation for those instances
# - OptimizationConfiguration validates structural pairing at construction: a
#   scheduler list requires an optimizer list of matching length, and a single
#   scheduler configuration cannot be paired with multiple optimizers, so
#   mismatches surface before training starts rather than mid-run
#
# Author: Rahul Sawhney

import warnings
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, fields
from typing import (
    ClassVar,
    Generator,
    Literal,
    NotRequired,
    Protocol,
    Required,
    TypedDict,
    overload,
    override,
    runtime_checkable
)

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR, ReduceLROnPlateau

from syntheticmind.core.protocols import (
    OptimizerEpochLoopProtocol,
    OptimizerModuleProtocol,
    OptimizerTrainerProtocol
)
from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.types import SchedulerInterval

__all__: list[str] = [
    "SyntheticOptimizer",
    "LRSchedulerConfig",
    "LRSchedulerConfigType",
    "LRSchedulerType",
    "LRSchedulerTypeTuple",
    "LRSchedulerTypeUnion",
    "OptimizerConfig",
    "OptimizerLRSchedulerConfig",
    "OptimizerLRScheduler"
]


class _OptimizerFactoryConfig(BaseModel):
    # Validated optimizer selection. The name selects the torch.optim family;
    # lr is mandatory and must be positive, while the remaining fields carry
    # family-specific hyperparameters that the builder projects per family.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    name: Literal["adamw", "sgd", "adam"] = "adamw"
    lr: float = Field(gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    betas: tuple[float, float] = (0.9, 0.999)
    momentum: float = Field(default=0.0, ge=0.0)
    nesterov: bool = False


class _SchedulerFactoryConfig(BaseModel):
    # Validated scheduler selection. The name selects the schedule family, the
    # interval declares whether the schedule advances per optimizer step or
    # per epoch, and monitor names the metric consumed by the plateau family.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    name: Literal[
        "poly",
        "cosine",
        "cosine_warm_restarts",
        "step",
        "multistep",
        "exponential",
        "reduce_on_plateau",
        "one_cycle",
    ] = "poly"
    interval: SchedulerInterval = "step"
    monitor: str | None = None

    warmup_epochs: int = Field(default=0, ge=0)
    power: float = Field(default=0.9, gt=0.0)
    t_max: int | None = None
    eta_min: float = Field(default=0.0, ge=0.0)
    step_size: int = Field(default=10, gt=0)
    milestones: list[int] | None = None
    gamma: float = Field(default=0.1, gt=0.0)
    patience: int = Field(default=10, gt=0)
    factor: float = Field(default=0.1, gt=0.0)
    max_lr: float | None = None

    @model_validator(mode="after")
    def _validate_reduce_on_plateau(self) -> _SchedulerFactoryConfig:
        # Enforces the plateau family's structural requirements at
        # construction: the schedule advances on a monitored validation metric,
        # which exists only at epoch boundaries, so both the epoch interval and
        # a monitor name are mandatory.
        if self.name == "reduce_on_plateau":
            if self.interval != "epoch":
                raise ValueError(
                    f"ReduceLROnPlateau requires interval='epoch', got '{self.interval}'"
                )
            if self.monitor is None:
                raise ValueError(
                    "ReduceLROnPlateau requires a monitor metric. Set monitor in SchedulerConfig."
                )
        return self


type _OptimizerDefinition = _OptimizerFactoryConfig | torch.optim.Optimizer

type _SchedulerDefinition = _SchedulerFactoryConfig | torch.optim.lr_scheduler.LRScheduler


class _OptimizationConfiguration(BaseModel):
    # Return contract of Module.configure_optimizers. Carries one optimizer or
    # an ordered optimizer list, with an optional scheduler or scheduler list
    # paired by position. Arbitrary types are allowed because the definition
    # unions admit live torch objects alongside validated configurations.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    optimizer: _OptimizerDefinition | list[_OptimizerDefinition]
    scheduler: _SchedulerDefinition | list[_SchedulerDefinition] | None = None

    @model_validator(mode="after")
    def validate_optimizer_scheduler_structure(self) -> _OptimizationConfiguration:
        # Validates the pairing structure between optimizers and schedulers: an
        # empty optimizer list is rejected, a scheduler list demands an
        # optimizer list of exactly matching length, and a single scheduler
        # configuration cannot serve multiple optimizers because the pairing
        # would be ambiguous.
        if isinstance(self.optimizer, list):
            if not self.optimizer:
                raise ValueError("Expected at least one optimizer configuration")
            optimizer_count: int = len(self.optimizer)
        else:
            optimizer_count: int = 1

        if isinstance(self.scheduler, list):
            if optimizer_count == 1:
                raise ValueError(
                    "A scheduler list requires a matching optimizer list"
                )
            if len(self.scheduler) != optimizer_count:
                raise ValueError(
                    "The number of scheduler configurations must match the number of optimizer configurations"
                )
        elif isinstance(self.scheduler, _SchedulerFactoryConfig) and optimizer_count > 1:
            raise ValueError(
                "When multiple optimizers are configured, scheduler must be None or a list of scheduler configurations"
            )
        return self


@runtime_checkable
class _StatefulScheduler(Protocol):
    optimizer: torch.optim.Optimizer

    def step(self, *args: object, **kwargs: object) -> object: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state_dict: dict[str, object]) -> None: ...


LRSchedulerTypeTuple: tuple[type[LRScheduler], type[ReduceLROnPlateau]] = (
    LRScheduler,
    ReduceLROnPlateau
)

type LRSchedulerTypeUnion = LRScheduler | ReduceLROnPlateau

type LRSchedulerType = type[LRScheduler] | type[ReduceLROnPlateau]

type _SchedulerInstance = LRSchedulerTypeUnion | _StatefulScheduler


@dataclass
class LRSchedulerConfig:
    scheduler: LRSchedulerTypeUnion
    name: str | None = None
    interval: str = "epoch"
    frequency: int = 1
    reduce_on_plateau: bool = False
    monitor: str | None = None
    strict: bool = True


class LRSchedulerConfigType(TypedDict, total=False):
    scheduler: Required[LRSchedulerTypeUnion]
    name: str | None
    interval: str
    frequency: int
    reduce_on_plateau: bool
    monitor: str | None
    strict: bool


class OptimizerConfig(TypedDict):
    optimizer: torch.optim.Optimizer


class OptimizerLRSchedulerConfig(TypedDict):
    optimizer: torch.optim.Optimizer
    lr_scheduler: LRSchedulerTypeUnion | LRSchedulerConfigType
    monitor: NotRequired[str]


type OptimizerLRScheduler = (
    torch.optim.Optimizer
    | Sequence[torch.optim.Optimizer]
    | tuple[
        Sequence[torch.optim.Optimizer],
        Sequence[LRSchedulerTypeUnion | LRSchedulerConfig]
    ]
    | OptimizerConfig
    | OptimizerLRSchedulerConfig
    | Sequence[OptimizerConfig]
    | Sequence[OptimizerLRSchedulerConfig]
    | None
)


class SyntheticOptimizer:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self._optimizer: torch.optim.Optimizer = optimizer
        self._trainer: OptimizerTrainerProtocol | None = None
        self._module: OptimizerModuleProtocol | None = None
        self._on_before_step: Callable[[], None] = do_nothing_closure
        self._on_after_step: Callable[[], None] = do_nothing_closure
        self._automatic_step: bool = False
        self.__class__: type[SyntheticOptimizer] = type(
            "Synthetic" + optimizer.__class__.__name__,
            (self.__class__, optimizer.__class__),
            {}
        )

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @contextmanager
    def toggle_model(self, sync_grad: bool = True) -> Generator[None, None, None]:
        trainer: OptimizerTrainerProtocol | None = self._trainer
        module: OptimizerModuleProtocol | None = self._module
        if trainer is None or module is None:
            raise RuntimeError("The SyntheticOptimizer is not attached to a Trainer and Module.")
        model: torch.nn.Module | None = trainer.strategy.model
        synchronization_context: AbstractContextManager[None] = nullcontext()
        if not sync_grad and isinstance(model, DistributedDataParallel):
            synchronization_context: AbstractContextManager[None] = model.no_sync()
        with synchronization_context:
            with module.toggled_optimizer(self):
                yield

    def step(
        self,
        closure: Callable[[], object] | None = None,
        **kwargs: object
    ) -> object:
        if closure is not None and not callable(closure):
            raise MisconfigurationError(
                "When optimizer.step(closure) is called, the closure must be callable."
            )
        if closure is None:
            closure: Callable[[], object] | None = do_nothing_closure
        trainer: OptimizerTrainerProtocol | None = self._trainer
        module: OptimizerModuleProtocol | None = self._module
        if trainer is None or module is None:
            raise RuntimeError("The SyntheticOptimizer is not attached to a Trainer and Module.")
        epoch_loop: OptimizerEpochLoopProtocol | None = trainer.fit_loop.epoch_loop
        if epoch_loop is None:
            raise RuntimeError("Trainer fit loop is not initialized.")
        self._on_before_step()
        output: object = epoch_loop.optimizer_step(
            trainer,
            module,
            self._optimizer,
            closure,
            kwargs,
            automatic=self._automatic_step,
            increment_global_step=not self._automatic_step
        )
        self._on_after_step()
        return output

    @classmethod
    def _to_synthetic_optimizer(
        cls,
        optimizer: torch.optim.Optimizer | SyntheticOptimizer,
        trainer: OptimizerTrainerProtocol,
        module: OptimizerModuleProtocol
    ) -> SyntheticOptimizer:
        synthetic_optimizer: SyntheticOptimizer = (
            optimizer if isinstance(optimizer, SyntheticOptimizer) else cls(optimizer)
        )
        synthetic_optimizer._trainer: OptimizerTrainerProtocol | None = trainer
        synthetic_optimizer._module: OptimizerModuleProtocol | None = module
        return synthetic_optimizer

    def __getattr__(self, name: str) -> object:
        return getattr(self._optimizer, name)


def do_nothing_closure() -> None:
    return None


class _MockOptimizer(torch.optim.Optimizer):
    def __init__(self) -> None:
        super().__init__([torch.zeros(1)], {})

    @override
    def add_param_group(self, param_group: dict[str, object]) -> None:
        del param_group

    @override
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        del state_dict

    @override
    def state_dict(self) -> dict[str, object]:
        return {}

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        if closure is None:
            return None
        return closure()

    @override
    def zero_grad(self, set_to_none: bool = True) -> None:
        del set_to_none

    @override
    def __repr__(self) -> str:
        return "No Optimizer"


def _init_optimizers_and_lr_schedulers(
    optim_conf: object,
    automatic_optimization: bool,
    lr_scheduler_step_overridden: bool = False
) -> tuple[list[torch.optim.Optimizer], list[LRSchedulerConfig]]:
    if optim_conf is None:
        warnings.warn(
            "Module.configure_optimizers returned None; this fit will run with no optimizer.",
            UserWarning,
            stacklevel=3
        )
        optim_conf: object = _MockOptimizer()
    optimizer_configuration: tuple[
        list[torch.optim.Optimizer],
        list[object],
        str | None
    ] = _configure_optimizers(optim_conf)
    optimizers: list[torch.optim.Optimizer] = optimizer_configuration[0]
    schedulers: list[object] = optimizer_configuration[1]
    monitor: str | None = optimizer_configuration[2]
    scheduler_configs: list[LRSchedulerConfig] = (
        _configure_schedulers_automatic_opt(schedulers, monitor)
        if automatic_optimization
        else _configure_schedulers_manual_opt(schedulers)
    )
    _validate_multiple_optimizers_support(optimizers, automatic_optimization)
    _validate_optimizers_attached(optimizers, scheduler_configs)
    _validate_scheduler_api(
        scheduler_configs,
        automatic_optimization,
        lr_scheduler_step_overridden
    )
    return optimizers, scheduler_configs


def _configure_optimizers(
    optim_conf: object
) -> tuple[list[torch.optim.Optimizer], list[object], str | None]:
    optimizers: list[torch.optim.Optimizer] = []
    schedulers: list[object] = []
    monitor: str | None = None
    if isinstance(optim_conf, torch.optim.Optimizer):
        optimizers: list[torch.optim.Optimizer] = [optim_conf]
    elif (
        isinstance(optim_conf, (list, tuple))
        and len(optim_conf) == 2
        and isinstance(optim_conf[0], list)
        and all(isinstance(optimizer, torch.optim.Optimizer) for optimizer in optim_conf[0])
    ):
        optimizers: list[torch.optim.Optimizer] = list(optim_conf[0])
        raw_schedulers: object = optim_conf[1]
        schedulers: list[object] = (
            list(raw_schedulers) if isinstance(raw_schedulers, list) else [raw_schedulers]
        )
    elif isinstance(optim_conf, dict):
        normalized: dict[str, object] = dict(optim_conf)
        _validate_optim_conf(normalized)
        optimizer: object = normalized["optimizer"]
        assert isinstance(optimizer, torch.optim.Optimizer)
        optimizers: list[torch.optim.Optimizer] = [optimizer]
        raw_monitor: object | None = normalized.get("monitor")
        if raw_monitor is not None and not isinstance(raw_monitor, str):
            raise MisconfigurationError("The optimizer configuration monitor must be a string or None.")
        monitor: str | None = raw_monitor
        if "lr_scheduler" in normalized:
            schedulers: list[object] = [normalized["lr_scheduler"]]
    elif (
        isinstance(optim_conf, (list, tuple))
        and all(isinstance(configuration, dict) for configuration in optim_conf)
    ):
        for raw_configuration in optim_conf:
            normalized: dict[str, object] = dict(raw_configuration)
            _validate_optim_conf(normalized)
            optimizer: object = normalized["optimizer"]
            assert isinstance(optimizer, torch.optim.Optimizer)
            optimizers.append(optimizer)
            if "lr_scheduler" in normalized:
                raw_scheduler: object = normalized["lr_scheduler"]
                schedulers.append(
                    dict(raw_scheduler)
                    if isinstance(raw_scheduler, dict)
                    else {"scheduler": raw_scheduler}
                )
    elif (
        isinstance(optim_conf, (list, tuple))
        and all(isinstance(optimizer, torch.optim.Optimizer) for optimizer in optim_conf)
    ):
        optimizers: list[torch.optim.Optimizer] = list(optim_conf)
    else:
        raise MisconfigurationError(
            "Unknown configuration for module optimizers. Output from configure_optimizers() should be one of: "
            "Optimizer, [Optimizer], ([Optimizer], [LRScheduler]), or "
            "{'optimizer': Optimizer, 'lr_scheduler': LRScheduler}."
        )
    optimizers: list[torch.optim.Optimizer] = [
        optimizer.optimizer if isinstance(optimizer, SyntheticOptimizer) else optimizer
        for optimizer in optimizers
    ]
    return optimizers, schedulers, monitor


def _configure_schedulers_automatic_opt(
    schedulers: list[object],
    monitor: str | None
) -> list[LRSchedulerConfig]:
    configurations: list[LRSchedulerConfig] = []
    for scheduler in schedulers:
        if isinstance(scheduler, LRSchedulerConfig):
            configuration: LRSchedulerConfig = _validate_scheduler_config(scheduler)
        elif isinstance(scheduler, dict):
            configuration: LRSchedulerConfig = _scheduler_config_from_mapping(dict(scheduler))
        elif isinstance(scheduler, ReduceLROnPlateau):
            if monitor is None:
                raise MisconfigurationError(
                    "configure_optimizers() must include a monitor when a ReduceLROnPlateau scheduler is used."
                )
            configuration: LRSchedulerConfig = LRSchedulerConfig(
                scheduler=scheduler,
                reduce_on_plateau=True,
                monitor=monitor
            )
        elif isinstance(scheduler, _StatefulScheduler):
            configuration: LRSchedulerConfig = LRSchedulerConfig(scheduler=scheduler)
        else:
            raise MisconfigurationError(
                f"The provided lr scheduler {type(scheduler).__name__!r} is invalid."
            )
        configurations.append(configuration)
    return configurations


def _configure_schedulers_manual_opt(schedulers: list[object]) -> list[LRSchedulerConfig]:
    configurations: list[LRSchedulerConfig] = []
    ignored_keys: set[str] = {"reduce_on_plateau", "monitor", "strict"}
    for scheduler in schedulers:
        if isinstance(scheduler, LRSchedulerConfig):
            configuration: LRSchedulerConfig = LRSchedulerConfig(
                scheduler=scheduler.scheduler,
                name=scheduler.name,
                interval=scheduler.interval,
                frequency=scheduler.frequency
            )
        elif isinstance(scheduler, dict):
            normalized: dict[str, object] = dict(scheduler)
            present_ignored_keys: list[str] = [key for key in normalized if key in ignored_keys]
            if present_ignored_keys:
                warnings.warn(
                    f"The lr scheduler configuration contains {present_ignored_keys}, but these keys are ignored "
                    "during manual optimization.",
                    RuntimeWarning,
                    stacklevel=3
                )
            manual_configuration: dict[str, object] = {
                key: value for key, value in normalized.items() if key not in ignored_keys
            }
            manual_configuration["reduce_on_plateau"] = False
            configuration: LRSchedulerConfig = _scheduler_config_from_mapping(manual_configuration)
        elif isinstance(scheduler, _StatefulScheduler):
            configuration: LRSchedulerConfig = LRSchedulerConfig(scheduler=scheduler)
        else:
            raise MisconfigurationError(
                f"The provided lr scheduler {type(scheduler).__name__!r} is invalid."
            )
        configurations.append(configuration)
    return configurations


def _scheduler_config_from_mapping(mapping: dict[str, object]) -> LRSchedulerConfig:
    supported_keys: set[str] = {field.name for field in fields(LRSchedulerConfig)}
    extra_keys: set[str] = set(mapping) - supported_keys
    if extra_keys:
        warnings.warn(
            f"Found unsupported keys in the lr scheduler configuration: {extra_keys}.",
            RuntimeWarning,
            stacklevel=3
        )
        mapping: dict[str, object] = {
            key: value for key, value in mapping.items() if key in supported_keys
        }
    if "scheduler" not in mapping:
        raise MisconfigurationError(
            'The lr scheduler configuration must contain the key "scheduler".'
        )
    scheduler: object = mapping["scheduler"]
    if not isinstance(scheduler, _StatefulScheduler):
        raise TypeError(
            f"The provided lr scheduler {type(scheduler).__name__!r} is invalid."
        )
    interval: object = mapping.get("interval", "epoch")
    frequency: object = mapping.get("frequency", 1)
    reduce_on_plateau: object = mapping.get(
        "reduce_on_plateau", isinstance(scheduler, ReduceLROnPlateau)
    )
    monitor: object | None = mapping.get("monitor")
    strict: object = mapping.get("strict", True)
    name: object | None = mapping.get("name")
    if interval not in ("step", "epoch"):
        raise MisconfigurationError(
            f'The "interval" key in the lr scheduler configuration must be "step" or "epoch", got {interval!r}.'
        )
    if not isinstance(frequency, int) or isinstance(frequency, bool) or frequency < 1:
        raise MisconfigurationError("The lr scheduler frequency must be an integer greater than zero.")
    if not isinstance(reduce_on_plateau, bool):
        raise MisconfigurationError("The lr scheduler reduce_on_plateau value must be a boolean.")
    if monitor is not None and not isinstance(monitor, str):
        raise MisconfigurationError("The lr scheduler monitor must be a string or None.")
    if not isinstance(strict, bool):
        raise MisconfigurationError("The lr scheduler strict value must be a boolean.")
    if name is not None and not isinstance(name, str):
        raise MisconfigurationError("The lr scheduler name must be a string or None.")
    configuration: LRSchedulerConfig = LRSchedulerConfig(
        scheduler=scheduler,
        name=name,
        interval=interval,
        frequency=frequency,
        reduce_on_plateau=reduce_on_plateau,
        monitor=monitor,
        strict=strict
    )
    return _validate_scheduler_config(configuration)


def _validate_scheduler_config(configuration: LRSchedulerConfig) -> LRSchedulerConfig:
    if configuration.interval not in ("step", "epoch"):
        raise MisconfigurationError(
            f'The lr scheduler interval must be "step" or "epoch", got {configuration.interval!r}.'
        )
    if (
        not isinstance(configuration.frequency, int)
        or isinstance(configuration.frequency, bool)
        or configuration.frequency < 1
    ):
        raise MisconfigurationError("The lr scheduler frequency must be an integer greater than zero.")
    if not isinstance(configuration.reduce_on_plateau, bool):
        raise MisconfigurationError("The lr scheduler reduce_on_plateau value must be a boolean.")
    if configuration.monitor is not None and not isinstance(configuration.monitor, str):
        raise MisconfigurationError("The lr scheduler monitor must be a string or None.")
    if not isinstance(configuration.strict, bool):
        raise MisconfigurationError("The lr scheduler strict value must be a boolean.")
    if configuration.name is not None and not isinstance(configuration.name, str):
        raise MisconfigurationError("The lr scheduler name must be a string or None.")
    if configuration.reduce_on_plateau and configuration.monitor is None:
        raise MisconfigurationError(
            "The lr scheduler configuration must include a monitor when ReduceLROnPlateau is used."
        )
    if isinstance(configuration.scheduler, OneCycleLR) and configuration.interval == "epoch":
        warnings.warn(
            "OneCycleLR is configured with interval='epoch'; interval='step' is normally required.",
            RuntimeWarning,
            stacklevel=3
        )
    return configuration


def _validate_multiple_optimizers_support(
    optimizers: list[torch.optim.Optimizer],
    automatic_optimization: bool
) -> None:
    if automatic_optimization and len(optimizers) > 1:
        raise RuntimeError(
            "Training with multiple optimizers is only supported with manual optimization. Set "
            "automatic_optimization = False and access the optimizers through self.optimizers()."
        )


def _validate_optimizers_attached(
    optimizers: list[torch.optim.Optimizer],
    scheduler_configs: list[LRSchedulerConfig]
) -> None:
    for configuration in scheduler_configs:
        if configuration.scheduler.optimizer not in optimizers:
            raise MisconfigurationError(
                "Some schedulers are attached to an optimizer that was not returned from configure_optimizers()."
            )


def _validate_scheduler_api(
    scheduler_configs: list[LRSchedulerConfig],
    automatic_optimization: bool,
    lr_scheduler_step_overridden: bool
) -> None:
    for configuration in scheduler_configs:
        scheduler: _SchedulerInstance = configuration.scheduler
        if not isinstance(scheduler, _StatefulScheduler):
            raise TypeError(
                f"The provided lr scheduler {type(scheduler).__name__!r} is invalid. "
                "It must define state_dict() and load_state_dict()."
            )
        if (
            automatic_optimization
            and not isinstance(scheduler, LRSchedulerTypeTuple)
            and not lr_scheduler_step_overridden
        ):
            raise MisconfigurationError(
                f"The provided lr scheduler {type(scheduler).__name__!r} does not follow PyTorch's "
                "LRScheduler API. Override Module.lr_scheduler_step() to define its stepping behavior."
            )


def _validate_optim_conf(configuration: dict[str, object]) -> None:
    valid_keys: set[str] = {"optimizer", "lr_scheduler", "monitor"}
    extra_keys: set[str] = set(configuration) - valid_keys
    if extra_keys:
        warnings.warn(
            f"Found unsupported keys in the optimizer configuration: {extra_keys}.",
            RuntimeWarning,
            stacklevel=3
        )
    optimizer: object | None = configuration.get("optimizer")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise MisconfigurationError(
            'The optimizer configuration must contain the key "optimizer" with a torch.optim.Optimizer value.'
        )
