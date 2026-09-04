import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import (
    Annotated,
    ClassVar,
    NotRequired,
    Protocol,
    Required,
    TypedDict,
    cast,
    overload,
    override,
    runtime_checkable
)

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR, ReduceLROnPlateau

from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.types import (
    OptimizerClosure,
    OptimizerParameterGroup,
    OptimizerState,
    OptimizerStateKey,
    OptimizerStateValue,
    OptimizerStepOptions,
    OptimizerStepOutput,
    OptimizerStepValue,
    SchedulerInterval,
    StateDict
)

__all__: list[str] = [
    "LRSchedulerConfig",
    "LRSchedulerConfigType",
    "LRSchedulerType",
    "LRSchedulerTypeUnion",
    "OptimizerConfig",
    "OptimizerLRScheduler",
    "OptimizerLRSchedulerConfig",
    "StatefulScheduler",
    "SyntheticOptimizer"
]

type _ManagedOptimizerStep = Callable[
    [torch.optim.Optimizer, OptimizerClosure | None, OptimizerStepOptions, bool],
    OptimizerStepOutput
]
type _OptimizerToggle = Callable[
    [torch.optim.Optimizer],
    AbstractContextManager[None]
]
type _SchedulerMetric = torch.Tensor | float | None
type LRSchedulerTypeUnion = LRScheduler | ReduceLROnPlateau
type LRSchedulerType = type[LRScheduler] | type[ReduceLROnPlateau]


@runtime_checkable
class StatefulScheduler(Protocol):
    optimizer: torch.optim.Optimizer

    def step(self, metric: _SchedulerMetric = None) -> None: ...

    def state_dict(self) -> StateDict: ...

    def load_state_dict(self, state_dict: StateDict) -> None: ...


type _SchedulerInstance = StatefulScheduler
type _SchedulerConfigurationValue = StatefulScheduler | str | int | bool | None
type _SchedulerConfiguration = dict[str, _SchedulerConfigurationValue]
type _SchedulerInput = StatefulScheduler | LRSchedulerConfig | _SchedulerConfiguration
type _OptimizerConfigurationValue = torch.optim.Optimizer | _SchedulerInput | str
type _OptimizerConfiguration = dict[str, _OptimizerConfigurationValue]
type _OptimizerSequence = Sequence[torch.optim.Optimizer]
type _SchedulerSequence = Sequence[_SchedulerInput]
type _ParsedOptimization = tuple[
    list[torch.optim.Optimizer],
    list[_SchedulerInput],
    str | None
]


class LRSchedulerConfig(BaseModel):
    scheduler: StatefulScheduler
    name: str | None = None
    interval: SchedulerInterval = "epoch"
    frequency: Annotated[int, Field(ge=1)] = 1
    reduce_on_plateau: bool = False
    monitor: str | None = None
    strict: bool = True
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        strict=True,
        validate_default=True
    )

    @model_validator(mode="after")
    def _validate_monitor(self) -> LRSchedulerConfig:
        if self.reduce_on_plateau and self.monitor is None:
            raise ValueError("A plateau scheduler requires a monitor metric.")
        return self


class LRSchedulerConfigType(TypedDict, total=False):
    scheduler: Required[StatefulScheduler]
    name: str | None
    interval: SchedulerInterval
    frequency: int
    reduce_on_plateau: bool
    monitor: str | None
    strict: bool


class OptimizerConfig(TypedDict):
    optimizer: torch.optim.Optimizer


class OptimizerLRSchedulerConfig(TypedDict):
    optimizer: torch.optim.Optimizer
    lr_scheduler: StatefulScheduler | LRSchedulerConfigType
    monitor: NotRequired[str]


type OptimizerLRScheduler = (
    torch.optim.Optimizer
    | Sequence[torch.optim.Optimizer]
    | tuple[_OptimizerSequence, _SchedulerSequence]
    | OptimizerConfig
    | OptimizerLRSchedulerConfig
    | Sequence[OptimizerConfig]
    | Sequence[OptimizerLRSchedulerConfig]
    | None
)


class SyntheticOptimizer:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("SyntheticOptimizer requires a torch optimizer.")
        self._optimizer: torch.optim.Optimizer = optimizer
        self._model: nn.Module | None = None
        self._step_operation: _ManagedOptimizerStep | None = None
        self._toggle_operation: _OptimizerToggle | None = None
        self._on_before_step: Callable[[], None] = self._noop
        self._on_after_step: Callable[[], None] = self._noop
        self._automatic_step: bool = False

    def connect(
        self,
        model: nn.Module,
        step_operation: _ManagedOptimizerStep,
        toggle_operation: _OptimizerToggle
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("SyntheticOptimizer model must be a torch module.")
        if not callable(step_operation) or not callable(toggle_operation):
            raise TypeError("SyntheticOptimizer operations must be callable.")
        self._model: nn.Module | None = model
        self._step_operation: _ManagedOptimizerStep | None = step_operation
        self._toggle_operation: _OptimizerToggle | None = toggle_operation

    def step(
        self,
        closure: OptimizerClosure | None = None,
        **kwargs: OptimizerStepValue
    ) -> OptimizerStepOutput:
        if closure is not None and not callable(closure):
            raise MisconfigurationError("optimizer.step closure must be callable.")
        self._on_before_step()
        try:
            return self._require_step_operation()(
                self._optimizer,
                closure,
                dict(kwargs),
                self._automatic_step
            )
        finally:
            self._on_after_step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group: OptimizerParameterGroup) -> None:
        self._optimizer.add_param_group(param_group)

    def state_dict(self) -> OptimizerState:
        return cast(OptimizerState, self._optimizer.state_dict())

    def load_state_dict(self, state_dict: OptimizerState) -> None:
        self._optimizer.load_state_dict(state_dict)

    def move_state_to(self, device: torch.device) -> None:
        parameter_state: dict[torch.Tensor, dict[OptimizerStateKey, OptimizerStateValue]] = cast(
            dict[torch.Tensor, dict[OptimizerStateKey, OptimizerStateValue]],
            self._optimizer.state
        )
        parameter: torch.Tensor
        state: dict[OptimizerStateKey, OptimizerStateValue]
        for parameter, state in parameter_state.items():
            state_key: OptimizerStateKey
            state_value: OptimizerStateValue
            for state_key, state_value in state.items():
                state[state_key]: OptimizerStateValue = self._move_state_value(
                    state_value,
                    device
                )

    @contextmanager
    def automatic_step_context(self) -> Iterator[None]:
        self._automatic_step: bool = True
        try:
            yield
        finally:
            self._automatic_step: bool = False

    @contextmanager
    def toggle_model(self, sync_grad: bool = True) -> Iterator[None]:
        model: nn.Module = self._require_model()
        synchronization_context: AbstractContextManager[None] = nullcontext()
        if not sync_grad and isinstance(model, DistributedDataParallel):
            synchronization_context: AbstractContextManager[None] = model.no_sync()
        with synchronization_context, self._require_toggle_operation()(self._optimizer):
            yield

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def param_groups(self) -> list[OptimizerParameterGroup]:
        return cast(list[OptimizerParameterGroup], self._optimizer.param_groups)

    def _require_model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("SyntheticOptimizer is not connected to a model.")
        return self._model

    def _require_step_operation(self) -> _ManagedOptimizerStep:
        if self._step_operation is None:
            raise RuntimeError("SyntheticOptimizer is not connected to a step operation.")
        return self._step_operation

    def _require_toggle_operation(self) -> _OptimizerToggle:
        if self._toggle_operation is None:
            raise RuntimeError("SyntheticOptimizer is not connected to a toggle operation.")
        return self._toggle_operation

    def _move_state_value(
        self,
        value: OptimizerStateValue,
        device: torch.device
    ) -> OptimizerStateValue:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            moved_mapping: dict[OptimizerStateKey, OptimizerStateValue] = {}
            state_key: OptimizerStateKey
            state_value: OptimizerStateValue
            for state_key, state_value in value.items():
                moved_mapping[state_key]: OptimizerStateValue = self._move_state_value(
                    state_value,
                    device
                )
            return moved_mapping
        if isinstance(value, list):
            moved_values: list[OptimizerStateValue] = []
            state_value: OptimizerStateValue
            for state_value in value:
                moved_values.append(self._move_state_value(state_value, device))
            return moved_values
        if isinstance(value, tuple):
            moved_values: list[OptimizerStateValue] = []
            state_value: OptimizerStateValue
            for state_value in value:
                moved_values.append(self._move_state_value(state_value, device))
            return tuple(moved_values)
        return value

    def _noop(self) -> None:
        pass


class _NoOpOptimizer(torch.optim.Optimizer):
    @override
    def __init__(self) -> None:
        super().__init__([torch.zeros(1)], {})

    @override
    def add_param_group(self, param_group: OptimizerParameterGroup) -> None:
        del param_group

    @override
    def load_state_dict(self, state_dict: OptimizerState) -> None:
        del state_dict

    @override
    def state_dict(self) -> OptimizerState:
        return {}

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: OptimizerClosure) -> OptimizerStepOutput: ...

    @override
    def step(
        self,
        closure: OptimizerClosure | None = None
    ) -> OptimizerStepOutput:
        if closure is None:
            return None
        return closure()

    @override
    def zero_grad(self, set_to_none: bool = True) -> None:
        del set_to_none

    @override
    def __repr__(self) -> str:
        return "No Optimizer"


class _OptimizerConfigurationParser:
    def __init__(
        self,
        automatic_optimization: bool,
        lr_scheduler_step_overridden: bool
    ) -> None:
        self._automatic_optimization: bool = automatic_optimization
        self._lr_scheduler_step_overridden: bool = lr_scheduler_step_overridden

    def resolve(
        self,
        declaration: OptimizerLRScheduler
    ) -> tuple[list[torch.optim.Optimizer], list[LRSchedulerConfig]]:
        normalized_declaration: OptimizerLRScheduler = declaration
        if normalized_declaration is None:
            warnings.warn(
                "configure_optimizers returned None; fitting will use no optimizer.",
                UserWarning,
                stacklevel=3
            )
            normalized_declaration: OptimizerLRScheduler = _NoOpOptimizer()
        parsed: _ParsedOptimization = self._parse(normalized_declaration)
        optimizers: list[torch.optim.Optimizer] = parsed[0]
        scheduler_configs: list[LRSchedulerConfig] = self._resolve_schedulers(
            parsed[1],
            parsed[2]
        )
        self._validate_multiple_optimizers(optimizers)
        self._validate_scheduler_optimizers(optimizers, scheduler_configs)
        self._validate_scheduler_api(scheduler_configs)
        return optimizers, scheduler_configs

    def _parse(self, declaration: OptimizerLRScheduler) -> _ParsedOptimization:
        if isinstance(declaration, SyntheticOptimizer):
            return [declaration.optimizer], [], None
        if isinstance(declaration, torch.optim.Optimizer):
            return [declaration], [], None
        if isinstance(declaration, Mapping):
            return self._parse_mapping(cast(_OptimizerConfiguration, dict(declaration)))
        if not isinstance(declaration, Sequence) or isinstance(declaration, str | bytes):
            raise self._invalid_declaration()
        values: list[torch.optim.Optimizer | Mapping | Sequence] = list(
            cast(Sequence[torch.optim.Optimizer | Mapping | Sequence], declaration)
        )
        if len(values) == 2 and self._is_optimizer_sequence(values[0]):
            optimizers: list[torch.optim.Optimizer] = self._optimizer_sequence(values[0])
            scheduler_value: torch.optim.Optimizer | Mapping | Sequence = values[1]
            schedulers: list[_SchedulerInput] = self._scheduler_sequence(scheduler_value)
            return optimizers, schedulers, None
        if values and all(isinstance(value, Mapping) for value in values):
            configurations: list[_OptimizerConfiguration] = [
                cast(_OptimizerConfiguration, dict(value))
                for value in values
                if isinstance(value, Mapping)
            ]
            return self._parse_mappings(configurations)
        if self._is_optimizer_sequence(values):
            return self._optimizer_sequence(values), [], None
        raise self._invalid_declaration()

    def _parse_mapping(
        self,
        configuration: _OptimizerConfiguration
    ) -> _ParsedOptimization:
        self._validate_optimizer_mapping(configuration)
        optimizer_value: _OptimizerConfigurationValue = configuration["optimizer"]
        if not isinstance(optimizer_value, torch.optim.Optimizer):
            raise MisconfigurationError("optimizer must be a torch.optim.Optimizer.")
        monitor_value: _OptimizerConfigurationValue | None = configuration.get(
            "monitor"
        )
        if monitor_value is not None and not isinstance(monitor_value, str):
            raise MisconfigurationError("optimizer monitor must be a string or None.")
        scheduler_value: _OptimizerConfigurationValue | None = configuration.get(
            "lr_scheduler"
        )
        schedulers: list[_SchedulerInput] = (
            [cast(_SchedulerInput, scheduler_value)]
            if scheduler_value is not None
            else []
        )
        return [optimizer_value], schedulers, monitor_value

    def _parse_mappings(
        self,
        configurations: list[_OptimizerConfiguration]
    ) -> _ParsedOptimization:
        optimizers: list[torch.optim.Optimizer] = []
        schedulers: list[_SchedulerInput] = []
        configuration: _OptimizerConfiguration
        for configuration in configurations:
            self._validate_optimizer_mapping(configuration)
            optimizer_value: _OptimizerConfigurationValue = configuration["optimizer"]
            if not isinstance(optimizer_value, torch.optim.Optimizer):
                raise MisconfigurationError("optimizer must be a torch.optim.Optimizer.")
            optimizers.append(optimizer_value)
            scheduler_value: _OptimizerConfigurationValue | None = configuration.get(
                "lr_scheduler"
            )
            if scheduler_value is not None:
                schedulers.append(cast(_SchedulerInput, scheduler_value))
        return optimizers, schedulers, None

    def _resolve_schedulers(
        self,
        schedulers: list[_SchedulerInput],
        monitor: str | None
    ) -> list[LRSchedulerConfig]:
        configurations: list[LRSchedulerConfig] = []
        scheduler: _SchedulerInput
        for scheduler in schedulers:
            configuration: LRSchedulerConfig = (
                self._automatic_scheduler(scheduler, monitor)
                if self._automatic_optimization
                else self._manual_scheduler(scheduler)
            )
            configurations.append(configuration)
        return configurations

    def _automatic_scheduler(
        self,
        scheduler: _SchedulerInput,
        monitor: str | None
    ) -> LRSchedulerConfig:
        if isinstance(scheduler, LRSchedulerConfig):
            configuration: LRSchedulerConfig = scheduler
        elif isinstance(scheduler, Mapping):
            configuration: LRSchedulerConfig = self._scheduler_from_mapping(
                cast(_SchedulerConfiguration, dict(scheduler))
            )
        elif isinstance(scheduler, ReduceLROnPlateau):
            if monitor is None:
                raise MisconfigurationError(
                    "ReduceLROnPlateau requires a monitor in configure_optimizers."
                )
            configuration: LRSchedulerConfig = LRSchedulerConfig(
                scheduler=scheduler,
                reduce_on_plateau=True,
                monitor=monitor
            )
        elif isinstance(scheduler, StatefulScheduler):
            configuration: LRSchedulerConfig = LRSchedulerConfig(scheduler=scheduler)
        else:
            raise MisconfigurationError(
                f"Invalid lr scheduler {type(scheduler).__name__!r}."
            )
        return self._validate_scheduler(configuration)

    def _manual_scheduler(self, scheduler: _SchedulerInput) -> LRSchedulerConfig:
        if isinstance(scheduler, LRSchedulerConfig):
            return LRSchedulerConfig(
                scheduler=scheduler.scheduler,
                name=scheduler.name,
                interval=scheduler.interval,
                frequency=scheduler.frequency
            )
        if isinstance(scheduler, Mapping):
            mapping: _SchedulerConfiguration = cast(
                _SchedulerConfiguration,
                dict(scheduler)
            )
            ignored_keys: set[str] = {"reduce_on_plateau", "monitor", "strict"}
            present_ignored_keys: list[str] = [
                key for key in mapping if key in ignored_keys
            ]
            if present_ignored_keys:
                warnings.warn(
                    f"Manual optimization ignores scheduler keys {present_ignored_keys}.",
                    RuntimeWarning,
                    stacklevel=3
                )
            filtered_mapping: _SchedulerConfiguration = {
                key: value
                for key, value in mapping.items()
                if key not in ignored_keys
            }
            filtered_mapping["reduce_on_plateau"]: _SchedulerConfigurationValue = False
            return self._scheduler_from_mapping(filtered_mapping)
        if isinstance(scheduler, StatefulScheduler):
            return LRSchedulerConfig(scheduler=scheduler)
        raise MisconfigurationError(
            f"Invalid lr scheduler {type(scheduler).__name__!r}."
        )

    def _scheduler_from_mapping(
        self,
        mapping: _SchedulerConfiguration
    ) -> LRSchedulerConfig:
        supported_keys: set[str] = set(LRSchedulerConfig.model_fields)
        extra_keys: set[str] = set(mapping) - supported_keys
        if extra_keys:
            warnings.warn(
                f"Unsupported lr scheduler configuration keys: {extra_keys}.",
                RuntimeWarning,
                stacklevel=3
            )
        scheduler_value: _SchedulerConfigurationValue | None = mapping.get("scheduler")
        if not isinstance(scheduler_value, StatefulScheduler):
            raise MisconfigurationError(
                "lr scheduler configuration requires a stateful scheduler."
            )
        interval_value: _SchedulerConfigurationValue | None = mapping.get(
            "interval",
            "epoch"
        )
        frequency_value: _SchedulerConfigurationValue | None = mapping.get(
            "frequency",
            1
        )
        reduce_value: _SchedulerConfigurationValue | None = mapping.get(
            "reduce_on_plateau",
            isinstance(scheduler_value, ReduceLROnPlateau)
        )
        monitor_value: _SchedulerConfigurationValue | None = mapping.get("monitor")
        strict_value: _SchedulerConfigurationValue | None = mapping.get("strict", True)
        name_value: _SchedulerConfigurationValue | None = mapping.get("name")
        if interval_value not in {"step", "epoch"}:
            raise MisconfigurationError("scheduler interval must be step or epoch.")
        if (
            isinstance(frequency_value, bool)
            or not isinstance(frequency_value, int)
            or frequency_value < 1
        ):
            raise MisconfigurationError("scheduler frequency must be a positive integer.")
        if type(reduce_value) is not bool:
            raise MisconfigurationError("reduce_on_plateau must be a boolean.")
        if monitor_value is not None and not isinstance(monitor_value, str):
            raise MisconfigurationError("scheduler monitor must be a string or None.")
        if type(strict_value) is not bool:
            raise MisconfigurationError("scheduler strict must be a boolean.")
        if name_value is not None and not isinstance(name_value, str):
            raise MisconfigurationError("scheduler name must be a string or None.")
        configuration: LRSchedulerConfig = LRSchedulerConfig(
            scheduler=scheduler_value,
            name=name_value,
            interval=cast(SchedulerInterval, interval_value),
            frequency=frequency_value,
            reduce_on_plateau=reduce_value,
            monitor=monitor_value,
            strict=strict_value
        )
        return self._validate_scheduler(configuration)

    def _validate_scheduler(
        self,
        configuration: LRSchedulerConfig
    ) -> LRSchedulerConfig:
        if isinstance(configuration.scheduler, OneCycleLR) and configuration.interval == "epoch":
            warnings.warn(
                "OneCycleLR normally requires interval='step'.",
                RuntimeWarning,
                stacklevel=3
            )
        return configuration

    def _validate_multiple_optimizers(
        self,
        optimizers: list[torch.optim.Optimizer]
    ) -> None:
        if self._automatic_optimization and len(optimizers) > 1:
            raise RuntimeError(
                "Multiple optimizers require manual optimization."
            )

    def _validate_scheduler_optimizers(
        self,
        optimizers: list[torch.optim.Optimizer],
        configurations: list[LRSchedulerConfig]
    ) -> None:
        configuration: LRSchedulerConfig
        for configuration in configurations:
            if configuration.scheduler.optimizer not in optimizers:
                raise MisconfigurationError(
                    "A scheduler is attached to an optimizer not returned by configure_optimizers."
                )

    def _validate_scheduler_api(
        self,
        configurations: list[LRSchedulerConfig]
    ) -> None:
        configuration: LRSchedulerConfig
        for configuration in configurations:
            scheduler: StatefulScheduler = configuration.scheduler
            if (
                self._automatic_optimization
                and not isinstance(scheduler, LRScheduler | ReduceLROnPlateau)
                and not self._lr_scheduler_step_overridden
            ):
                raise MisconfigurationError(
                    f"Scheduler {type(scheduler).__name__!r} requires "
                    "Module.lr_scheduler_step override."
                )

    def _validate_optimizer_mapping(
        self,
        configuration: _OptimizerConfiguration
    ) -> None:
        valid_keys: set[str] = {"optimizer", "lr_scheduler", "monitor"}
        extra_keys: set[str] = set(configuration) - valid_keys
        if extra_keys:
            warnings.warn(
                f"Unsupported optimizer configuration keys: {extra_keys}.",
                RuntimeWarning,
                stacklevel=3
            )
        if not isinstance(configuration.get("optimizer"), torch.optim.Optimizer):
            raise MisconfigurationError(
                "optimizer configuration requires a torch.optim.Optimizer."
            )

    def _is_optimizer_sequence(
        self,
        value: torch.optim.Optimizer | Mapping | Sequence
    ) -> bool:
        return (
            isinstance(value, Sequence)
            and not isinstance(value, str | bytes)
            and all(isinstance(optimizer, torch.optim.Optimizer) for optimizer in value)
        )

    def _optimizer_sequence(
        self,
        value: torch.optim.Optimizer | Mapping | Sequence
    ) -> list[torch.optim.Optimizer]:
        if not self._is_optimizer_sequence(value):
            raise self._invalid_declaration()
        return [
            optimizer.optimizer if isinstance(optimizer, SyntheticOptimizer) else optimizer
            for optimizer in cast(Sequence[torch.optim.Optimizer | SyntheticOptimizer], value)
        ]

    def _scheduler_sequence(
        self,
        value: torch.optim.Optimizer | Mapping | Sequence
    ) -> list[_SchedulerInput]:
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            return [cast(_SchedulerInput, scheduler) for scheduler in value]
        return [cast(_SchedulerInput, value)]

    def _invalid_declaration(self) -> MisconfigurationError:
        return MisconfigurationError(
            "configure_optimizers must return an Optimizer, optimizer sequence, "
            "optimizer and scheduler sequences, or optimizer configuration mapping."
        )
