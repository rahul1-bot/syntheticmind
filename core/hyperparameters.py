import copy
import inspect
import pickle
import types
import warnings
from argparse import Namespace
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import fields, is_dataclass
from typing import ClassVar, cast, override

from torch import nn

from syntheticmind.types import HyperparameterDict, HyperparameterValue

__all__: list[str] = ["HyperparametersMixin"]

type _HyperparameterContainer = Namespace | Mapping[str, HyperparameterValue]


class _AttributeDict(dict[str, HyperparameterValue]):
    def __getattr__(self, name: str) -> HyperparameterValue:
        try:
            return self[name]
        except KeyError as exception:
            raise AttributeError(name) from exception

    @override
    def __setattr__(self, name: str, value: HyperparameterValue) -> None:
        self[name] = value

    @override
    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exception:
            raise AttributeError(name) from exception


class HyperparametersMixin:
    _checkpoint_hyperparameters_key: ClassVar[str]
    _checkpoint_hyperparameters_name: ClassVar[str]
    _checkpoint_hyperparameters_type: ClassVar[str]

    def __init__(self) -> None:
        super().__init__()
        self._log_hyperparams: bool = False
        self._hparams: MutableMapping[str, HyperparameterValue] = _AttributeDict()
        self._hparams_initial: MutableMapping[str, HyperparameterValue] = _AttributeDict()
        self._hparams_name: str | None = None

    def save_hyperparameters(
        self,
        *args: str | _HyperparameterContainer,
        ignore: Sequence[str] | str | None = None,
        frame: types.FrameType | None = None,
        logger: bool = True
    ) -> None:
        if type(logger) is not bool:
            raise TypeError("`logger` must be a bool.")
        self._log_hyperparams: bool = logger
        if self._is_empty_container(args):
            return
        resolved_frame: types.FrameType | None = frame
        if resolved_frame is None:
            current_frame: types.FrameType | None = inspect.currentframe()
            resolved_frame: types.FrameType | None = (
                current_frame.f_back if current_frame is not None else None
            )
        if not isinstance(resolved_frame, types.FrameType):
            raise AttributeError("There is no frame available for hyperparameter capture.")
        captured_arguments: HyperparameterDict = self._captured_arguments(resolved_frame)
        ignored_names: list[str] = self._ignored_names(ignore)
        filtered_arguments: HyperparameterDict = {
            key: value
            for key, value in captured_arguments.items()
            if key not in ignored_names
        }
        hyperparameters: _HyperparameterContainer
        if not args:
            hyperparameters = filtered_arguments
            self._hparams_name: str | None = "kwargs" if hyperparameters else None
        else:
            non_string_indices: list[int] = [
                index
                for index, argument in enumerate(args)
                if not isinstance(argument, str)
            ]
            if len(non_string_indices) == 1:
                selected_container: str | _HyperparameterContainer = args[
                    non_string_indices[0]
                ]
                hyperparameters = cast(_HyperparameterContainer, selected_container)
                self._hparams_name: str | None = self._argument_name(
                    filtered_arguments,
                    hyperparameters
                )
            else:
                selected_hyperparameters: HyperparameterDict = {}
                argument: str | _HyperparameterContainer
                for argument in args:
                    if isinstance(argument, str):
                        selected_hyperparameters[argument] = filtered_arguments[argument]
                hyperparameters = selected_hyperparameters
                self._hparams_name: str | None = "kwargs"
        self._set_hparams(hyperparameters)
        self._remove_ignored_hparams(ignored_names)
        key: str
        value: HyperparameterValue
        for key, value in self._hparams.items():
            if isinstance(value, nn.Module):
                warnings.warn(
                    f"Attribute {key!r} is an nn.Module and is already saved through model state. "
                    f"Pass `ignore={key!r}` to save_hyperparameters to exclude it.",
                    UserWarning,
                    stacklevel=2
                )
        self._hparams_initial: MutableMapping[str, HyperparameterValue] = cast(
            MutableMapping[str, HyperparameterValue],
            copy.deepcopy(self._hparams)
        )

    @property
    def hparams(self) -> MutableMapping[str, HyperparameterValue]:
        return self._hparams

    @property
    def hparams_initial(self) -> MutableMapping[str, HyperparameterValue]:
        return cast(
            MutableMapping[str, HyperparameterValue],
            copy.deepcopy(self._hparams_initial)
        )

    def _set_hparams(self, hyperparameters: _HyperparameterContainer) -> None:
        resolved_hyperparameters: MutableMapping[str, HyperparameterValue] = (
            self._to_hparams_dict(hyperparameters)
        )
        self._hparams.update(resolved_hyperparameters)

    def _remove_ignored_hparams(self, ignored_names: list[str]) -> None:
        key: str
        for key in ignored_names:
            self._hparams.pop(key, None)

    def _clean_hparams(self) -> None:
        removable_keys: list[str] = [
            key
            for key, value in self._hparams.items()
            if not self._is_picklable(value)
        ]
        key: str
        for key in removable_keys:
            warnings.warn(
                f"Hyperparameter {key!r} was removed because it cannot be pickled. "
                f"Pass `ignore={key!r}` to save_hyperparameters to suppress this warning.",
                UserWarning,
                stacklevel=3
            )
            self._hparams.pop(key, None)

    def _captured_arguments(self, frame: types.FrameType) -> HyperparameterDict:
        captured_arguments: HyperparameterDict = {}
        current_frame: types.FrameType | None = frame
        inside_initialization: bool = False
        while current_frame is not None:
            local_instance: HyperparametersMixin | None
            local_arguments: HyperparameterDict
            local_instance, local_arguments = self._frame_arguments(current_frame)
            if local_instance is self:
                captured_arguments.update(local_arguments)
                inside_initialization: bool = True
            elif inside_initialization:
                break
            current_frame: types.FrameType | None = current_frame.f_back
        if is_dataclass(self):
            dataclass_arguments: HyperparameterDict = {
                field.name: cast(HyperparameterValue, getattr(self, field.name))
                for field in fields(self)
                if field.init
            }
            captured_arguments.update(dataclass_arguments)
        return captured_arguments

    def _frame_arguments(
        self,
        frame: types.FrameType
    ) -> tuple[HyperparametersMixin | None, HyperparameterDict]:
        local_variables: dict[str, HyperparameterValue | HyperparametersMixin | type] = cast(
            dict[str, HyperparameterValue | HyperparametersMixin | type],
            dict(frame.f_locals)
        )
        if frame.f_code.co_name != "__init__" or "__class__" not in local_variables:
            return None, {}
        owner_candidate: HyperparameterValue | HyperparametersMixin | type = local_variables[
            "__class__"
        ]
        if not isinstance(owner_candidate, type):
            return None, {}
        parameters: list[inspect.Parameter] = list(
            inspect.signature(owner_candidate.__init__).parameters.values()
        )
        if not parameters:
            return None, {}
        self_name: str = parameters[0].name
        positional_name: str | None = next(
            (
                parameter.name
                for parameter in parameters
                if parameter.kind is inspect.Parameter.VAR_POSITIONAL
            ),
            None
        )
        keyword_name: str | None = next(
            (
                parameter.name
                for parameter in parameters
                if parameter.kind is inspect.Parameter.VAR_KEYWORD
            ),
            None
        )
        excluded_names: set[str] = {
            self_name,
            "__class__",
            "frame",
            "frame_args"
        }
        if positional_name is not None:
            excluded_names.add(positional_name)
        if keyword_name is not None:
            excluded_names.add(keyword_name)
        captured_arguments: HyperparameterDict = {
            parameter.name: cast(HyperparameterValue, local_variables[parameter.name])
            for parameter in parameters
            if parameter.name in local_variables
            and parameter.name not in excluded_names
        }
        if keyword_name is not None:
            keyword_arguments: HyperparameterValue | HyperparametersMixin | type | None = (
                local_variables.get(keyword_name)
            )
            if isinstance(keyword_arguments, Mapping):
                captured_arguments.update(cast(HyperparameterDict, keyword_arguments))
        instance_candidate: HyperparameterValue | HyperparametersMixin | type | None = (
            local_variables.get(self_name)
        )
        local_instance: HyperparametersMixin | None = (
            instance_candidate
            if isinstance(instance_candidate, HyperparametersMixin)
            else None
        )
        return local_instance, captured_arguments

    def _ignored_names(self, ignore: Sequence[str] | str | None) -> list[str]:
        if ignore is None:
            return []
        if isinstance(ignore, str):
            return [ignore]
        return list(dict.fromkeys(ignore))

    def _to_hparams_dict(
        self,
        hyperparameters: _HyperparameterContainer
    ) -> MutableMapping[str, HyperparameterValue]:
        if isinstance(hyperparameters, Namespace):
            namespace_values: HyperparameterDict = cast(
                HyperparameterDict,
                dict(vars(hyperparameters))
            )
            return _AttributeDict(namespace_values)
        return _AttributeDict(dict(hyperparameters))

    def _argument_name(
        self,
        arguments: HyperparameterDict,
        selected_value: _HyperparameterContainer
    ) -> str | None:
        name: str
        value: HyperparameterValue
        for name, value in arguments.items():
            if value is selected_value:
                return name
            try:
                equality: HyperparameterValue = cast(
                    HyperparameterValue,
                    value == selected_value
                )
            except Exception:
                continue
            if isinstance(equality, bool) and equality:
                return name
        return None

    def _is_empty_container(
        self,
        args: tuple[str | _HyperparameterContainer, ...]
    ) -> bool:
        if len(args) != 1 or isinstance(args[0], str):
            return False
        value: _HyperparameterContainer = args[0]
        if isinstance(value, Namespace):
            return not vars(value)
        return not value

    def _is_picklable(self, value: HyperparameterValue) -> bool:
        try:
            pickle.dumps(value)
            return True
        except (pickle.PickleError, AttributeError, RuntimeError, TypeError):
            return False
