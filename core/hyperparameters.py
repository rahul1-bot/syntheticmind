import copy
import inspect
import pickle
import types
import warnings
from argparse import Namespace
from collections.abc import MutableMapping, Sequence
from dataclasses import fields, is_dataclass
from typing import ClassVar, cast

from torch import nn

from syntheticmind.utilities.types import HyperparameterDict

__all__: list[str] = ["AttributeDict", "HyperparametersMixin"]


class AttributeDict(dict[object, object]):
    def __getattr__(self, name: str) -> object:
        try:
            return self[name]
        except KeyError as exception:
            raise AttributeError(name) from exception

    def __setattr__(self, name: str, value: object) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exception:
            raise AttributeError(name) from exception


class HyperparametersMixin:
    CHECKPOINT_HYPER_PARAMS_KEY: ClassVar[str]
    CHECKPOINT_HYPER_PARAMS_NAME: ClassVar[str]
    CHECKPOINT_HYPER_PARAMS_TYPE: ClassVar[str]
    __jit_unused_properties__: ClassVar[list[str]] = ["hparams", "hparams_initial"]

    def __init__(self) -> None:
        super().__init__()
        self._log_hyperparams: bool = False
        self._hparams: MutableMapping[object, object] = AttributeDict()
        self._hparams_initial: MutableMapping[object, object] = AttributeDict()
        self._hparams_name: str | None = None

    def save_hyperparameters(
        self,
        *args: object,
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
            if not isinstance(key, str) or key not in ignored_names
        }
        if not args:
            hyperparameters: object = filtered_arguments
            self._hparams_name: str | None = "kwargs" if hyperparameters else None
        else:
            non_string_indices: list[int] = [
                index
                for index, argument in enumerate(args)
                if not isinstance(argument, str)
            ]
            if len(non_string_indices) == 1:
                hyperparameters: object = args[non_string_indices[0]]
                self._hparams_name: str | None = self._argument_name(
                    filtered_arguments,
                    hyperparameters
                )
            else:
                selected_hyperparameters: HyperparameterDict = {}
                argument: object
                for argument in args:
                    if isinstance(argument, str):
                        selected_hyperparameters[argument] = filtered_arguments[argument]
                hyperparameters: object = selected_hyperparameters
                self._hparams_name: str | None = "kwargs"
        self._set_hparams(hyperparameters)
        self.remove_ignored_hparams(ignored_names)
        key: object
        value: object
        for key, value in self._hparams.items():
            if isinstance(value, nn.Module):
                warnings.warn(
                    f"Attribute {key!r} is an nn.Module and is already saved through model state. "
                    f"Pass `ignore={key!r}` to save_hyperparameters to exclude it.",
                    UserWarning,
                    stacklevel=2
                )
        self._hparams_initial: MutableMapping[object, object] = copy.deepcopy(
            self._hparams
        )

    def remove_ignored_hparams(self, ignore_list: list[str]) -> None:
        key: str
        for key in ignore_list:
            self._hparams.pop(key, None)

    @property
    def hparams(self) -> MutableMapping[object, object]:
        return self._hparams

    @property
    def hparams_initial(self) -> MutableMapping[object, object]:
        return copy.deepcopy(self._hparams_initial)

    def _set_hparams(self, hyperparameters: object) -> None:
        resolved_hyperparameters: MutableMapping[object, object] = self._to_hparams_dict(
            hyperparameters
        )
        if isinstance(resolved_hyperparameters, dict) and isinstance(self.hparams, dict):
            self._hparams.update(resolved_hyperparameters)
        else:
            self._hparams: MutableMapping[object, object] = resolved_hyperparameters

    def _clean_hparams(self) -> None:
        removable_keys: list[object] = [
            key
            for key, value in self._hparams.items()
            if not self._is_picklable(value)
        ]
        key: object
        for key in removable_keys:
            warnings.warn(
                f"Hyperparameter {key!r} was removed because it cannot be pickled. "
                f"Pass `ignore={key!r}` to save_hyperparameters to suppress this warning.",
                UserWarning,
                stacklevel=3
            )
            self._hparams.pop(key, None)

    def _captured_arguments(
        self,
        frame: types.FrameType
    ) -> HyperparameterDict:
        captured_arguments: HyperparameterDict = {}
        current_frame: types.FrameType | None = frame
        inside_initialization: bool = False
        while current_frame is not None:
            frame_arguments: tuple[object | None, HyperparameterDict] = (
                self._frame_arguments(current_frame)
            )
            local_instance: object | None = frame_arguments[0]
            local_arguments: HyperparameterDict = frame_arguments[1]
            if local_instance is self:
                captured_arguments.update(local_arguments)
                inside_initialization: bool = True
            elif inside_initialization:
                break
            current_frame: types.FrameType | None = current_frame.f_back
        if is_dataclass(self):
            dataclass_arguments: HyperparameterDict = {
                field.name: getattr(self, field.name)
                for field in fields(self)
                if field.init
            }
            captured_arguments.update(dataclass_arguments)
        return captured_arguments

    def _frame_arguments(
        self,
        frame: types.FrameType
    ) -> tuple[object | None, HyperparameterDict]:
        local_variables: dict[str, object] = dict(frame.f_locals)
        if frame.f_code.co_name != "__init__" or "__class__" not in local_variables:
            return None, {}
        owner: object = local_variables["__class__"]
        if not isinstance(owner, type):
            return None, {}
        parameters: list[inspect.Parameter] = list(
            inspect.signature(owner.__init__).parameters.values()
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
            parameter.name: local_variables[parameter.name]
            for parameter in parameters
            if parameter.name in local_variables
            and parameter.name not in excluded_names
        }
        if keyword_name is not None:
            keyword_arguments: object = local_variables.get(keyword_name)
            if isinstance(keyword_arguments, dict):
                captured_arguments.update(keyword_arguments)
        return local_variables.get(self_name), captured_arguments

    def _ignored_names(
        self,
        ignore: Sequence[str] | str | None
    ) -> list[str]:
        if ignore is None:
            return []
        if isinstance(ignore, str):
            return [ignore]
        return list(dict.fromkeys(name for name in ignore if isinstance(name, str)))

    def _to_hparams_dict(
        self,
        hyperparameters: object
    ) -> MutableMapping[object, object]:
        if isinstance(hyperparameters, Namespace):
            namespace_values: dict[str, object] = dict(vars(hyperparameters))
            resolved_hyperparameters: AttributeDict = AttributeDict()
            namespace_key: str
            namespace_value: object
            for namespace_key, namespace_value in namespace_values.items():
                resolved_hyperparameters[namespace_key] = namespace_value
            return resolved_hyperparameters
        if isinstance(hyperparameters, dict):
            resolved_hyperparameters: AttributeDict = AttributeDict()
            key: object
            value: object
            for key, value in hyperparameters.items():
                resolved_hyperparameters[key] = value
            return resolved_hyperparameters
        if isinstance(hyperparameters, MutableMapping):
            return cast(MutableMapping[object, object], hyperparameters)
        if isinstance(hyperparameters, bool | int | float | str | bytes | None):
            raise ValueError(
                "Primitive hyperparameter containers are unsupported; pass names, "
                "a mapping, or a Namespace."
            )
        raise ValueError(
            f"Unsupported hyperparameter container type {type(hyperparameters).__qualname__}."
        )

    def _argument_name(
        self,
        arguments: HyperparameterDict,
        selected_value: object
    ) -> str | None:
        name: object
        value: object
        for name, value in arguments.items():
            if not isinstance(name, str):
                continue
            if value is selected_value:
                return name
            try:
                equality: object = value == selected_value
            except Exception:
                continue
            if isinstance(equality, bool) and equality:
                return name
        return None

    def _is_empty_container(self, args: tuple[object, ...]) -> bool:
        if len(args) != 1 or isinstance(args[0], str):
            return False
        value: object = args[0]
        if isinstance(value, Namespace):
            return not vars(value)
        if isinstance(value, MutableMapping | Sequence):
            return len(value) == 0
        return False

    def _is_picklable(self, value: object) -> bool:
        try:
            pickle.dumps(value)
            return True
        except (pickle.PickleError, AttributeError, RuntimeError, TypeError):
            return False
