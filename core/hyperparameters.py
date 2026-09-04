# This module:
# 1. Defines HyperparametersMixin, the constructor-argument capture and
#    checkpoint reconstruction surface shared by Module and DataModule
# 2. Defines the attribute-accessible mapping used for live and initial
#    hyperparameter views
#
# Design decisions:
# - save_hyperparameters walks the active cooperative __init__ frame chain and
#   captures arguments belonging to the same instance, preserving parameters
#   introduced by several classes in an inheritance graph
# - Clients may capture every constructor argument, selected names, or one
#   explicit Namespace or mapping; explicit ignore names are removed after
#   capture so inherited arguments obey the same exclusion policy
# - The live hparams mapping remains mutable for established client behavior,
#   while hparams_initial returns a deep defensive copy of the post-capture
#   baseline used by loggers and checkpoint construction
# - nn.Module values are warned because model state already persists them, and
#   non-picklable values are removed only at the checkpoint boundary with
#   corrective guidance rather than failing long after capture
# - Frame inspection is contained here so Trainer, logger, and checkpoint code
#   consume explicit mappings instead of acquiring constructor-introspection
#   responsibility
#
# Author: Rahul Sawhney

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

from syntheticmind.primitives.types import HyperparameterDict, HyperparameterValue

__all__: list[str] = ["HyperparametersMixin"]

type _HyperparameterContainer = Namespace | Mapping[str, HyperparameterValue]


class _AttributeDict(dict[str, HyperparameterValue]):
    # Dictionary retaining normal mapping behavior while exposing keys through
    # attribute access for client inspection. Missing keys preserve AttributeError
    # semantics expected by Python tooling and interactive use.
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
    # Constructor-hyperparameter capture shared by model and data contracts.
    # Clients call save_hyperparameters from __init__; the resulting initial
    # values enter checkpoints and, when logger=True, the experiment logger.
    #
    # Example::
    #
    #     from syntheticmind import Module
    #
    #     class Encoder(Module):
    #         def __init__(self, width: int, dropout: float) -> None:
    #             super().__init__()
    #             self.save_hyperparameters()
    #             self._width: int = width
    #             self._dropout: float = dropout
    #
    # The same subclass can later be reconstructed through
    # Encoder.load_from_checkpoint with saved values or explicit overrides.
    _checkpoint_hyperparameters_key: ClassVar[str]
    _checkpoint_hyperparameters_name: ClassVar[str]
    _checkpoint_hyperparameters_type: ClassVar[str]

    def __init__(self) -> None:
        # Starts with empty live and initial mappings and logging disabled until
        # the client explicitly calls save_hyperparameters.
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
        # Captures constructor arguments from the cooperative initialization
        # frames of this instance, or selects explicit names or one supplied
        # mapping. The capture baseline is deep-copied after exclusions.
        #
        # Args:
        #     *args: Zero arguments to capture every constructor value, string
        #         names to select values, or one Namespace or mapping to capture
        #         directly.
        #     ignore: One name or sequence excluded from the final mapping.
        #     frame: Explicit constructor frame for framework-controlled capture;
        #         clients normally leave this as None.
        #     logger: Whether the initial mapping is emitted to configured
        #         experiment loggers. Default: ``True``.
        #
        # Raises:
        #     TypeError: If logger is not a bool.
        #     AttributeError: If no constructor frame is available.
        #     KeyError: If a requested argument name was not captured.
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
        if not args:
            hyperparameters: _HyperparameterContainer = filtered_arguments
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
                hyperparameters: _HyperparameterContainer = cast(
                    _HyperparameterContainer,
                    selected_container
                )
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
                hyperparameters: _HyperparameterContainer = selected_hyperparameters
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
        # Live attribute-accessible mapping. Client mutation affects subsequent
        # checkpoint metadata but does not rewrite hparams_initial.
        return self._hparams

    @property
    def hparams_initial(self) -> MutableMapping[str, HyperparameterValue]:
        # Deep defensive copy of the initial post-capture mapping used for
        # one-time experiment metadata and reproducible reconstruction.
        return cast(
            MutableMapping[str, HyperparameterValue],
            copy.deepcopy(self._hparams_initial)
        )

    def _set_hparams(self, hyperparameters: _HyperparameterContainer) -> None:
        # Normalizes Namespace or mapping input into the attribute-accessible
        # container and merges it into the live mapping.
        resolved_hyperparameters: MutableMapping[str, HyperparameterValue] = (
            self._to_hparams_dict(hyperparameters)
        )
        self._hparams.update(resolved_hyperparameters)

    def _remove_ignored_hparams(self, ignored_names: list[str]) -> None:
        # Removes exclusions after all inherited constructor frames and explicit
        # containers have been merged.
        key: str
        for key in ignored_names:
            self._hparams.pop(key, None)

    def _clean_hparams(self) -> None:
        # Drops values that cannot survive pickle before checkpoint assembly and
        # names each removal so clients can make the exclusion explicit.
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
        # Walks outward through consecutive __init__ frames owned by this
        # instance, then incorporates dataclass initialization fields.
        captured_arguments: HyperparameterDict = {}
        current_frame: types.FrameType | None = frame
        inside_initialization: bool = False
        while current_frame is not None:
            frame_arguments: tuple[
                HyperparametersMixin | None,
                HyperparameterDict
            ] = self._frame_arguments(current_frame)
            local_instance: HyperparametersMixin | None = frame_arguments[0]
            local_arguments: HyperparameterDict = frame_arguments[1]
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
        # Extracts one constructor frame's declared positional, keyword-only, and
        # expanded keyword arguments while excluding interpreter bookkeeping.
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
        # Normalizes exclusions and preserves first occurrence order while
        # removing duplicates.
        if ignore is None:
            return []
        if isinstance(ignore, str):
            return [ignore]
        return list(dict.fromkeys(ignore))

    def _to_hparams_dict(
        self,
        hyperparameters: _HyperparameterContainer
    ) -> MutableMapping[str, HyperparameterValue]:
        # Copies the supplied container so later client mutations of the source
        # cannot alter captured state implicitly.
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
        # Identifies the constructor argument that supplied an explicit container,
        # preferring identity and falling back to safe scalar equality.
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
        # Detects the single empty mapping or Namespace form that intentionally
        # requests no capture.
        if len(args) != 1 or isinstance(args[0], str):
            return False
        value: _HyperparameterContainer = args[0]
        if isinstance(value, Namespace):
            return not vars(value)
        return not value

    def _is_picklable(self, value: HyperparameterValue) -> bool:
        # Probes the same serialization family checkpoints require without
        # letting expected pickle failures escape the cleanup boundary.
        try:
            pickle.dumps(value)
            return True
        except (pickle.PickleError, AttributeError, RuntimeError, TypeError):
            return False
