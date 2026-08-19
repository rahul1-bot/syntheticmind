import ast
import csv
import inspect
import warnings
from argparse import Namespace
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import IO, cast

import torch
import yaml
from torch import nn

from syntheticmind.core.hooks import ModelHooks
from syntheticmind.core.hyperparameters import AttributeDict
from syntheticmind.core.saving import CheckpointHooks
from syntheticmind.utilities.filesystem import basename, exists, open_binary, open_text
from syntheticmind.utilities.model_helpers import is_overridden
from syntheticmind.utilities.types import CheckpointDict, CheckpointValue, MapLocation

__all__: list[str] = ["ModelCheckpointLoader"]


class _CheckpointPayloadLoader:
    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path | IO[bytes],
        map_location: MapLocation,
        weights_only: bool | None
    ) -> CheckpointDict:
        resolved_map_location: MapLocation = (
            cls._default_map_location
            if map_location is None
            else map_location
        )
        checkpoint: object
        if isinstance(checkpoint_path, str | Path):
            if not exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            with open_binary(checkpoint_path, "rb") as checkpoint_file:
                checkpoint = torch.load(
                    checkpoint_file,
                    map_location=resolved_map_location,
                    weights_only=weights_only
                )
        else:
            checkpoint_file: IO[bytes] = cast(IO[bytes], checkpoint_path)
            checkpoint = torch.load(
                checkpoint_file,
                map_location=resolved_map_location,
                weights_only=weights_only
            )
        if not isinstance(checkpoint, dict):
            raise TypeError(
                "Checkpoint payload must be a dictionary, got "
                f"{type(checkpoint).__name__}."
            )
        return cast(CheckpointDict, checkpoint)

    @classmethod
    def _default_map_location(
        cls,
        storage: torch.types.Storage,
        location: str
    ) -> torch.types.Storage | None:
        if location.startswith("cuda") and not torch.cuda.is_available():
            return storage.cpu()
        if (
            location.startswith("mps")
            and not torch.backends.mps.is_available()
        ):
            return storage.cpu()
        if location.startswith("xla"):
            return storage.cpu()
        return None


class _CheckpointHyperparameterLoader:
    @classmethod
    def constructor_arguments[ModelT: nn.Module](
        cls,
        model_class: type[ModelT],
        checkpoint: CheckpointDict,
        hparams_file: str | Path | None,
        overrides: dict[str, object]
    ) -> dict[str, object]:
        hyperparameters_key: str = cls._class_key(
            model_class,
            "CHECKPOINT_HYPER_PARAMS_KEY"
        )
        hyperparameters_name_key: str = cls._class_key(
            model_class,
            "CHECKPOINT_HYPER_PARAMS_NAME"
        )
        hyperparameters_type_key: str = cls._class_key(
            model_class,
            "CHECKPOINT_HYPER_PARAMS_TYPE"
        )
        loaded_hyperparameters: dict[object, object] = {}
        legacy_key: str
        for legacy_key in ("hparams", "module_arguments"):
            loaded_hyperparameters.update(
                cls._mapping(checkpoint.get(legacy_key), legacy_key)
            )
        if hparams_file is None:
            loaded_hyperparameters.update(
                cls._mapping(
                    checkpoint.get(hyperparameters_key),
                    hyperparameters_key
                )
            )
        else:
            loaded_hyperparameters.update(cls._load_hparams_file(hparams_file))
        loaded_hyperparameters.update(overrides)
        if "_instantiator" in loaded_hyperparameters:
            raise ValueError(
                "Checkpoint-controlled model instantiators are unsupported because "
                "constructor execution must remain owned by the requested model class."
            )
        checkpoint[hyperparameters_key] = cast(
            CheckpointValue,
            loaded_hyperparameters
        )
        converted_hyperparameters: object = cls._convert_hyperparameters(
            loaded_hyperparameters,
            checkpoint.get(hyperparameters_type_key)
        )
        parameters: list[inspect.Parameter] = list(
            inspect.signature(model_class.__init__).parameters.values()
        )
        accepted_names: set[str] = {
            parameter.name
            for parameter in parameters
            if parameter.kind in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY
            }
        }
        if parameters:
            accepted_names.discard(parameters[0].name)
        accepts_keyword_arguments: bool = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        saved_name: object = checkpoint.get(hyperparameters_name_key)
        constructor_arguments: dict[str, object]
        if isinstance(saved_name, str) and saved_name in accepted_names:
            constructor_arguments = {saved_name: converted_hyperparameters}
        else:
            constructor_arguments = cls._string_mapping(converted_hyperparameters)
        constructor_arguments.update(overrides)
        if not accepts_keyword_arguments:
            constructor_arguments = {
                name: value
                for name, value in constructor_arguments.items()
                if name in accepted_names
            }
        return constructor_arguments

    @classmethod
    def _class_key[ModelT: nn.Module](
        cls,
        model_class: type[ModelT],
        attribute_name: str
    ) -> str:
        value: object = getattr(model_class, attribute_name, None)
        if not isinstance(value, str):
            raise TypeError(
                f"{model_class.__qualname__}.{attribute_name} must be a string."
            )
        return value

    @classmethod
    def _load_hparams_file(
        cls,
        hparams_file: str | Path
    ) -> dict[object, object]:
        if not exists(hparams_file):
            raise FileNotFoundError(f"Hyperparameter file not found: {hparams_file}")
        suffix: str = Path(basename(hparams_file)).suffix.lower()
        if suffix == ".csv":
            with open_text(hparams_file, "r", newline="") as source_file:
                rows: list[list[str]] = list(csv.reader(source_file))
            loaded: dict[object, object] = {}
            row: list[str]
            for row in rows[1:]:
                if len(row) < 2:
                    raise ValueError(
                        f"Malformed hyperparameter CSV row in {hparams_file}: {row!r}."
                    )
                loaded[row[0]] = cls._parse_scalar(row[1])
            return loaded
        if suffix in {".yaml", ".yml"}:
            with open_text(hparams_file, "r") as source_file:
                loaded_yaml: object = yaml.safe_load(source_file)
            if loaded_yaml is None:
                return {}
            return cls._mapping(loaded_yaml, "hparams_file")
        raise ValueError("`hparams_file` must use a .csv, .yaml, or .yml extension.")

    @classmethod
    def _parse_scalar(cls, value: str) -> object:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value

    @classmethod
    def _mapping(
        cls,
        value: object,
        source_name: str
    ) -> dict[object, object]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError(
                f"Checkpoint `{source_name}` must be a mapping, got "
                f"{type(value).__name__}."
            )
        return dict(value.items())

    @classmethod
    def _convert_hyperparameters(
        cls,
        hyperparameters: dict[object, object],
        hyperparameters_type: object
    ) -> object:
        if hyperparameters_type is None:
            return hyperparameters
        if isinstance(hyperparameters_type, str):
            return AttributeDict(hyperparameters)
        if not callable(hyperparameters_type):
            raise TypeError(
                "Checkpoint `hparams_type` must be callable, a legacy string, or None."
            )
        converter: Callable[[dict[object, object]], object] = cast(
            Callable[[dict[object, object]], object],
            hyperparameters_type
        )
        return converter(hyperparameters)

    @classmethod
    def _string_mapping(
        cls,
        hyperparameters: object
    ) -> dict[str, object]:
        if isinstance(hyperparameters, Namespace):
            return dict(vars(hyperparameters))
        if not isinstance(hyperparameters, Mapping):
            raise TypeError(
                "Expanded checkpoint hyperparameters must be a mapping, got "
                f"{type(hyperparameters).__name__}."
            )
        return {
            key: value
            for key, value in hyperparameters.items()
            if isinstance(key, str)
        }


class ModelCheckpointLoader:
    @classmethod
    def load[ModelT: nn.Module](
        cls,
        model_class: type[ModelT],
        checkpoint_path: str | Path | IO[bytes],
        map_location: MapLocation = None,
        hparams_file: str | Path | None = None,
        strict: bool | None = None,
        weights_only: bool | None = None,
        constructor_overrides: dict[str, object] | None = None
    ) -> ModelT:
        cls._validate_options(
            checkpoint_path,
            hparams_file,
            strict,
            weights_only
        )
        checkpoint: CheckpointDict = _CheckpointPayloadLoader.load(
            checkpoint_path,
            map_location,
            weights_only
        )
        overrides: dict[str, object] = dict(constructor_overrides or {})
        constructor_arguments: dict[str, object] = (
            _CheckpointHyperparameterLoader.constructor_arguments(
                model_class,
                checkpoint,
                hparams_file,
                overrides
            )
        )
        model: ModelT = model_class(**constructor_arguments)
        if (
            not isinstance(model, ModelHooks)
            or not isinstance(model, CheckpointHooks)
        ):
            raise TypeError(
                f"{model_class.__qualname__} must implement the standard model "
                "and checkpoint hook contracts."
            )
        resolved_strict: bool = cls._resolve_strictness(model, strict)
        if is_overridden("configure_model", model, ModelHooks):
            model.configure_model()
        model.on_load_checkpoint(checkpoint)
        state_dict: Mapping[str, object] = cls._state_dict(checkpoint)
        incompatible_keys: object = model.load_state_dict(
            state_dict,
            strict=resolved_strict
        )
        if not resolved_strict:
            cls._warn_incompatible_keys(incompatible_keys)
        if not state_dict:
            warnings.warn(
                f"The state dict in {checkpoint_path!r} contains no parameters.",
                UserWarning,
                stacklevel=3
            )
            return model
        device: torch.device = next(
            (
                value.device
                for value in state_dict.values()
                if isinstance(value, torch.Tensor)
            ),
            torch.device("cpu")
        )
        model.to(device)
        return model

    @classmethod
    def _validate_options(
        cls,
        checkpoint_path: str | Path | IO[bytes],
        hparams_file: str | Path | None,
        strict: bool | None,
        weights_only: bool | None
    ) -> None:
        if not isinstance(checkpoint_path, str | Path) and not (
            hasattr(checkpoint_path, "read")
            and hasattr(checkpoint_path, "seek")
        ):
            raise TypeError(
                "`checkpoint_path` must be a string, Path, or seekable binary file."
            )
        if hparams_file is not None and not isinstance(hparams_file, str | Path):
            raise TypeError("`hparams_file` must be a string, Path, or None.")
        if strict is not None and type(strict) is not bool:
            raise TypeError("`strict` must be a bool or None.")
        if weights_only is not None and type(weights_only) is not bool:
            raise TypeError("`weights_only` must be a bool or None.")

    @classmethod
    def _resolve_strictness(
        cls,
        model: nn.Module,
        strict: bool | None
    ) -> bool:
        model_strict_loading: object = getattr(model, "strict_loading", True)
        if type(model_strict_loading) is not bool:
            raise TypeError("Model `strict_loading` must resolve to a bool.")
        explicit_model_strictness: object = getattr(
            model,
            "_strict_loading",
            None
        )
        if (
            explicit_model_strictness is not None
            and strict is not None
            and strict != model_strict_loading
        ):
            raise ValueError(
                f"`strict={strict!r}` conflicts with "
                f"{type(model).__name__}.strict_loading={model_strict_loading!r}."
            )
        return model_strict_loading if strict is None else strict

    @classmethod
    def _state_dict(
        cls,
        checkpoint: CheckpointDict
    ) -> Mapping[str, object]:
        state_dict: object = checkpoint.get(
            "state_dict",
            checkpoint.get("model_state_dict")
        )
        if not isinstance(state_dict, Mapping):
            raise KeyError("Checkpoint is missing a model `state_dict`.")
        if not all(isinstance(key, str) for key in state_dict):
            raise TypeError("Checkpoint model state keys must be strings.")
        return cast(Mapping[str, object], state_dict)

    @classmethod
    def _warn_incompatible_keys(
        cls,
        incompatible_keys: object
    ) -> None:
        missing_keys: object = getattr(incompatible_keys, "missing_keys", [])
        unexpected_keys: object = getattr(incompatible_keys, "unexpected_keys", [])
        if missing_keys:
            warnings.warn(
                "Found keys that are in the model state dict but not in the "
                f"checkpoint: {missing_keys}",
                UserWarning,
                stacklevel=4
            )
        if unexpected_keys:
            warnings.warn(
                "Found keys that are not in the model state dict but in the "
                f"checkpoint: {unexpected_keys}",
                UserWarning,
                stacklevel=4
            )
