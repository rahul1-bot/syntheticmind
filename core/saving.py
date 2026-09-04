import ast
import csv
import inspect
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import IO, cast

import torch
import yaml
from torch import nn

from syntheticmind.storage.filesystem import StoragePath
from syntheticmind.types import (
    CheckpointDict,
    CheckpointValue,
    HyperparameterDict,
    HyperparameterValue,
    MapLocation,
    ModelState
)

__all__: list[str] = []

type _ConstructorArguments = dict[str, HyperparameterValue]


class _CheckpointLoader:
    @classmethod
    def load[ModelT: nn.Module](
        cls,
        model_class: type[ModelT],
        checkpoint_path: str | Path | IO[bytes],
        map_location: MapLocation = None,
        hparams_file: str | Path | None = None,
        strict: bool | None = None,
        weights_only: bool | None = None,
        constructor_overrides: _ConstructorArguments | None = None
    ) -> ModelT:
        cls._validate_options(
            checkpoint_path,
            hparams_file,
            strict,
            weights_only
        )
        checkpoint: CheckpointDict = cls._load_payload(
            checkpoint_path,
            map_location,
            weights_only
        )
        cls._validate_schema(checkpoint)
        overrides: _ConstructorArguments = dict(constructor_overrides or {})
        constructor_arguments: _ConstructorArguments = cls._constructor_arguments(
            model_class,
            checkpoint,
            hparams_file,
            overrides
        )
        model: ModelT = model_class(**constructor_arguments)
        resolved_strict: bool = cls._resolve_strictness(model, strict)
        model.configure_model()
        model.on_load_checkpoint(checkpoint)
        state_dict: ModelState = cls._state_dict(checkpoint)
        incompatible_keys: tuple[list[str], list[str]] = cast(
            tuple[list[str], list[str]],
            model.load_state_dict(state_dict, strict=resolved_strict)
        )
        if not resolved_strict:
            cls._warn_incompatible_keys(
                incompatible_keys[0],
                incompatible_keys[1]
            )
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
    def _load_payload(
        cls,
        checkpoint_path: str | Path | IO[bytes],
        map_location: MapLocation,
        weights_only: bool | None
    ) -> CheckpointDict:
        resolved_map_location: MapLocation = (
            cls._default_map_location if map_location is None else map_location
        )
        if isinstance(checkpoint_path, str | Path):
            if not StoragePath(checkpoint_path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            with StoragePath(checkpoint_path).open_binary("rb") as checkpoint_file:
                checkpoint_value: CheckpointValue = cast(
                    CheckpointValue,
                    torch.load(
                        checkpoint_file,
                        map_location=resolved_map_location,
                        weights_only=weights_only
                    )
                )
        else:
            checkpoint_value: CheckpointValue = cast(
                CheckpointValue,
                torch.load(
                    checkpoint_path,
                    map_location=resolved_map_location,
                    weights_only=weights_only
                )
            )
        if not isinstance(checkpoint_value, dict):
            raise TypeError("Checkpoint payload must be a mapping.")
        return cast(CheckpointDict, checkpoint_value)

    @classmethod
    def _constructor_arguments[ModelT: nn.Module](
        cls,
        model_class: type[ModelT],
        checkpoint: CheckpointDict,
        hparams_file: str | Path | None,
        overrides: _ConstructorArguments
    ) -> _ConstructorArguments:
        hyperparameters_key: str = cls._class_key(
            model_class,
            "_checkpoint_hyperparameters_key"
        )
        hyperparameters_name_key: str = cls._class_key(
            model_class,
            "_checkpoint_hyperparameters_name"
        )
        loaded_hyperparameters: HyperparameterDict = (
            cls._load_hparams_file(hparams_file)
            if hparams_file is not None
            else cls._mapping(
                checkpoint.get(hyperparameters_key),
                hyperparameters_key
            )
        )
        loaded_hyperparameters.update(overrides)
        if "_instantiator" in loaded_hyperparameters:
            raise ValueError("Checkpoint-controlled model instantiators are unsupported.")
        checkpoint[hyperparameters_key]: CheckpointValue = cast(
            CheckpointValue,
            loaded_hyperparameters
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
        saved_name: CheckpointValue | None = checkpoint.get(
            hyperparameters_name_key
        )
        if isinstance(saved_name, str) and saved_name in accepted_names:
            constructor_arguments: _ConstructorArguments = {
                saved_name: loaded_hyperparameters
            }
        else:
            constructor_arguments: _ConstructorArguments = cls._string_mapping(
                loaded_hyperparameters
            )
        constructor_arguments.update(overrides)
        if not accepts_keyword_arguments:
            constructor_arguments: _ConstructorArguments = {
                name: value
                for name, value in constructor_arguments.items()
                if name in accepted_names
            }
        return constructor_arguments

    @classmethod
    def _load_hparams_file(
        cls,
        hparams_file: str | Path
    ) -> HyperparameterDict:
        if not StoragePath(hparams_file).exists():
            raise FileNotFoundError(f"Hyperparameter file not found: {hparams_file}")
        suffix: str = Path(StoragePath(hparams_file).name).suffix.lower()
        if suffix == ".csv":
            with StoragePath(hparams_file).open_text(
                "r",
                newline=""
            ) as source_file:
                rows: list[list[str]] = list(csv.reader(source_file))
            loaded: HyperparameterDict = {}
            row: list[str]
            for row in rows[1:]:
                if len(row) < 2:
                    raise ValueError(
                        f"Malformed hyperparameter CSV row in {hparams_file}: {row!r}."
                    )
                loaded[row[0]]: HyperparameterValue = cls._parse_scalar(row[1])
            return loaded
        if suffix in {".yaml", ".yml"}:
            with StoragePath(hparams_file).open_text("r") as source_file:
                loaded_yaml: HyperparameterValue = cast(
                    HyperparameterValue,
                    yaml.safe_load(source_file)
                )
            if loaded_yaml is None:
                return {}
            return cls._mapping(loaded_yaml, "hparams_file")
        raise ValueError("hparams_file must use a .csv, .yaml, or .yml extension.")

    @classmethod
    def _mapping(
        cls,
        value: HyperparameterValue | CheckpointValue | None,
        source_name: str
    ) -> HyperparameterDict:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError(f"Checkpoint {source_name!r} must be a mapping.")
        return dict(value.items())

    @classmethod
    def _string_mapping(
        cls,
        hyperparameters: HyperparameterDict
    ) -> _ConstructorArguments:
        return {
            key: value
            for key, value in hyperparameters.items()
            if isinstance(key, str)
        }

    @classmethod
    def _class_key[ModelT: nn.Module](
        cls,
        model_class: type[ModelT],
        attribute_name: str
    ) -> str:
        value: HyperparameterValue = getattr(model_class, attribute_name, None)
        if not isinstance(value, str):
            raise TypeError(
                f"{model_class.__qualname__}.{attribute_name} must be a string."
            )
        return value

    @classmethod
    def _resolve_strictness(
        cls,
        model: nn.Module,
        strict: bool | None
    ) -> bool:
        model_strict_loading: HyperparameterValue = getattr(
            model,
            "strict_loading",
            True
        )
        if type(model_strict_loading) is not bool:
            raise TypeError("Model strict_loading must resolve to a bool.")
        explicit_strictness: HyperparameterValue = getattr(
            model,
            "_strict_loading",
            None
        )
        if (
            explicit_strictness is not None
            and strict is not None
            and strict != model_strict_loading
        ):
            raise ValueError(
                f"strict={strict!r} conflicts with "
                f"{type(model).__name__}.strict_loading={model_strict_loading!r}."
            )
        return model_strict_loading if strict is None else strict

    @classmethod
    def _state_dict(cls, checkpoint: CheckpointDict) -> ModelState:
        state_dict: CheckpointValue | None = checkpoint.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise KeyError("Checkpoint is missing a model state_dict.")
        if not all(isinstance(key, str) for key in state_dict):
            raise TypeError("Checkpoint model state keys must be strings.")
        return cast(ModelState, state_dict)

    @classmethod
    def _validate_options(
        cls,
        checkpoint_path: str | Path | IO[bytes],
        hparams_file: str | Path | None,
        strict: bool | None,
        weights_only: bool | None
    ) -> None:
        if not isinstance(checkpoint_path, str | Path) and not (
            hasattr(checkpoint_path, "read") and hasattr(checkpoint_path, "seek")
        ):
            raise TypeError("checkpoint_path must be a path or seekable binary file.")
        if hparams_file is not None and not isinstance(hparams_file, str | Path):
            raise TypeError("hparams_file must be a string, Path, or None.")
        if strict is not None and type(strict) is not bool:
            raise TypeError("strict must be a bool or None.")
        if weights_only is not None and type(weights_only) is not bool:
            raise TypeError("weights_only must be a bool or None.")

    @classmethod
    def _validate_schema(cls, checkpoint: CheckpointDict) -> None:
        schema_version: CheckpointValue | None = checkpoint.get("schema_version")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise TypeError("Checkpoint schema_version must be an integer.")
        if schema_version != 2:
            raise ValueError(
                f"Unsupported SyntheticMind checkpoint schema {schema_version!r}."
            )

    @classmethod
    def _warn_incompatible_keys(
        cls,
        missing_keys: list[str],
        unexpected_keys: list[str]
    ) -> None:
        if missing_keys:
            warnings.warn(
                "Found model keys missing from the checkpoint: "
                f"{missing_keys}",
                UserWarning,
                stacklevel=4
            )
        if unexpected_keys:
            warnings.warn(
                "Found checkpoint keys absent from the model: "
                f"{unexpected_keys}",
                UserWarning,
                stacklevel=4
            )

    @classmethod
    def _default_map_location(
        cls,
        storage: torch.types.Storage,
        location: str
    ) -> torch.types.Storage | None:
        if location.startswith("cuda") and not torch.cuda.is_available():
            return storage.cpu()
        if location.startswith("mps") and not torch.backends.mps.is_available():
            return storage.cpu()
        if location.startswith("xla"):
            return storage.cpu()
        return None

    @classmethod
    def _parse_scalar(cls, value: str) -> HyperparameterValue:
        try:
            return cast(HyperparameterValue, ast.literal_eval(value))
        except (SyntaxError, ValueError):
            return value
