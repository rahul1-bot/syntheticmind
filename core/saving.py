# This module:
# 1. Reconstructs Module subclasses from SyntheticMind schema-v2 checkpoints,
#    saved or external hyperparameters, explicit constructor overrides, and
#    model-owned strictness
# 2. Reconstructs DataModule subclasses through the same payload and
#    hyperparameter boundary while restoring only DataModule-owned state
# 3. Handles local, remote, and seekable checkpoint sources with safe device
#    fallback and corrective incompatible-key reporting
#
# Design decisions:
# - Reconstruction builds the client architecture before applying tensor state:
#   constructor hyperparameters resolve first, configure_model runs next,
#   on_load_checkpoint observes the payload, and load_state_dict mutates last
# - Explicit keyword overrides have highest precedence over hparams files and
#   saved checkpoint values, allowing a client to change constructor policy
#   deliberately without modifying the artifact
# - Constructor arguments are filtered against the subclass signature unless it
#   accepts arbitrary keywords; a saved single-container argument retains its
#   original nesting through the persisted hparams-name field
# - Checkpoint-controlled instantiators are rejected because loading an artifact
#   must not execute arbitrary imported constructors selected by serialized data
# - Missing CUDA, MPS, or XLA destinations fall back to CPU only under the
#   default mapping; an explicit map_location remains fully client-controlled
# - DataModule reconstruction shares schema and hyperparameter parsing but does
#   not invoke model configuration, strict parameter loading, or device movement
# - Every payload and option is validated before model state mutates, keeping a
#   rejected checkpoint from partially changing the target object
#
# Author: Rahul Sawhney

import ast
import csv
import inspect
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import BinaryIO, cast

import torch
import yaml
from torch import nn

from syntheticmind.core.hyperparameters import HyperparametersMixin
from syntheticmind.storage.filesystem import StoragePath
from syntheticmind.primitives.types import (
    CheckpointDict,
    CheckpointValue,
    HyperparameterDict,
    HyperparameterValue,
    MapLocation,
    ModelState,
    StateDict
)

__all__: list[str] = []

type _ConstructorArguments = dict[str, HyperparameterValue]


class _CheckpointLoader:
    # Internal model reconstruction owner used by Module.load_from_checkpoint and
    # Trainer restoration support. Clients use the public classmethod rather than
    # constructing or invoking this class directly.
    @classmethod
    def load[ModelT: nn.Module](
        cls,
        model_class: type[ModelT],
        checkpoint_path: str | Path | BinaryIO,
        map_location: MapLocation = None,
        hparams_file: str | Path | None = None,
        strict: bool | None = None,
        weights_only: bool | None = None,
        constructor_overrides: _ConstructorArguments | None = None
    ) -> ModelT:
        # Executes the complete reconstruction transaction: options and schema,
        # constructor resolution, model hooks, strict state application,
        # incompatible-key reporting, and final device placement.
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
        checkpoint_path: str | Path | BinaryIO,
        map_location: MapLocation,
        weights_only: bool | None
    ) -> CheckpointDict:
        # Loads one path-backed or seekable payload through torch.load and
        # enforces the mapping boundary before downstream schema inspection.
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
    def _constructor_arguments[ComponentT: HyperparametersMixin](
        cls,
        model_class: type[ComponentT],
        checkpoint: CheckpointDict,
        hparams_file: str | Path | None,
        overrides: _ConstructorArguments
    ) -> _ConstructorArguments:
        # Resolves saved, external-file, and explicit hyperparameters against the
        # component constructor while preserving a saved container argument and
        # rejecting artifact-selected instantiation behavior.
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
        # Loads CSV key/value rows with literal scalar parsing or a YAML mapping
        # through StoragePath, rejecting malformed rows and unsupported suffixes.
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
        # Converts an optional mapping-like hyperparameter source into an owned
        # dictionary and rejects non-mapping checkpoint values.
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
        # Restricts constructor keyword arguments to string keys without
        # mutating the wider saved hyperparameter mapping.
        return {
            key: value
            for key, value in hyperparameters.items()
            if isinstance(key, str)
        }

    @classmethod
    def _class_key[ComponentT: HyperparametersMixin](
        cls,
        model_class: type[ComponentT],
        attribute_name: str
    ) -> str:
        # Reads a component's schema field name from its class contract and
        # rejects missing or non-string declarations.
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
        # Applies explicit strictness over the model-owned default while
        # rejecting a direct conflict with a model that set strict_loading.
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
        # Extracts the string-keyed model state required for parameter loading.
        state_dict: CheckpointValue | None = checkpoint.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise KeyError("Checkpoint is missing a model state_dict.")
        if not all(isinstance(key, str) for key in state_dict):
            raise TypeError("Checkpoint model state keys must be strings.")
        return cast(ModelState, state_dict)

    @classmethod
    def _validate_options(
        cls,
        checkpoint_path: str | Path | BinaryIO,
        hparams_file: str | Path | None,
        strict: bool | None,
        weights_only: bool | None
    ) -> None:
        # Validates checkpoint source capability and optional loading controls
        # before filesystem or deserialization work begins.
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
        # Accepts exactly SyntheticMind schema version two and rejects bools that
        # would otherwise satisfy Python's integer relationship.
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
        # Reports missing and unexpected model keys only under non-strict loading,
        # preserving strict load_state_dict failure behavior unchanged.
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
        # Redirects storage from an unavailable saved accelerator to CPU and
        # returns None when torch should retain its ordinary mapping.
        if location.startswith("cuda") and not torch.cuda.is_available():
            return storage.cpu()
        if location.startswith("mps") and not torch.backends.mps.is_available():
            return storage.cpu()
        if location.startswith("xla"):
            return storage.cpu()
        return None

    @classmethod
    def _parse_scalar(cls, value: str) -> HyperparameterValue:
        # Parses CSV values through Python literals while retaining ordinary text
        # verbatim when it is not a literal expression.
        try:
            return cast(HyperparameterValue, ast.literal_eval(value))
        except (SyntaxError, ValueError):
            return value


class _DataModuleCheckpointLoader:
    # DataModule reconstruction owner sharing checkpoint transport and
    # constructor resolution with models while applying only DataModule state.
    @classmethod
    def load[DataModuleT: HyperparametersMixin](
        cls,
        datamodule_class: type[DataModuleT],
        checkpoint_path: str | Path | BinaryIO,
        map_location: MapLocation = None,
        hparams_file: str | Path | None = None,
        weights_only: bool | None = None,
        constructor_overrides: _ConstructorArguments | None = None
    ) -> DataModuleT:
        # Constructs the DataModule from resolved hyperparameters and restores
        # its class-qualified mapping when present. Missing component state is a
        # valid stateless checkpoint rather than an error.
        del cls
        _CheckpointLoader._validate_options(
            checkpoint_path,
            hparams_file,
            None,
            weights_only
        )
        checkpoint: CheckpointDict = _CheckpointLoader._load_payload(
            checkpoint_path,
            map_location,
            weights_only
        )
        _CheckpointLoader._validate_schema(checkpoint)
        overrides: _ConstructorArguments = dict(constructor_overrides or {})
        constructor_arguments: _ConstructorArguments = (
            _CheckpointLoader._constructor_arguments(
                datamodule_class,
                checkpoint,
                hparams_file,
                overrides
            )
        )
        datamodule: DataModuleT = datamodule_class(**constructor_arguments)
        state_value: CheckpointValue | None = checkpoint.get(
            datamodule_class.__qualname__
        )
        if state_value is None:
            return datamodule
        if not isinstance(state_value, Mapping):
            raise TypeError(
                f"Checkpoint state for DataModule {datamodule_class.__qualname__!r} "
                "must be a mapping."
            )
        if not all(isinstance(key, str) for key in state_value):
            raise TypeError("DataModule checkpoint state keys must be strings.")
        load_state: Callable[[StateDict], None] = cast(
            Callable[[StateDict], None],
            getattr(datamodule, "load_state_dict")
        )
        load_state(cast(StateDict, state_value))
        return datamodule
