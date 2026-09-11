# This module:
# 1. Defines Logger, the stateful experiment-record contract for scalar metrics,
#    portable hyperparameters, run-directory identity, and finalization
# 2. Defines the recursive JSON-compatible value boundary shared by concrete
#    logging backends
#
# Design decisions:
# - Logger is a regular inheritance contract with explicit unsupported defaults,
#   not an ABC; concrete backends override three operations while reusing one
#   validated identity and normalization policy
# - Trainer's logger connector owns metric reduction, tensor-to-scalar conversion,
#   rank-zero routing, and fan-out. Logger implementations receive only completed
#   Python scalar observations and never inspect Module metric buffers
# - Run names and versions are single safe path segments, preventing experiment
#   identity from escaping save_dir or changing storage roots
# - Hyperparameters normalize recursively into deterministic JSON-compatible
#   values; live modules, non-scalar tensors, non-scalar arrays, non-finite
#   floats, and unsupported runtime objects fail instead of degrading to repr
# - Sets sort by their canonical JSON representation so repeated runs persist
#   stable metadata independent of hash iteration order
#
# Author: Rahul Sawhney

import json
import math
from argparse import Namespace
from datetime import date, datetime, timedelta
from operator import itemgetter
from pathlib import Path
from typing import cast, override

import numpy as np
import torch
from torch import nn

from syntheticmind.storage.filesystem import StoragePath
from syntheticmind.primitives.types import HyperparameterDict, HyperparameterValue

__all__: list[str] = ["Logger"]

type _JsonValue = (
    bool | int | float | str | None | list[_JsonValue] | dict[str, _JsonValue]
)


class Logger:
    # Stateful destination contract for experiment backends. Client loggers
    # subclass this class, call its constructor, and implement log_metrics,
    # log_hyperparams, and finalize without reimplementing validation.
    def __init__(
        self,
        save_dir: str | Path,
        name: str = "default",
        version: int | str | None = None
    ) -> None:
        # Establishes one safe run identity without touching the destination.
        #
        # Args:
        #     save_dir: Local or supported remote storage root.
        #     name: Optional single directory segment grouping the run.
        #         Default: ``"default"``.
        #     version: Non-negative integer, safe string segment, or None for no
        #         version directory. Integer versions use ``version_N``.
        if not isinstance(save_dir, str | Path):
            raise TypeError("`save_dir` must be a string or Path.")
        self._validate_path_segment(name, "name")
        if isinstance(version, bool) or not isinstance(version, int | str | None):
            raise TypeError("`version` must be an integer, string, or None.")
        if isinstance(version, int) and version < 0:
            raise ValueError(f"Integer logger versions must be non-negative, got {version}.")
        if isinstance(version, str):
            self._validate_path_segment(version, "version")
        self._save_dir: str = StoragePath(save_dir).normalized
        self._name: str = name
        self._version: int | str | None = version

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        # Persists one completed scalar observation at a non-negative training
        # coordinate. Concrete backends must validate through _validated_metrics.
        del metrics, step
        raise NotImplementedError(f"{type(self).__name__} must implement log_metrics().")

    def log_hyperparams(self, params: HyperparameterDict) -> None:
        # Persists one portable configuration mapping. Concrete backends should
        # normalize through _normalize_hyperparameters before serialization.
        del params
        raise NotImplementedError(f"{type(self).__name__} must implement log_hyperparams().")

    def finalize(self) -> None:
        # Flushes and releases backend resources. Trainer attempts this operation
        # during every stage teardown, including failure.
        raise NotImplementedError(f"{type(self).__name__} must implement finalize().")

    @property
    def save_dir(self) -> str:
        # Returns the normalized storage root supplied by the client.
        return self._save_dir

    @property
    def name(self) -> str:
        # Returns the run's optional grouping segment.
        return self._name

    @property
    def version(self) -> int | str | None:
        # Returns the configured version identity before path formatting.
        return self._version

    @property
    def log_dir(self) -> str:
        # Returns the complete backend directory derived from root, name, and
        # version without creating it.
        return self._log_directory()

    def _log_directory(self) -> str:
        # Composes safe identity segments through StoragePath so local and remote
        # implementations share one path convention.
        directory: str = (
            StoragePath(self._save_dir).join(self._name)
            if self._name
            else self._save_dir
        )
        if self._version is None or self._version == "":
            return directory
        version_segment: str = (
            f"version_{self._version}"
            if isinstance(self._version, int)
            else self._version
        )
        return StoragePath(directory).join(version_segment)

    def _validated_metrics(
        self,
        metrics: dict[str, float],
        step: int
    ) -> dict[str, float]:
        # Enforces one non-negative integer coordinate and non-boolean numeric
        # scalar mapping before a backend sees the observation.
        if type(step) is not int:
            raise TypeError(f"Logger step must be an integer, got {step!r}.")
        if step < 0:
            raise ValueError(f"Logger step must be non-negative, got {step}.")
        if not isinstance(metrics, dict):
            raise TypeError("Logger metrics must be a dictionary.")
        validated_metrics: dict[str, float] = {}
        metric_name: str
        metric_value: float
        for metric_name, metric_value in metrics.items():
            if not isinstance(metric_name, str) or not metric_name:
                raise TypeError("Logger metric names must be non-empty strings.")
            if isinstance(metric_value, bool) or not isinstance(metric_value, int | float):
                raise TypeError(
                    f"Logger metric {metric_name!r} must be numeric, got "
                    f"{type(metric_value).__name__}."
                )
            validated_metrics[metric_name]: float = float(metric_value)
        return validated_metrics

    def _normalize_hyperparameters(
        self,
        params: HyperparameterDict
    ) -> dict[str, _JsonValue]:
        # Validates public names and recursively normalizes values to the shared
        # portable metadata representation.
        if not isinstance(params, dict):
            raise TypeError("Logger hyperparameters must be a dictionary.")
        normalized: dict[str, _JsonValue] = {}
        name: str
        value: HyperparameterValue
        for name, value in params.items():
            if not isinstance(name, str) or not name:
                raise TypeError("Hyperparameter names must be non-empty strings.")
            normalized[name]: _JsonValue = self._normalize_hyperparameter(value)
        return normalized

    def _normalize_hyperparameter(self, value: HyperparameterValue) -> _JsonValue:
        # Converts supported scalar, temporal, path, dtype, device, Namespace,
        # array, and container values without retaining live compute objects.
        if value is None or isinstance(value, bool | int | str):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("Hyperparameter floats must be finite for logging.")
            return value
        if isinstance(value, bytes):
            return {"bytes_hex": value.hex()}
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, timedelta):
            return {"duration_seconds": value.total_seconds()}
        if isinstance(value, Path | torch.dtype | torch.device):
            return str(value)
        if isinstance(value, nn.Module):
            raise TypeError(
                f"Live module {type(value).__qualname__} cannot be logged as a hyperparameter."
            )
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise TypeError("Only scalar tensors can be logged as hyperparameters.")
            tensor_value: HyperparameterValue = cast(
                HyperparameterValue,
                value.detach().cpu().item()
            )
            return self._normalize_hyperparameter(tensor_value)
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise TypeError("Only scalar arrays can be logged as hyperparameters.")
            array_value: HyperparameterValue = cast(
                HyperparameterValue,
                value.item()
            )
            return self._normalize_hyperparameter(array_value)
        if isinstance(value, np.generic):
            numpy_value: HyperparameterValue = cast(HyperparameterValue, value.item())
            return self._normalize_hyperparameter(numpy_value)
        if isinstance(value, Namespace):
            namespace_values: HyperparameterDict = cast(HyperparameterDict, vars(value))
            return self._normalize_hyperparameters(namespace_values)
        if isinstance(value, dict):
            return self._normalize_hyperparameters(value)
        if isinstance(value, list | tuple):
            item: HyperparameterValue
            return [self._normalize_hyperparameter(item) for item in value]
        if isinstance(value, set | frozenset):
            ordered_values: list[tuple[str, _JsonValue]] = []
            item: HyperparameterValue
            for item in value:
                normalized_item: _JsonValue = self._normalize_hyperparameter(item)
                ordering_key: str = json.dumps(
                    normalized_item,
                    allow_nan=False,
                    sort_keys=True
                )
                ordered_values.append((ordering_key, normalized_item))
            ordered_values.sort(key=itemgetter(0))
            return [item[1] for item in ordered_values]
        raise TypeError(
            f"Unsupported hyperparameter value type: {type(value).__qualname__}."
        )

    def _validate_path_segment(self, value: str, name: str) -> None:
        # Restricts name and string version values to one traversal-free segment.
        if not isinstance(value, str):
            raise TypeError(f"Logger {name} must be a string.")
        if value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(
                f"Logger {name} must be one path segment without traversal, got {value!r}."
            )

    @override
    def __repr__(self) -> str:
        # Exposes the complete run identity without opening backend resources.
        return (
            f"{type(self).__name__}(save_dir={self._save_dir!r}, "
            f"name={self._name!r}, version={self._version!r})"
        )
