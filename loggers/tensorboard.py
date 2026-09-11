# This module:
# 1. Defines TensorBoardLogger, the local experiment backend for scalar event
#    series and hyperparameter records
# 2. Defines the narrow structural writer contract used to isolate the optional
#    TensorBoard dependency from the logger lifecycle
#
# Design decisions:
# - TensorBoard storage is local-only because SummaryWriter does not implement
#   the framework's fsspec transport contract
# - SummaryWriter imports and construction are lazy, so importing SyntheticMind
#   or constructing an unused logger does not require the optional dependency or
#   create filesystem state
# - The private Protocol describes only operations this adapter invokes; it does
#   not duplicate the third-party writer class or become a public abstraction
# - Nested hyperparameter values serialize deterministically to JSON strings
#   because TensorBoard accepts only scalar fields
# - finalize clears the cached writer before flush and close, making repeated
#   teardown and later logger reuse deterministic
#
# Author: Rahul Sawhney

import json
from pathlib import Path
from typing import Protocol, override

from syntheticmind.loggers.logger import Logger, _JsonValue
from syntheticmind.storage.filesystem import StoragePath
from syntheticmind.primitives.types import HyperparameterDict

__all__: list[str] = ["TensorBoardLogger"]

type _TensorBoardHyperparameter = bool | int | float | str


class _TensorBoardWriter(Protocol):
    # Minimal structural dependency implemented by torch SummaryWriter and by
    # focused test doubles without importing the optional backend eagerly.
    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None
    ) -> None:
        # Records one scalar at an optional global training coordinate.
        ...

    def add_hparams(
        self,
        hparam_dict: dict[str, _TensorBoardHyperparameter],
        metric_dict: dict[str, float]
    ) -> None:
        # Records a scalar-only hyperparameter mapping.
        ...

    def flush(self) -> None:
        # Forces buffered event data to its destination.
        ...

    def close(self) -> None:
        # Releases writer resources.
        ...


class TensorBoardLogger(Logger):
    # Optional local event logger selected explicitly by client code.
    @override
    def __init__(
        self,
        save_dir: str | Path,
        name: str = "default",
        version: int | str | None = None
    ) -> None:
        # Validates local destination semantics while deferring writer creation.
        super().__init__(save_dir=save_dir, name=name, version=version)
        if not StoragePath(self.log_dir).is_local:
            raise ValueError("TensorBoardLogger requires a local filesystem directory.")
        self._writer: _TensorBoardWriter | None = None

    @override
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        # Validates and emits every scalar under the shared training coordinate.
        validated_metrics: dict[str, float] = self._validated_metrics(metrics, step)
        writer: _TensorBoardWriter = self._get_writer()
        key: str
        value: float
        for key, value in validated_metrics.items():
            writer.add_scalar(key, value, global_step=step)

    @override
    def log_hyperparams(self, params: HyperparameterDict) -> None:
        # Normalizes portable values and encodes nested structures into stable
        # JSON strings accepted by TensorBoard's scalar hyperparameter boundary.
        normalized_params: dict[str, _JsonValue] = self._normalize_hyperparameters(
            params
        )
        tensorboard_params: dict[str, _TensorBoardHyperparameter] = {}
        name: str
        value: _JsonValue
        for name, value in normalized_params.items():
            if isinstance(value, bool | int | float | str):
                tensorboard_params[name]: _TensorBoardHyperparameter = value
            elif value is None:
                tensorboard_params[name]: _TensorBoardHyperparameter = "null"
            else:
                tensorboard_params[name]: _TensorBoardHyperparameter = json.dumps(
                    value,
                    allow_nan=False,
                    ensure_ascii=False,
                    sort_keys=True
                )
        self._get_writer().add_hparams(tensorboard_params, {})

    @override
    def finalize(self) -> None:
        # Flushes and closes an acquired writer exactly once.
        writer: _TensorBoardWriter | None = self._writer
        self._writer: _TensorBoardWriter | None = None
        if writer is None:
            return
        try:
            writer.flush()
        finally:
            writer.close()

    def _get_writer(self) -> _TensorBoardWriter:
        # Imports and constructs SummaryWriter only on the first actual emission.
        if self._writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self._writer: _TensorBoardWriter | None = SummaryWriter(
                log_dir=self.log_dir
            )
        return self._writer
