# This module:
# 1. Resolves one or several experiment loggers and owns callback, logged, and
#    stage-specific progress metric views
# 2. Routes scalar metric emission under global-rank ownership and merges Module
#    and DataModule hyperparameters exactly once per run
#
# Design decisions:
# - Metric production belongs to Module, reduction belongs to loops and
#   MetricAccumulator, and persistence belongs to Logger; this connector owns only
#   destination views, rank policy, and emission
# - Callback metrics retain tensor dtype and device, logged metrics retain tensors
#   for inspection, progress metrics convert to Python floats, and concrete
#   loggers receive detached Python scalars
# - Sanity validation may preview callback observations but cannot contaminate
#   durable logged or progress views
# - Logger True or None selects a versioned default backend, preferring
#   TensorBoard when installed and otherwise CSV; fast_dev_run disables all
#   durable logger instances
# - Model and DataModule hyperparameters merge only when shared names have equal
#   values and emit once across sequential or multiprocessing stages
# - log_dir broadcasts the rank-zero destination so every process observes one
#   experiment identity without duplicate writes
#
# Author: Rahul Sawhney

from collections.abc import Iterable, Mapping
from importlib.util import find_spec
from pathlib import Path

import torch

from syntheticmind.data.datamodule import DataModule
from syntheticmind.core.metric import LoggedMetric
from syntheticmind.core.module import Module
from syntheticmind.loggers.csv_logger import CSVLogger
from syntheticmind.loggers.logger import Logger
from syntheticmind.loggers.tensorboard import TensorBoardLogger
from syntheticmind.strategies.strategy import Strategy
from syntheticmind.storage.filesystem import StoragePath
from syntheticmind.primitives.types import (
    HyperparameterDict,
    HyperparameterValue,
    RunningStage
)

__all__: list[str] = []


class _LoggerConnector:
    # Private experiment-observation and logger-composition owner attached to one
    # Strategy and Trainer root directory.
    def __init__(
        self,
        strategy: Strategy,
        logger: bool | Logger | Iterable[Logger] | None,
        default_root_dir: str | Path,
        fast_dev_run: bool,
        enable_autolog_hparams: bool
    ) -> None:
        # Resolves logger instances and initializes independent metric views plus
        # one-time hyperparameter state.
        self._strategy: Strategy = strategy
        self._default_root_dir: str | Path = default_root_dir
        self._fast_dev_run: bool = fast_dev_run
        self._enable_autolog_hparams: bool = enable_autolog_hparams
        self._loggers: list[Logger] = self._resolve_loggers(logger)
        self._hyperparameters_logged: bool = False
        self.callback_metrics: dict[str, torch.Tensor] = {}
        self.logged_metrics: dict[str, torch.Tensor] = {}
        self.progress_bar_metrics_by_stage: dict[
            RunningStage,
            dict[str, float]
        ] = self._empty_progress_metrics()
        self.last_progress_bar_stage: RunningStage | None = None

    def reset(self, running_stage: RunningStage | None = None) -> None:
        # Clears every view for a new run or only one progress stage for a reused
        # loop, retaining other stage results for inspection.
        if running_stage is None:
            self.callback_metrics.clear()
            self.logged_metrics.clear()
            self.progress_bar_metrics_by_stage: dict[
                RunningStage,
                dict[str, float]
            ] = self._empty_progress_metrics()
            self.last_progress_bar_stage: RunningStage | None = None
            return
        self.progress_bar_metrics_by_stage[running_stage]: dict[str, float] = {}
        self.last_progress_bar_stage: RunningStage | None = running_stage

    def clear_metric_views(self) -> None:
        # Clears callback and logged tensors without changing progress history.
        self.callback_metrics.clear()
        self.logged_metrics.clear()

    def update_callback_metrics(self, metrics: Mapping[str, torch.Tensor]) -> None:
        self.callback_metrics.update(metrics)

    def update_logged_metrics(self, metrics: Mapping[str, torch.Tensor]) -> None:
        self.logged_metrics.update(metrics)

    def update_progress_bar_metrics(
        self,
        running_stage: RunningStage,
        metrics: Mapping[str, float]
    ) -> None:
        if not metrics:
            return
        self.progress_bar_metrics_by_stage[running_stage].update(metrics)
        self.last_progress_bar_stage: RunningStage | None = running_stage

    def preview_logged_metrics(
        self,
        module: Module,
        running_stage: RunningStage,
        sanity_checking: bool
    ) -> None:
        # Computes pending step observations before batch-end hooks so progress
        # callbacks can inspect live values; sanity execution remains non-durable.
        callback_metrics: dict[str, torch.Tensor] = {}
        logged_metrics: dict[str, torch.Tensor] = {}
        progress_bar_metrics: dict[str, float] = {}
        metric: LoggedMetric
        for metric in module.logged_metrics:
            if not metric.on_step:
                continue
            value: torch.Tensor | None = metric.compute_step(
                self._strategy.root_device,
                self._strategy.reduce
            )
            if value is None:
                continue
            metric_name: str = metric.output_name(on_step=True)
            if running_stage == "training":
                callback_metrics[metric.base_name()]: torch.Tensor = value
                callback_metrics[metric_name]: torch.Tensor = value
            if metric.logger:
                logged_metrics[metric_name]: torch.Tensor = value
            if metric.progress_bar:
                progress_bar_metrics[metric_name]: float = self.metric_to_float(value)
        self.update_callback_metrics(callback_metrics)
        if not sanity_checking:
            self.update_logged_metrics(logged_metrics)
            self.update_progress_bar_metrics(running_stage, progress_bar_metrics)

    def progress_bar_metrics(self, running_stage: RunningStage) -> dict[str, float]:
        # Defensive stage-specific progress mapping.
        return dict(self.progress_bar_metrics_by_stage[running_stage])

    def log_metrics(
        self,
        metrics: Mapping[str, torch.Tensor | float],
        step: int
    ) -> None:
        # Converts scalar tensors and emits one copied mapping to every logger on
        # global rank zero only.
        if not self._strategy.is_global_zero:
            return
        scalar_metrics: dict[str, float] = {
            name: self.metric_to_float(value)
            for name, value in metrics.items()
        }
        configured_logger: Logger
        for configured_logger in self._loggers:
            configured_logger.log_metrics(dict(scalar_metrics), step=step)

    def log_hyperparams(self, params: HyperparameterDict) -> None:
        # Emits copied hyperparameters on global rank zero and marks completion on
        # every rank so later stages do not repeat metadata.
        if not self._loggers:
            return
        if self._strategy.is_global_zero:
            configured_logger: Logger
            for configured_logger in self._loggers:
                configured_logger.log_hyperparams(dict(params))
        self._hyperparameters_logged: bool = True

    def autolog_hyperparameters(
        self,
        module: Module,
        datamodule: DataModule | None
    ) -> None:
        # Merges enabled Module and DataModule initial mappings, rejects conflicting
        # shared values, removes internal class-path metadata, and emits once.
        if (
            not self._enable_autolog_hparams
            or not self._loggers
            or self._fast_dev_run
            or self._hyperparameters_logged
        ):
            return
        module_hyperparameters: HyperparameterDict | None = (
            dict(module.hparams_initial) if module._log_hyperparams else None
        )
        datamodule_hyperparameters: HyperparameterDict | None = (
            dict(datamodule.hparams_initial)
            if datamodule is not None and datamodule._log_hyperparams
            else None
        )
        merged_hyperparameters: HyperparameterDict | None = None
        if (
            module_hyperparameters is not None
            and datamodule_hyperparameters is not None
        ):
            inconsistent_keys: list[HyperparameterValue] = []
            key: HyperparameterValue
            for key in module_hyperparameters.keys() & datamodule_hyperparameters.keys():
                if key == "_class_path":
                    continue
                if not self._hyperparameters_equal(
                    module_hyperparameters[key],
                    datamodule_hyperparameters[key]
                ):
                    inconsistent_keys.append(key)
            if inconsistent_keys:
                raise RuntimeError(
                    "Cannot merge model and datamodule hyperparameters because "
                    f"the shared keys {inconsistent_keys!r} have different values."
                )
            merged_hyperparameters: HyperparameterDict | None = {
                **module_hyperparameters,
                **datamodule_hyperparameters
            }
        elif module_hyperparameters is not None:
            merged_hyperparameters: HyperparameterDict | None = module_hyperparameters
        elif datamodule_hyperparameters is not None:
            merged_hyperparameters: HyperparameterDict | None = datamodule_hyperparameters
        if merged_hyperparameters is None:
            return
        filtered_hyperparameters: HyperparameterDict = {
            key: value
            for key, value in merged_hyperparameters.items()
            if key != "_class_path"
        }
        self.log_hyperparams(filtered_hyperparameters)

    def metric_to_float(self, value: torch.Tensor | float) -> float:
        # Converts one scalar tensor or numeric value at the logger or presentation
        # boundary and rejects non-scalar tensors.
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Logged metric tensors must contain exactly one element.")
            return float(value.detach().cpu().item())
        return float(value)

    @property
    def logger(self) -> Logger | None:
        return self._loggers[0] if self._loggers else None

    @logger.setter
    def logger(self, logger: Logger | None) -> None:
        if logger is not None and not isinstance(logger, Logger):
            raise TypeError("logger must be a Logger or None.")
        self._loggers: list[Logger] = [] if logger is None else [logger]
        self._hyperparameters_logged: bool = False

    @property
    def loggers(self) -> list[Logger]:
        return list(self._loggers)

    @loggers.setter
    def loggers(self, loggers: list[Logger] | None) -> None:
        resolved_loggers: list[Logger] = [] if loggers is None else list(loggers)
        if not all(isinstance(logger, Logger) for logger in resolved_loggers):
            raise TypeError("loggers must contain only Logger instances.")
        if len({id(logger) for logger in resolved_loggers}) != len(resolved_loggers):
            raise ValueError("The same Logger instance cannot be configured more than once.")
        self._loggers: list[Logger] = resolved_loggers
        self._hyperparameters_logged: bool = False

    @property
    def hyperparameters_logged(self) -> bool:
        return self._hyperparameters_logged

    @hyperparameters_logged.setter
    def hyperparameters_logged(self, value: bool) -> None:
        if type(value) is not bool:
            raise TypeError("hyperparameters_logged must be a bool.")
        self._hyperparameters_logged: bool = value

    @property
    def log_dir(self) -> str:
        # Primary logger directory or Trainer root, broadcast from rank zero.
        logger: Logger | None = self.logger
        directory: str = (
            logger.log_dir
            if logger is not None
            else str(self._default_root_dir)
        )
        return self._strategy.broadcast(directory)

    def _resolve_loggers(
        self,
        logger: bool | Logger | Iterable[Logger] | None
    ) -> list[Logger]:
        # Resolves default, disabled, single, or iterable logger configuration,
        # rejects duplicate instances, and applies fast-development suppression.
        if logger is None or logger is True:
            if self._fast_dev_run:
                return []
            return [self._default_logger()]
        if logger is False:
            return []
        if isinstance(logger, Logger):
            return [] if self._fast_dev_run else [logger]
        if isinstance(logger, str | bytes) or not isinstance(logger, Iterable):
            raise TypeError(
                "logger must be a Logger, an iterable of Logger instances, "
                "True, False, or None."
            )
        loggers: list[Logger] = list(logger)
        index: int
        configured_logger: Logger
        for index, configured_logger in enumerate(loggers):
            if not isinstance(configured_logger, Logger):
                raise TypeError(
                    f"logger item {index} must be a Logger, got "
                    f"{type(configured_logger).__name__}."
                )
        if len({id(configured_logger) for configured_logger in loggers}) != len(loggers):
            raise ValueError("The same Logger instance cannot be configured more than once.")
        if self._fast_dev_run:
            return []
        return loggers

    def _default_logger(self) -> Logger:
        # Creates the next versioned TensorBoard logger when available or the CSV
        # fallback under syntheticmind_logs.
        name: str = "syntheticmind_logs"
        version: int = self._next_version(
            StoragePath(self._default_root_dir).join(name)
        )
        if find_spec("tensorboard") is not None:
            return TensorBoardLogger(
                save_dir=self._default_root_dir,
                name=name,
                version=version
            )
        return CSVLogger(
            save_dir=self._default_root_dir,
            name=name,
            version=version
        )

    def _next_version(self, root_dir: str | Path) -> int:
        # Scans direct version_N directories and selects one greater than the
        # highest valid integer suffix.
        versions: list[int] = []
        path: str
        for path in StoragePath(root_dir).list_directory():
            storage_path: StoragePath = StoragePath(path)
            name: str = storage_path.name
            if not storage_path.is_directory() or not name.startswith("version_"):
                continue
            suffix: str = name.removeprefix("version_")
            if suffix.isdigit():
                versions.append(int(suffix))
        return max(versions, default=-1) + 1

    def _hyperparameters_equal(
        self,
        first: HyperparameterValue,
        second: HyperparameterValue
    ) -> bool:
        # Compares exact types, preserves tensor identity semantics, and contains
        # user-defined equality failures at the merge boundary.
        if type(first) is not type(second):
            return False
        if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
            return first is second
        if first is second:
            return True
        try:
            return bool(first == second)
        except (RuntimeError, TypeError, ValueError):
            return False

    def _empty_progress_metrics(
        self
    ) -> dict[RunningStage, dict[str, float]]:
        # Creates independent mutable mappings for every running stage.
        return {
            "training": {},
            "validating": {},
            "testing": {},
            "predicting": {},
            "sanity_checking": {}
        }
