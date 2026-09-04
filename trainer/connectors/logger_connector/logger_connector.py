from collections.abc import Iterable, Mapping
from importlib.util import find_spec
from pathlib import Path

import torch

from syntheticmind.core.datamodule import DataModule
from syntheticmind.core.module import LoggedMetric, Module
from syntheticmind.loggers.csv_logger import CSVLogger
from syntheticmind.loggers.logger import Logger, _NoOpLogger
from syntheticmind.loggers.tensorboard import TensorBoardLogger
from syntheticmind.strategies.strategy import Strategy
from syntheticmind.storage.filesystem import (
    basename,
    is_directory,
    join_path,
    list_directory
)
from syntheticmind.types import (
    HyperparameterDict,
    HyperparameterValue,
    RunningStage
)

__all__: list[str] = []


class _LoggerConnector:
    def __init__(
        self,
        strategy: Strategy,
        logger: bool | Logger | Iterable[Logger] | None,
        default_root_dir: str | Path,
        fast_dev_run: bool,
        enable_autolog_hparams: bool
    ) -> None:
        self._strategy: Strategy = strategy
        self._default_root_dir: str | Path = default_root_dir
        self._fast_dev_run: bool = fast_dev_run
        self._enable_autolog_hparams: bool = enable_autolog_hparams
        self._loggers: list[Logger] = self._resolve_loggers(logger)
        self.callback_metrics: dict[str, torch.Tensor] = {}
        self.logged_metrics: dict[str, torch.Tensor] = {}
        self.progress_bar_metrics_by_stage: dict[
            RunningStage,
            dict[str, float]
        ] = self._empty_progress_metrics()
        self.last_progress_bar_stage: RunningStage | None = None

    def reset(self, running_stage: RunningStage | None = None) -> None:
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
        callback_metrics: dict[str, torch.Tensor] = {}
        logged_metrics: dict[str, torch.Tensor] = {}
        progress_bar_metrics: dict[str, float] = {}
        metric: LoggedMetric
        for metric in module.logged_metrics:
            if not metric.on_step:
                continue
            value: torch.Tensor | None = metric.compute_step(
                self._strategy.root_device
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
        return dict(self.progress_bar_metrics_by_stage[running_stage])

    def log_metrics(
        self,
        metrics: Mapping[str, torch.Tensor | float],
        step: int
    ) -> None:
        scalar_metrics: dict[str, float] = {
            name: self.metric_to_float(value)
            for name, value in metrics.items()
        }
        configured_logger: Logger
        for configured_logger in self._loggers:
            configured_logger.log_metrics(scalar_metrics, step=step)

    def log_hyperparams(self, params: HyperparameterDict) -> None:
        configured_logger: Logger
        for configured_logger in self._loggers:
            configured_logger.log_hyperparams(params)

    def autolog_hyperparameters(
        self,
        module: Module,
        datamodule: DataModule | None
    ) -> None:
        if (
            not self._enable_autolog_hparams
            or not self._loggers
            or self._fast_dev_run
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
        if self._strategy.is_global_zero:
            self.log_hyperparams(filtered_hyperparameters)

    def metric_to_float(self, value: torch.Tensor | float) -> float:
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

    @property
    def loggers(self) -> list[Logger]:
        return self._loggers

    @loggers.setter
    def loggers(self, loggers: list[Logger] | None) -> None:
        resolved_loggers: list[Logger] = [] if loggers is None else list(loggers)
        if not all(isinstance(logger, Logger) for logger in resolved_loggers):
            raise TypeError("loggers must contain only Logger instances.")
        self._loggers: list[Logger] = resolved_loggers

    @property
    def log_dir(self) -> str:
        logger: Logger | None = self.logger
        if isinstance(logger, CSVLogger | TensorBoardLogger):
            directory: str = logger.log_dir
        elif logger is not None:
            directory: str = logger.save_dir
        else:
            directory: str = str(self._default_root_dir)
        return self._strategy.broadcast(directory)

    def _resolve_loggers(
        self,
        logger: bool | Logger | Iterable[Logger] | None
    ) -> list[Logger]:
        if logger is None or logger is True:
            if self._fast_dev_run:
                return [_NoOpLogger()]
            return [self._default_logger()]
        if logger is False:
            return []
        if isinstance(logger, Logger):
            return [_NoOpLogger()] if self._fast_dev_run else [logger]
        if not isinstance(logger, Iterable):
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
        if self._fast_dev_run and loggers:
            return [_NoOpLogger()]
        return loggers

    def _default_logger(self) -> Logger:
        name: str = "syntheticmind_logs"
        version: str = self._next_version(join_path(self._default_root_dir, name))
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

    def _next_version(self, root_dir: str | Path) -> str:
        versions: list[int] = []
        path: str
        for path in list_directory(root_dir):
            name: str = basename(path)
            if not is_directory(path) or not name.startswith("version_"):
                continue
            suffix: str = name.removeprefix("version_")
            if suffix.isdigit():
                versions.append(int(suffix))
        return f"version_{max(versions, default=-1) + 1}"

    def _hyperparameters_equal(
        self,
        first: HyperparameterValue,
        second: HyperparameterValue
    ) -> bool:
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
        return {
            "training": {},
            "validating": {},
            "testing": {},
            "predicting": {},
            "sanity_checking": {}
        }
