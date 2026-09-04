
import inspect
import math
import os
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import IO, Literal, Never, cast

import torch
from loguru import logger as log
from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.accelerators.cpu import CPUAccelerator
from syntheticmind.callbacks.callback import Callback
from syntheticmind.callbacks.checkpoint import Checkpoint
from syntheticmind.callbacks.early_stopping import EarlyStopping
from syntheticmind.callbacks.model_checkpoint import ModelCheckpoint
from syntheticmind.callbacks.progress_bar import ProgressBarCallback
from syntheticmind.core.datamodule import DataModule
from syntheticmind.core.hooks import DataHooks, ModelHooks
from syntheticmind.core.module import Module
from syntheticmind.core.optimizer import (
    LRSchedulerConfig,
    OptimizerLRScheduler,
    SyntheticOptimizer,
    _OptimizerConfigurationParser,
    _SchedulerInstance,
)
from syntheticmind.loggers.logger import Logger
from syntheticmind.loops.batch_limits import (
    normalize_batch_limit,
    normalize_fast_dev_run,
    normalize_sanity_steps,
    resolve_batch_limit,
)
from syntheticmind.loops.evaluation_loop import EvaluationLoop
from syntheticmind.loops.fit_loop import FitLoop
from syntheticmind.loops.prediction_loop import PredictionLoop
from syntheticmind.loops.training_epoch_loop import TrainingEpochLoop
from syntheticmind.loops.validation_schedule import ValidationSchedule
from syntheticmind.precision.precision import Precision
from syntheticmind.plugins import Plugin
from syntheticmind.plugins.io import CheckpointStorageOptions
from syntheticmind.plugins.layer_sync import LayerSync
from syntheticmind.profilers import PassThroughProfiler, Profiler, _resolve_profiler
from syntheticmind.trainer.states import TrainerState
from syntheticmind.strategies.ddp import DDPStrategy
from syntheticmind.strategies.launcher import _ProcessLauncher
from syntheticmind.strategies.parallel import ParallelStrategy
from syntheticmind.strategies.strategy import Strategy
from syntheticmind.trainer.connectors.accelerator_connector import (
    _AcceleratorConnector
)
from syntheticmind.trainer.connectors.callback_connector import _CallbackConnector
from syntheticmind.trainer.connectors.checkpoint_connector import _CheckpointConnector
from syntheticmind.trainer.connectors.data_connector import _DataConnector
from syntheticmind.trainer.connectors.logger_connector.logger_connector import (
    _LoggerConnector
)
from syntheticmind.trainer.connectors.signal_connector import _SignalConnector
from syntheticmind.utilities.console import ConsoleRenderer
from syntheticmind.utilities.combined_loader import CombinedLoader
from syntheticmind.utilities.distributed import is_rank_zero
from syntheticmind.utilities.exceptions import MisconfigurationError, SIGTERMException
from syntheticmind.utilities.filesystem import (
    basename,
    list_directory,
    modified_time,
    normalize_path,
)
from syntheticmind.utilities.model_helpers import ModuleMode, is_overridden
from syntheticmind.utilities.seed import SeedManager
from syntheticmind.utilities.torch_flags import configure_torch_flags
from syntheticmind.utilities.types import (
    CheckpointDict,
    CheckpointValue,
    ClipAlgorithm,
    DataLoaderState,
    DataLoaderSource,
    HyperparameterDict,
    ModelOutput,
    PrecisionInput,
    PrecisionType,
    RunningStage,
    TrainerStage,
)

__all__: list[str] = ["Trainer"]

type _TeardownFailure = tuple[str, BaseException]


def _resolve_training_limits(
    max_epochs: int | None,
    min_epochs: int | None,
    max_steps: int,
    min_steps: int | None
) -> tuple[int, int]:
    if max_epochs is not None and (
        isinstance(max_epochs, bool) or not isinstance(max_epochs, int)
    ):
        raise TypeError(f"`max_epochs` must be an int or None, got {max_epochs!r}.")
    if min_epochs is not None and (
        isinstance(min_epochs, bool) or not isinstance(min_epochs, int)
    ):
        raise TypeError(f"`min_epochs` must be an int or None, got {min_epochs!r}.")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError(f"`max_steps` must be an int, got {max_steps!r}.")
    if min_steps is not None and (
        isinstance(min_steps, bool) or not isinstance(min_steps, int)
    ):
        raise TypeError(f"`min_steps` must be an int or None, got {min_steps!r}.")
    if max_epochs is not None and max_epochs < -1:
        raise MisconfigurationError(
            f"`max_epochs` must be a non-negative integer or -1, got {max_epochs}."
        )
    if min_epochs is not None and min_epochs < 0:
        raise MisconfigurationError(
            f"`min_epochs` must be a non-negative integer or None, got {min_epochs}."
        )
    if max_steps < -1:
        raise MisconfigurationError(
            f"`max_steps` must be a non-negative integer or -1, got {max_steps}."
        )
    if min_steps is not None and min_steps < 0:
        raise MisconfigurationError(
            f"`min_steps` must be a non-negative integer or None, got {min_steps}."
        )
    if max_epochs is None:
        resolved_max_epochs: int = 1000 if max_steps == -1 else -1
    else:
        resolved_max_epochs: int = max_epochs
    if min_epochs is None:
        resolved_min_epochs: int = 1 if min_steps is not None else 0
    else:
        resolved_min_epochs: int = min_epochs
    return resolved_min_epochs, resolved_max_epochs


class Trainer:
    def __init__(
        self,
        *,
        accelerator: str | Accelerator = "auto",
        strategy: str | Strategy = "auto",
        devices: int | str | list[int] = "auto",
        num_nodes: int = 1,
        precision: PrecisionInput | None = None,
        logger: bool | Logger | Iterable[Logger] | None = None,
        callbacks: Callback | list[Callback] | None = None,
        fast_dev_run: int | bool = False,
        max_epochs: int | None = None,
        min_epochs: int | None = None,
        max_steps: int = -1,
        min_steps: int | None = None,
        max_time: str | timedelta | dict[str, int] | None = None,
        limit_train_batches: int | float | None = None,
        limit_val_batches: int | float | None = None,
        limit_test_batches: int | float | None = None,
        limit_predict_batches: int | float | None = None,
        overfit_batches: int | float = 0.0,
        val_check_interval: int | float | str | timedelta | dict[str, int] | None = None,
        check_val_every_n_epoch: int | None = 1,
        num_sanity_val_steps: int | None = None,
        log_every_n_steps: int | None = None,
        enable_checkpointing: bool | None = None,
        enable_progress_bar: bool | None = None,
        enable_model_summary: bool | None = None,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
        deterministic: bool | Literal["warn"] | None = None,
        benchmark: bool | None = None,
        inference_mode: bool = True,
        use_distributed_sampler: bool = True,
        profiler: Profiler | str | None = None,
        detect_anomaly: bool = False,
        barebones: bool = False,
        plugins: Plugin | list[Plugin] | None = None,
        sync_batchnorm: bool = False,
        reload_dataloaders_every_n_epochs: int = 0,
        default_root_dir: str | Path | None = None,
        enable_autolog_hparams: bool = True,
        resume_from_checkpoint: str | Path | None = None,
        seed: int | None = None
    ) -> None:
        resolved_fast_dev_run: int | bool = normalize_fast_dev_run(fast_dev_run)
        resolved_overfit_batches: int | float = normalize_batch_limit(
            overfit_batches,
            "overfit_batches"
        )
        fast_dev_run_enabled: bool = bool(resolved_fast_dev_run)
        overfit_batches_enabled: bool = resolved_overfit_batches > 0
        effective_max_epochs: int | None = max_epochs
        effective_max_steps: int = max_steps
        effective_val_check_interval: int | float | str | timedelta | dict[str, int] | None = (
            val_check_interval
        )
        effective_check_val_every_n_epoch: int | None = check_val_every_n_epoch
        if enable_progress_bar is not None and type(enable_progress_bar) is not bool:
            raise TypeError(
                "`enable_progress_bar` must be a bool or None, got "
                f"{enable_progress_bar!r}."
            )
        if enable_checkpointing is not None and type(enable_checkpointing) is not bool:
            raise TypeError(
                "`enable_checkpointing` must be a bool or None, got "
                f"{enable_checkpointing!r}."
            )
        if enable_model_summary is not None and type(enable_model_summary) is not bool:
            raise TypeError(
                "`enable_model_summary` must be a bool or None, got "
                f"{enable_model_summary!r}."
            )
        if type(detect_anomaly) is not bool:
            raise TypeError(
                f"`detect_anomaly` must be a bool, got {detect_anomaly!r}."
            )
        if type(barebones) is not bool:
            raise TypeError(f"`barebones` must be a bool, got {barebones!r}.")
        if type(sync_batchnorm) is not bool:
            raise TypeError(
                f"`sync_batchnorm` must be a bool, got {sync_batchnorm!r}."
            )
        if type(reload_dataloaders_every_n_epochs) is not int:
            raise TypeError(
                "`reload_dataloaders_every_n_epochs` must be an int, got "
                f"{reload_dataloaders_every_n_epochs!r}."
            )
        if reload_dataloaders_every_n_epochs < 0:
            raise MisconfigurationError(
                "`reload_dataloaders_every_n_epochs` must be >= 0, got "
                f"{reload_dataloaders_every_n_epochs}."
            )
        if type(enable_autolog_hparams) is not bool:
            raise TypeError(
                "`enable_autolog_hparams` must be a bool, got "
                f"{enable_autolog_hparams!r}."
            )
        if (
            log_every_n_steps is not None
            and (
                type(log_every_n_steps) is not int
                or log_every_n_steps < 0
            )
        ):
            raise MisconfigurationError(
                "`log_every_n_steps` must be a non-negative integer or None."
            )
        if barebones:
            if enable_checkpointing:
                raise ValueError(
                    f"`Trainer(barebones=True, enable_checkpointing={enable_checkpointing!r})` was passed. "
                    "Checkpointing can impact raw speed and is disabled in barebones mode."
                )
            enable_checkpointing = False
            if logger is not None and logger is not False:
                raise ValueError(
                    f"`Trainer(barebones=True, logger={logger!r})` was passed. "
                    "Logging can impact raw speed and is disabled in barebones mode."
                )
            logger = False
            if enable_progress_bar:
                raise ValueError(
                    f"`Trainer(barebones=True, enable_progress_bar={enable_progress_bar!r})` was passed. "
                    "The progress bar can impact raw speed and is disabled in barebones mode."
                )
            enable_progress_bar = False
            if log_every_n_steps is not None and log_every_n_steps != 0:
                raise ValueError(
                    f"`Trainer(barebones=True, log_every_n_steps={log_every_n_steps!r})` was passed. "
                    "Logging can impact raw speed and is disabled in barebones mode."
                )
            log_every_n_steps = 0
            if enable_model_summary:
                raise ValueError(
                    f"`Trainer(barebones=True, enable_model_summary={enable_model_summary!r})` was passed. "
                    "Model summary can impact raw speed and is disabled in barebones mode."
                )
            enable_model_summary = False
            if num_sanity_val_steps is not None and num_sanity_val_steps != 0:
                raise ValueError(
                    f"`Trainer(barebones=True, num_sanity_val_steps={num_sanity_val_steps!r})` was passed. "
                    "Sanity checking can impact raw speed and is disabled in barebones mode."
                )
            num_sanity_val_steps = 0
            if resolved_fast_dev_run is not False and resolved_fast_dev_run != 0:
                raise ValueError(
                    f"`Trainer(barebones=True, fast_dev_run={fast_dev_run!r})` was passed. "
                    "Development runs are disabled in barebones mode."
                )
            if detect_anomaly:
                raise ValueError(
                    f"`Trainer(barebones=True, detect_anomaly={detect_anomaly!r})` was passed. "
                    "Anomaly detection can impact raw speed and is disabled in barebones mode."
                )
            if profiler is not None:
                raise ValueError(
                    f"`Trainer(barebones=True, profiler={profiler!r})` was passed. "
                    "Profiling can impact raw speed and is disabled in barebones mode."
                )
        self._accelerator_connector: _AcceleratorConnector = _AcceleratorConnector(
            accelerator,
            strategy,
            devices,
            num_nodes,
            precision,
            plugins,
            sync_batchnorm
        )
        self.accelerator: Accelerator = self._accelerator_connector.accelerator
        self.devices: int | list[int] = self._accelerator_connector.devices
        self.parallel_devices: list[torch.device] = (
            self._accelerator_connector.parallel_devices
        )
        self.strategy: Strategy = self._accelerator_connector.strategy
        self._precision_plugin: Precision = (
            self._accelerator_connector.precision_plugin
        )
        resolved_log_every_n_steps: int = (
            50 if log_every_n_steps is None else log_every_n_steps
        )
        resolved_enable_progress_bar: bool = (
            True if enable_progress_bar is None else enable_progress_bar
        )
        resolved_enable_checkpointing: bool = (
            True if enable_checkpointing is None else enable_checkpointing
        )
        resolved_enable_model_summary: bool = (
            True if enable_model_summary is None else enable_model_summary
        )
        if default_root_dir is None:
            self._default_root_dir: str = os.getcwd()
        else:
            configured_root_value: str | bytes = os.fspath(default_root_dir)
            if not isinstance(configured_root_value, str):
                raise TypeError("`default_root_dir` must resolve to a string path.")
            configured_root_dir: str = configured_root_value
            self._default_root_dir: str = configured_root_dir or os.getcwd()
        resolved_min_epochs, resolved_max_epochs = _resolve_training_limits(
            effective_max_epochs,
            min_epochs,
            effective_max_steps,
            min_steps
        )
        resolved_num_sanity_val_steps: int | float = normalize_sanity_steps(
            num_sanity_val_steps
        )
        self.limit_train_batches: int | float = normalize_batch_limit(
            limit_train_batches,
            "limit_train_batches"
        )
        self.limit_val_batches: int | float = normalize_batch_limit(
            limit_val_batches,
            "limit_val_batches"
        )
        self.limit_test_batches: int | float = normalize_batch_limit(
            limit_test_batches,
            "limit_test_batches"
        )
        self.limit_predict_batches: int | float = normalize_batch_limit(
            limit_predict_batches,
            "limit_predict_batches"
        )
        if overfit_batches_enabled:
            self.limit_train_batches = resolved_overfit_batches
            self.limit_val_batches = resolved_overfit_batches
        if fast_dev_run_enabled:
            num_batches: int = int(resolved_fast_dev_run)
            if not overfit_batches_enabled:
                self.limit_train_batches = num_batches
                self.limit_val_batches = num_batches
            self.limit_test_batches = num_batches
            self.limit_predict_batches = num_batches
            effective_max_epochs = 1
            effective_max_steps = num_batches
            resolved_num_sanity_val_steps = 0
            effective_val_check_interval = 1.0
            effective_check_val_every_n_epoch = 1
            resolved_min_epochs, resolved_max_epochs = _resolve_training_limits(
                effective_max_epochs,
                min_epochs,
                effective_max_steps,
                min_steps
            )
        if gradient_clip_val is not None and not isinstance(gradient_clip_val, (int, float)):
            raise TypeError(
                f"`gradient_clip_val` should be an int or a float. Got {gradient_clip_val}."
            )
        resolved_gradient_clip_algorithm: ClipAlgorithm | None = None
        if gradient_clip_algorithm is not None:
            normalized_gradient_clip_algorithm: str = gradient_clip_algorithm.lower()
            if normalized_gradient_clip_algorithm not in {"norm", "value"}:
                raise MisconfigurationError(
                    f"`gradient_clip_algorithm` {gradient_clip_algorithm} is invalid. "
                    "Allowed algorithms: ('norm', 'value')."
                )
            resolved_gradient_clip_algorithm: ClipAlgorithm | None = (
                "norm" if normalized_gradient_clip_algorithm == "norm" else "value"
            )

        self.gradient_clip_val: int | float | None = gradient_clip_val
        self.gradient_clip_algorithm: ClipAlgorithm | None = resolved_gradient_clip_algorithm
        self.log_every_n_steps: int = resolved_log_every_n_steps
        self.enable_progress_bar: bool = resolved_enable_progress_bar
        self.enable_model_summary: bool = resolved_enable_model_summary
        self.enable_autolog_hparams: bool = enable_autolog_hparams
        self.barebones: bool = barebones
        self.num_sanity_val_steps: int | float = resolved_num_sanity_val_steps
        self.fast_dev_run: int | bool = resolved_fast_dev_run
        self.overfit_batches: int | float = resolved_overfit_batches
        self.use_distributed_sampler: bool = use_distributed_sampler
        self.sync_batchnorm: bool = sync_batchnorm
        if type(inference_mode) is not bool:
            raise TypeError(
                f"`inference_mode` must be a bool, got {inference_mode!r}."
            )
        configure_torch_flags(deterministic, benchmark)

        self.deterministic: bool | Literal["warn"] | None = deterministic
        self.inference_mode: bool = inference_mode
        self._detect_anomaly: bool = detect_anomaly
        self.resume_from_checkpoint: str | Path | None = (
            normalize_path(resume_from_checkpoint)
            if resume_from_checkpoint is not None
            else None
        )

        if accumulate_grad_batches < 1:
            raise MisconfigurationError("accumulate_grad_batches must be >= 1")
        if seed is not None:
            SeedManager.seed(seed)

        self._callback_connector: _CallbackConnector = _CallbackConnector(
            callbacks,
            resolved_enable_checkpointing,
            resolved_enable_progress_bar,
            resolved_enable_model_summary,
            max_time
        )
        self._checkpoint_connector: _CheckpointConnector = _CheckpointConnector(
            self.strategy,
            self._callback_connector,
            fast_dev_run_enabled
        )

        self._logger_connector: _LoggerConnector = _LoggerConnector(
            self.strategy,
            logger,
            self.default_root_dir,
            fast_dev_run_enabled,
            enable_autolog_hparams
        )
        self.profiler: Profiler = _resolve_profiler(profiler)
        self._data_connector: _DataConnector = _DataConnector(
            self.strategy,
            self.profiler,
            self.limit_train_batches,
            self.limit_val_batches,
            self.limit_test_batches,
            self.limit_predict_batches,
            self.overfit_batches,
            self.use_distributed_sampler,
            reload_dataloaders_every_n_epochs
        )
        self.state: TrainerState = TrainerState()
        self._signal_connector: _SignalConnector = _SignalConnector(self.strategy)
        self._is_mid_epoch_validation: bool = False
        self._validation_schedule: ValidationSchedule = ValidationSchedule(
            effective_val_check_interval,
            effective_check_val_every_n_epoch
        )
        self.val_check_interval: int | float | str | timedelta | dict[str, int] = (
            self._validation_schedule.val_check_interval
        )
        self.check_val_every_n_epoch: int | None = (
            self._validation_schedule.check_val_every_n_epoch
        )
        self.val_check_batch: int | None = None

        self.fit_loop: FitLoop = FitLoop(
            min_epochs=resolved_min_epochs,
            max_epochs=resolved_max_epochs
        )
        self.fit_loop.epoch_loop = TrainingEpochLoop(
            min_steps=min_steps,
            max_steps=effective_max_steps,
            accumulate_grad_batches=accumulate_grad_batches,
            precision=self.precision,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm=self.gradient_clip_algorithm,
            log_every_n_steps=resolved_log_every_n_steps
        )
        self.validate_loop: EvaluationLoop = EvaluationLoop(
            stage="validate",
            inference_mode=inference_mode
        )
        self.test_loop: EvaluationLoop = EvaluationLoop(
            stage="test",
            inference_mode=inference_mode
        )
        self.predict_loop: PredictionLoop = PredictionLoop(
            inference_mode=inference_mode
        )

        self._connect_loops()

        self.module: Module | None = None
        self.datamodule: DataModule | None = None
        self.optimizers: list[torch.optim.Optimizer] = []
        self._synthetic_optimizers: list[SyntheticOptimizer] = []
        self.schedulers: list[_SchedulerInstance] = []
        self.lr_scheduler_configs: list[LRSchedulerConfig] = []
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: _SchedulerInstance | None = None
        self.scheduler_config: LRSchedulerConfig | None = None
        self._dataloader_fetch_time_seconds: dict[RunningStage, float] = {}
        self._dataloader_iterator_creation_time_seconds: dict[RunningStage, float] = {}
        self._active_exception: BaseException | None = None
        self._profiler_is_setup: bool = False
        self._layer_sync_model: Module | None = None

    def fit(
        self,
        model: Module,
        train_dataloaders: DataLoaderSource | CombinedLoader | None = None,
        val_dataloaders: DataLoaderSource | CombinedLoader | None = None,
        datamodule: DataModule | None = None,
        ckpt_path: str | Path | None = None,
        weights_only: bool | None = None
    ) -> None:
        if not isinstance(model, Module):
            raise TypeError(
                f"`model` must be a Module, got {type(model).__qualname__}."
            )
        module: Module = model
        if isinstance(train_dataloaders, DataModule):
            datamodule = train_dataloaders
            train_dataloaders = None
        if (
            datamodule is not None
            and (train_dataloaders is not None or val_dataloaders is not None)
        ):
            raise MisconfigurationError(
                "You cannot pass `train_dataloaders` or `val_dataloaders` "
                "to `Trainer.fit(datamodule=...)`."
            )
        if weights_only is not None and type(weights_only) is not bool:
            raise TypeError("`weights_only` must be a bool or None.")
        requested_checkpoint_path: str | Path | None = ckpt_path
        if (
            requested_checkpoint_path is None
            and not self._checkpoint_connector.user_managed
        ):
            requested_checkpoint_path = self.resume_from_checkpoint
        resume_path: str | Path | None = self._select_checkpoint_path(
            requested_checkpoint_path,
            stage="fit",
            model_provided=True,
            ckpt_path_provided=ckpt_path is not None
        )
        self._attach_model_callbacks(module, "fit")
        launcher: _ProcessLauncher | None = (
            self.strategy.launcher
            if isinstance(self.strategy, ParallelStrategy)
            else None
        )
        if launcher is not None and launcher.should_launch:
            try:
                launcher.launch(
                    "fit",
                    self,
                    module,
                    datamodule,
                    {
                        "train_dataloaders": train_dataloaders,
                        "val_dataloaders": val_dataloaders,
                        "ckpt_path": resume_path,
                        "weights_only": weights_only
                    }
                )
            finally:
                self._clear_model_callback_configuration("fit")
                self.state.finish()
            return
        self.state.set_stage("fit")
        self._active_exception = None
        self.state.reset_on_run()
        self._is_mid_epoch_validation = False
        self._logger_connector.clear_metric_views()
        module._reset_logging_state()
        self._reset_progress_bar_metrics()
        self._data_connector.reset_fit()
        self._checkpoint_connector.reset_training_continuation()
        self.fit_loop.should_stop = False
        if self.fit_loop.epoch_loop is not None:
            self.fit_loop.epoch_loop.reset_on_run()
        self._attach_data(module, datamodule)
        self._data_connector.attach_source(
            "train",
            module,
            datamodule,
            train_dataloaders
        )
        self._data_connector.attach_source(
            "val",
            module,
            datamodule,
            val_dataloaders
        )
        module._current_epoch = 0
        module._global_step = 0
        module_mode: ModuleMode = ModuleMode()
        module_mode.capture(module)
        fit_completed: bool = False

        try:
            self._signal_connector.register_signal_handlers()
            self.strategy.setup_environment()
            self._announce_barebones_stage("fit")
            self._setup_profiler("fit", module)

            self._prepare_data_and_setup(module, datamodule, "fit")

            preloaded_checkpoint: CheckpointDict | None = None
            if resume_path is not None:
                preloaded_checkpoint: CheckpointDict | None = (
                    self._checkpoint_connector.load_metadata(
                        resume_path,
                        weights_only
                    )
                )
                resume_epoch: int = self._checkpoint_connector.integer_value(
                    preloaded_checkpoint.get("epoch"),
                    0
                )
                resume_global_step: int = self._checkpoint_connector.integer_value(
                    preloaded_checkpoint.get("global_step"),
                    0
                )
                self.state.load_state_dict(
                    {
                        "current_epoch": resume_epoch,
                        "global_step": resume_global_step
                    }
                )
                module._current_epoch = resume_epoch
                module._global_step = resume_global_step
                if datamodule is not None:
                    self._checkpoint_connector.load_datamodule_state(
                        preloaded_checkpoint,
                        datamodule
                    )

            if (
                self.limit_train_batches != 0
                and is_overridden("training_step", module, Module)
            ):
                self._request_dataloaders("train")
            else:
                self._data_connector.total_train_batches = 0
            self._configure_validation_schedule(reset_time=True)

            if is_rank_zero():
                combined_loader: CombinedLoader | None = self._data_connector.train_combined_loader
                if combined_loader is not None:
                    for train_dataloader in combined_loader.flattened:
                        if (
                            isinstance(train_dataloader, torch.utils.data.DataLoader)
                            and train_dataloader.num_workers == 0
                            and not isinstance(self.accelerator, CPUAccelerator)
                        ):
                            log.warning("num_workers=0 with GPU accelerator. Consider increasing for better throughput.")
                            break

            device: torch.device = self._resolve_device()
            module = self._setup_model(module, device)

            training_step_signature: inspect.Signature = inspect.signature(module.training_step)
            if "optimizer_idx" in training_step_signature.parameters:
                raise RuntimeError(
                    "Training with multiple optimizers is only supported with manual optimization. Remove "
                    "the optimizer_idx parameter from training_step, set automatic_optimization = False, "
                    "and access optimizers through self.optimizers()."
                )
            optimization_configuration: OptimizerLRScheduler = self._call_module_hook(
                module,
                "configure_optimizers",
                module.configure_optimizers
            )
            optimizer_parser: _OptimizerConfigurationParser = (
                _OptimizerConfigurationParser(
                    module.automatic_optimization,
                    is_overridden("lr_scheduler_step", module, Module)
                )
            )
            optimizer_initialization: tuple[
                list[torch.optim.Optimizer],
                list[LRSchedulerConfig]
            ] = optimizer_parser.resolve(optimization_configuration)
            self.optimizers: list[torch.optim.Optimizer] = optimizer_initialization[0]
            self.lr_scheduler_configs: list[LRSchedulerConfig] = optimizer_initialization[1]
            module, self.optimizers, self.lr_scheduler_configs = (
                self._connect_precision_plugin(
                    module,
                    self.optimizers,
                    self.lr_scheduler_configs
                )
            )
            epoch_loop: TrainingEpochLoop | None = self.fit_loop.epoch_loop
            if epoch_loop is None:
                raise RuntimeError("Trainer fit loop has no training epoch loop.")
            strategy_model: torch.nn.Module | None = self.strategy.model
            if strategy_model is None:
                raise RuntimeError("Strategy has no model for optimizer connection.")
            self._synthetic_optimizers: list[SyntheticOptimizer] = []
            optimizer: torch.optim.Optimizer
            for optimizer in self.optimizers:
                synthetic_optimizer: SyntheticOptimizer = SyntheticOptimizer(optimizer)
                synthetic_optimizer.connect(
                    strategy_model,
                    partial(epoch_loop.optimizer_step, self, module),
                    module.toggled_optimizer
                )
                self._synthetic_optimizers.append(synthetic_optimizer)

            if not module.automatic_optimization:
                assert self.fit_loop.epoch_loop is not None
                if self.fit_loop.epoch_loop.accumulate_grad_batches != 1:
                    raise MisconfigurationError(
                        "Manual optimization does not support accumulate_grad_batches != 1. "
                        "Set accumulate_grad_batches = 1 and manage accumulation inside training_step if needed."
                    )
                if self.gradient_clip_val is not None and self.gradient_clip_val > 0:
                    raise MisconfigurationError(
                        "Manual optimization does not support automatic gradient clipping. "
                        "Call module.clip_gradients(...) explicitly inside training_step if needed."
                    )

            self.optimizer: torch.optim.Optimizer | None = self.optimizers[0] if self.optimizers else None
            self.schedulers: list[_SchedulerInstance] = [
                configuration.scheduler for configuration in self.lr_scheduler_configs
            ]
            self.scheduler: _SchedulerInstance | None = (
                self.schedulers[0] if self.schedulers else None
            )
            self.scheduler_config: LRSchedulerConfig | None = (
                self.lr_scheduler_configs[0] if self.lr_scheduler_configs else None
            )


            self._resolve_checkpoint_callback_directories()
            if resume_path is not None:
                assert preloaded_checkpoint is not None
                self._restore_checkpoint(
                    resume_path,
                    module,
                    weights_only=weights_only,
                    checkpoint=None,
                    restore_datamodule_state=False
                )
            synthetic_optimizer: SyntheticOptimizer
            for synthetic_optimizer in self._synthetic_optimizers:
                synthetic_optimizer.move_state_to(device)

            self._callback_connector.validate_state_keys()

            self._call_callback_hooks(module, "on_fit_start")
            self._call_module_hook(
                module,
                "on_fit_start",
                module.on_fit_start
            )

            self._autolog_hyperparameters(module, datamodule)

            if (
                self.enable_validation
                and self.num_sanity_val_steps > 0
                and resume_path is None
            ):
                self._sanity_check_validation(module)

            self.state.set_running_stage("training")
            module._running_stage = "training"

            with torch.autograd.set_detect_anomaly(self._detect_anomaly):
                self.fit_loop.run()

            self._call_callback_hooks(module, "on_fit_end")
            self._call_module_hook(module, "on_fit_end", module.on_fit_end)
            self._raise_if_sigterm()
            fit_completed: bool = True
        except BaseException as exception:
            self._handle_exception(module, datamodule, exception)
        finally:
            if not fit_completed:
                module_mode.restore(module)
            cleanup_exception: BaseException | None = self._teardown_stage(module, datamodule, "fit")
            module._running_stage = None
            module._current_fx_name = None
            self.state.finish()
            self._active_exception = None
            self._is_mid_epoch_validation = False
            self._clear_model_callback_configuration("fit")
            if self.fit_loop.epoch_loop is not None:
                self.fit_loop.epoch_loop.reset_on_teardown()
            if cleanup_exception is not None:
                raise cleanup_exception

    def validate(
        self,
        model: Module | None = None,
        dataloaders: DataLoaderSource | CombinedLoader | None = None,
        ckpt_path: str | Path | None = None,
        verbose: bool = True,
        datamodule: DataModule | None = None,
        weights_only: bool | None = None
    ) -> list[dict[str, float]]:
        model_provided: bool = model is not None
        module: Module = self._resolve_model(model, "validate")
        if isinstance(dataloaders, DataModule):
            datamodule = dataloaders
            dataloaders = None
        if dataloaders is not None and datamodule is not None:
            raise MisconfigurationError(
                "You cannot pass both `Trainer.validate(dataloaders=..., datamodule=...)`."
            )
        if type(verbose) is not bool:
            raise TypeError("`verbose` must be a bool.")
        if weights_only is not None and type(weights_only) is not bool:
            raise TypeError("`weights_only` must be a bool or None.")
        self.validate_loop.verbose = verbose
        selected_checkpoint_path: str | Path | None = self._select_checkpoint_path(
            ckpt_path,
            stage="validate",
            model_provided=model_provided,
            ckpt_path_provided=ckpt_path is not None
        )
        self._attach_model_callbacks(module, "validate")
        launcher: _ProcessLauncher | None = (
            self.strategy.launcher
            if isinstance(self.strategy, ParallelStrategy)
            else None
        )
        if launcher is not None and launcher.should_launch:
            try:
                return cast(
                    list[dict[str, float]],
                    launcher.launch(
                        "validate",
                        self,
                        module,
                        datamodule,
                        {
                            "dataloaders": dataloaders,
                            "ckpt_path": selected_checkpoint_path,
                            "verbose": verbose,
                            "weights_only": weights_only
                        }
                    )
                )
            finally:
                self._clear_model_callback_configuration("validate")
                self.state.finish()
        self.state.set_stage("validate")
        self._active_exception = None
        self._logger_connector.clear_metric_views()
        module._reset_logging_state()
        self._reset_progress_bar_metrics("validating")
        self._data_connector.reset_stage("val")
        self._attach_data(module, datamodule)
        self._data_connector.attach_source(
            "val",
            module,
            datamodule,
            dataloaders
        )

        module_mode: ModuleMode = ModuleMode()
        module_mode.capture(module)
        stage_completed: bool = False
        metrics: list[dict[str, float]] = []
        try:
            self._signal_connector.register_signal_handlers()
            self.strategy.setup_environment()
            self._announce_barebones_stage("validate")
            self._setup_profiler("validate", module)

            self._prepare_data_and_setup(module, datamodule, "validate")
            self._request_dataloaders("val")

            device: torch.device = self._resolve_device()
            module = self._setup_model(module, device)
            module, _, _ = self._connect_precision_plugin(module, [], [])
            if selected_checkpoint_path is not None:
                self._resolve_checkpoint_callback_directories()
                self._restore_checkpoint(
                    selected_checkpoint_path,
                    module,
                    weights_only=weights_only
                )

            self._autolog_hyperparameters(module, datamodule)

            metrics: list[dict[str, float]] = self.validate_loop.run()
            stage_completed: bool = True
        except BaseException as exception:
            self._handle_exception(module, datamodule, exception)
        finally:
            if not stage_completed:
                module_mode.restore(module)
            cleanup_exception: BaseException | None = self._teardown_stage(module, datamodule, "validate")
            module._running_stage = None
            module._current_fx_name = None
            self.state.finish()
            self._active_exception = None
            self._clear_model_callback_configuration("validate")
            if cleanup_exception is not None:
                raise cleanup_exception

        if verbose and self.is_global_zero:
            ConsoleRenderer().write_metrics("validation", metrics)
        return metrics

    def test(
        self,
        model: Module | None = None,
        dataloaders: DataLoaderSource | CombinedLoader | None = None,
        ckpt_path: str | Path | None = None,
        verbose: bool = True,
        datamodule: DataModule | None = None,
        weights_only: bool | None = None
    ) -> list[dict[str, float]]:
        model_provided: bool = model is not None
        module: Module = self._resolve_model(model, "test")
        if isinstance(dataloaders, DataModule):
            datamodule = dataloaders
            dataloaders = None
        if dataloaders is not None and datamodule is not None:
            raise MisconfigurationError(
                "You cannot pass both `Trainer.test(dataloaders=..., datamodule=...)`."
            )
        if type(verbose) is not bool:
            raise TypeError("`verbose` must be a bool.")
        if weights_only is not None and type(weights_only) is not bool:
            raise TypeError("`weights_only` must be a bool or None.")
        self.test_loop.verbose = verbose
        selected_checkpoint_path: str | Path | None = self._select_checkpoint_path(
            ckpt_path,
            stage="test",
            model_provided=model_provided,
            ckpt_path_provided=ckpt_path is not None
        )
        self._attach_model_callbacks(module, "test")
        launcher: _ProcessLauncher | None = (
            self.strategy.launcher
            if isinstance(self.strategy, ParallelStrategy)
            else None
        )
        if launcher is not None and launcher.should_launch:
            try:
                return cast(
                    list[dict[str, float]],
                    launcher.launch(
                        "test",
                        self,
                        module,
                        datamodule,
                        {
                            "dataloaders": dataloaders,
                            "ckpt_path": selected_checkpoint_path,
                            "verbose": verbose,
                            "weights_only": weights_only
                        }
                    )
                )
            finally:
                self._clear_model_callback_configuration("test")
                self.state.finish()
        self.state.set_stage("test")
        self._active_exception = None
        self._logger_connector.clear_metric_views()
        module._reset_logging_state()
        self._reset_progress_bar_metrics("testing")
        self._data_connector.total_test_batches: list[int | None] | None = None
        self._data_connector.reset_stage("test")
        self._attach_data(module, datamodule)
        self._data_connector.attach_source(
            "test",
            module,
            datamodule,
            dataloaders
        )

        module_mode: ModuleMode = ModuleMode()
        module_mode.capture(module)
        stage_completed: bool = False
        metrics: list[dict[str, float]] = []
        try:
            self._signal_connector.register_signal_handlers()
            self.strategy.setup_environment()
            self._announce_barebones_stage("test")
            self._setup_profiler("test", module)

            self._prepare_data_and_setup(module, datamodule, "test")
            self._request_dataloaders("test")

            device: torch.device = self._resolve_device()
            module = self._setup_model(module, device)
            module, _, _ = self._connect_precision_plugin(module, [], [])
            if selected_checkpoint_path is not None:
                self._resolve_checkpoint_callback_directories()
                self._restore_checkpoint(
                    selected_checkpoint_path,
                    module,
                    weights_only=weights_only
                )

            self._autolog_hyperparameters(module, datamodule)

            metrics: list[dict[str, float]] = self.test_loop.run()
            stage_completed: bool = True
        except BaseException as exception:
            self._handle_exception(module, datamodule, exception)
        finally:
            if not stage_completed:
                module_mode.restore(module)
            cleanup_exception: BaseException | None = self._teardown_stage(module, datamodule, "test")
            module._running_stage = None
            module._current_fx_name = None
            self.state.finish()
            self._active_exception = None
            self._clear_model_callback_configuration("test")
            if cleanup_exception is not None:
                raise cleanup_exception

        if verbose and self.is_global_zero:
            ConsoleRenderer().write_metrics("test", metrics)
        return metrics

    def predict(
        self,
        model: Module | None = None,
        dataloaders: DataLoaderSource | CombinedLoader | None = None,
        datamodule: DataModule | None = None,
        return_predictions: bool | None = None,
        ckpt_path: str | Path | None = None,
        weights_only: bool | None = None
    ) -> list[ModelOutput] | list[list[ModelOutput]] | None:
        model_provided: bool = model is not None
        module: Module = self._resolve_model(model, "predict")
        if isinstance(dataloaders, DataModule):
            datamodule = dataloaders
            dataloaders = None
        if dataloaders is not None and datamodule is not None:
            raise MisconfigurationError(
                "You cannot pass both `Trainer.predict(dataloaders=..., datamodule=...)`."
            )
        if weights_only is not None and type(weights_only) is not bool:
            raise TypeError("`weights_only` must be a bool or None.")
        if return_predictions is not None and type(return_predictions) is not bool:
            raise TypeError("`return_predictions` must be a bool or None.")
        selected_checkpoint_path: str | Path | None = self._select_checkpoint_path(
            ckpt_path,
            stage="predict",
            model_provided=model_provided,
            ckpt_path_provided=ckpt_path is not None
        )
        self._attach_model_callbacks(module, "predict")
        launcher: _ProcessLauncher | None = (
            self.strategy.launcher
            if isinstance(self.strategy, ParallelStrategy)
            else None
        )
        if launcher is not None and launcher.should_launch:
            try:
                return cast(
                    list[ModelOutput] | list[list[ModelOutput]] | None,
                    launcher.launch(
                        "predict",
                        self,
                        module,
                        datamodule,
                        {
                            "dataloaders": dataloaders,
                            "return_predictions": return_predictions,
                            "ckpt_path": selected_checkpoint_path,
                            "weights_only": weights_only
                        }
                    )
                )
            finally:
                self._clear_model_callback_configuration("predict")
                self.state.finish()
        self.state.set_stage("predict")
        self._active_exception = None
        self._logger_connector.clear_metric_views()
        module._reset_logging_state()
        self._reset_progress_bar_metrics("predicting")
        self._data_connector.reset_stage("predict")
        self._attach_data(module, datamodule)
        self._data_connector.attach_source(
            "predict",
            module,
            datamodule,
            dataloaders
        )

        module_mode: ModuleMode = ModuleMode()
        module_mode.capture(module)
        stage_completed: bool = False
        predictions: list[ModelOutput] | list[list[ModelOutput]] | None = None
        try:
            self._signal_connector.register_signal_handlers()
            self.strategy.setup_environment()
            self._announce_barebones_stage("predict")
            self._setup_profiler("predict", module)

            self._prepare_data_and_setup(module, datamodule, "predict")
            self._request_dataloaders("predict")

            device: torch.device = self._resolve_device()
            module = self._setup_model(module, device)
            module, _, _ = self._connect_precision_plugin(module, [], [])
            if selected_checkpoint_path is not None:
                self._resolve_checkpoint_callback_directories()
                self._restore_checkpoint(
                    selected_checkpoint_path,
                    module,
                    weights_only=weights_only
                )

            self._autolog_hyperparameters(module, datamodule)

            self.predict_loop.return_predictions = (
                launcher is None or launcher.start_method == "popen"
                if return_predictions is None
                else return_predictions
            )
            predictions = self.predict_loop.run()
            stage_completed: bool = True
        except BaseException as exception:
            self._handle_exception(module, datamodule, exception)
        finally:
            if not stage_completed:
                module_mode.restore(module)
            cleanup_exception: BaseException | None = self._teardown_stage(module, datamodule, "predict")
            module._running_stage = None
            module._current_fx_name = None
            self.state.finish()
            self._active_exception = None
            self._clear_model_callback_configuration("predict")
            if cleanup_exception is not None:
                raise cleanup_exception

        return predictions

    @contextmanager
    def init_module(
        self,
        empty_init: bool | None = None
    ) -> Iterator[None]:
        with self.strategy.tensor_init_context(empty_init=empty_init):
            yield

    def _resolve_model(
        self,
        model: Module | None,
        stage: Literal["validate", "test", "predict"]
    ) -> Module:
        if model is not None:
            if not isinstance(model, Module):
                raise TypeError(
                    f"`model` must be a Module or None, got "
                    f"{type(model).__qualname__}."
                )
            return model
        if self.module is None:
            raise TypeError(
                f"`Trainer.{stage}()` requires a `Module` when it has not "
                "been passed in a previous run."
            )
        return self.module

    def _attach_data(
        self,
        module: Module,
        datamodule: DataModule | None
    ) -> None:
        self.module = module
        self.datamodule = datamodule
        module.trainer = self
        self.strategy.connect(module)
        if datamodule is not None:
            datamodule.trainer = self

    def _attach_model_callbacks(
        self,
        module: Module,
        stage: TrainerStage
    ) -> None:
        self.state.set_stage(stage)
        try:
            self.module = module
            module.trainer = self
            self.strategy.connect(module)
            self._callback_connector.configure_model_callbacks(
                module,
                stage,
                partial(
                    self._call_module_hook,
                    module,
                    "configure_callbacks",
                    module.configure_callbacks
                )
            )
        except BaseException:
            self.state.finish()
            raise

    def _clear_model_callback_configuration(self, stage: TrainerStage) -> None:
        self._callback_connector.clear_model_configuration(stage)

    def _prepare_data_and_setup(
        self,
        module: Module,
        datamodule: DataModule | None,
        stage: TrainerStage
    ) -> None:
        if is_rank_zero() and datamodule is not None:
            self._call_datamodule_hook(
                datamodule,
                "prepare_data",
                datamodule.prepare_data
            )
        if is_rank_zero() and is_overridden(
            "prepare_data",
            module,
            DataHooks
        ):
            self._call_module_hook(
                module,
                "prepare_data",
                module.prepare_data
            )
        self._call_strategy_hook("barrier", self.strategy.barrier)
        if datamodule is not None:
            self._call_datamodule_hook(
                datamodule,
                "setup",
                datamodule.setup,
                stage
            )
        self._call_callback_hooks(module, "setup", stage)
        self._call_module_hook(module, "setup", module.setup, stage)

    def _request_dataloaders(
        self,
        stage: Literal["train", "val", "test", "predict"]
    ) -> CombinedLoader:
        trainer_stage: TrainerStage | None = self.state.stage
        if trainer_stage is None:
            raise RuntimeError("Dataloaders cannot be requested outside a Trainer stage.")
        return self._data_connector.request_dataloaders(
            stage,
            trainer_stage,
            self.current_epoch,
            self._call_module_hook,
            self._call_datamodule_hook
        )

    def _ensure_train_dataloader(self) -> None:
        if self._data_connector.train_loader_is_current(self.current_epoch):
            return
        self._request_dataloaders("train")
        self._configure_validation_schedule(reset_time=False)

    def _ensure_val_dataloader(self) -> None:
        if self._data_connector.val_loader_is_current(self.current_epoch):
            return
        self._request_dataloaders("val")
        self._configure_validation_schedule(reset_time=False)

    def _configure_validation_schedule(self, reset_time: bool) -> None:
        assert self.fit_loop.epoch_loop is not None
        max_batches: int | None = self._data_connector.total_train_batches
        module: Module | None = self.module
        source: DataModule | Module | DataLoaderSource | CombinedLoader | None = (
            self._data_connector.val_source
        )
        source_is_defined: bool = source is not None
        if isinstance(source, DataModule):
            source_is_defined = is_overridden(
                "val_dataloader",
                source,
                DataModule
            )
        elif isinstance(source, Module):
            source_is_defined = is_overridden(
                "val_dataloader",
                source,
                DataHooks
            )
        validation_enabled: bool = (
            self.limit_val_batches != 0
            and module is not None
            and is_overridden("validation_step", module, Module)
            and source_is_defined
        )
        if validation_enabled and self._data_connector.total_val_batches is not None:
            validation_enabled = any(
                resolve_batch_limit(total, self.limit_val_batches, "val") != 0
                for total in self._data_connector.total_val_batches
            )
        self._validation_schedule.configure(
            max_batches,
            validation_enabled,
            reset_time
        )
        self.val_check_batch: int | None = self._validation_schedule.batch_interval

    def _record_dataloader_fetch_time(self, running_stage: RunningStage, duration_seconds: float) -> None:
        self._dataloader_fetch_time_seconds[running_stage] = max(0.0, duration_seconds)

    def _reset_progress_bar_metrics(self, running_stage: RunningStage | None = None) -> None:
        self._logger_connector.reset(running_stage)

    def _update_progress_bar_metrics(
        self,
        running_stage: RunningStage,
        metrics: dict[str, float]
    ) -> None:
        self._logger_connector.update_progress_bar_metrics(running_stage, metrics)

    def _update_logged_metrics(self, metrics: dict[str, torch.Tensor]) -> None:
        self._logger_connector.update_logged_metrics(metrics)

    def _preview_logged_metrics(
        self,
        module: Module,
        running_stage: RunningStage
    ) -> None:
        self._logger_connector.preview_logged_metrics(
            module,
            running_stage,
            self.sanity_checking
        )

    def _get_progress_bar_metrics(self, running_stage: RunningStage) -> dict[str, float]:
        return self._logger_connector.progress_bar_metrics(running_stage)

    def _get_dataloader_fetch_time(self, running_stage: RunningStage) -> float | None:
        return self._dataloader_fetch_time_seconds.get(running_stage)

    def _record_dataloader_iterator_creation_time(
        self, running_stage: RunningStage, duration_seconds: float
    ) -> None:
        self._dataloader_iterator_creation_time_seconds[running_stage] = max(0.0, duration_seconds)

    def _consume_dataloader_iterator_creation_time(self, running_stage: RunningStage) -> float | None:
        return self._dataloader_iterator_creation_time_seconds.pop(running_stage, None)

    def _connect_loops(self) -> None:
        self.fit_loop.trainer = self
        assert self.fit_loop.epoch_loop is not None
        self.fit_loop.epoch_loop.trainer = self
        self.validate_loop.trainer = self
        self.test_loop.trainer = self
        self.predict_loop.trainer = self

    def _announce_barebones_stage(self, stage: TrainerStage) -> None:
        if not self.barebones or not self.is_global_zero:
            return
        ConsoleRenderer().write_permanent(
            f"syntheticmind · {stage} · barebones"
        )

    def _setup_profiler(self, stage: TrainerStage, module: Module) -> None:
        if self._profiler_is_setup:
            raise RuntimeError("The profiler is already active for another Trainer stage.")
        self.profiler._attach_module(module)
        local_rank: int | None = self.local_rank if self.world_size > 1 else None
        log_dir: str | None = (
            None if isinstance(self.profiler, PassThroughProfiler) else self.log_dir
        )
        try:
            self.profiler.setup(
                stage=stage,
                local_rank=local_rank,
                log_dir=log_dir
            )
        except BaseException as exception:
            try:
                self.profiler.teardown(stage=stage)
            except BaseException as teardown_exception:
                exception.add_note(
                    "Profiler teardown after setup failure raised "
                    f"{type(teardown_exception).__name__}: {teardown_exception}"
                )
            raise
        self._profiler_is_setup = True

    def _teardown_profiler(self) -> None:
        if not self._profiler_is_setup:
            return
        try:
            self.profiler.describe()
        finally:
            self._profiler_is_setup = False

    def _call_callback_hook[ReturnT](
        self,
        callback: Callback,
        module: Module,
        hook_name: str,
        hook: Callable[..., ReturnT],
        *args: object,
        **kwargs: object
    ) -> ReturnT:
        action_name: str = f"[Callback]{callback.state_key}.{hook_name}"
        if not self._profiler_is_setup:
            with module._hook_context(hook_name):
                return hook(self, module, *args, **kwargs)
        with module._hook_context(hook_name), self.profiler.profile(action_name):
            return hook(self, module, *args, **kwargs)

    def _call_callback_hooks(
        self,
        module: Module,
        hook_name: str,
        *args: object,
        callback_group: Iterable[Callback] | None = None,
        **kwargs: object
    ) -> None:
        callbacks: Iterable[Callback] = (
            self.callbacks if callback_group is None else callback_group
        )
        for callback in callbacks:
            hook: Callable[..., None] = cast(
                Callable[..., None],
                getattr(callback, hook_name)
            )
            self._call_callback_hook(
                callback,
                module,
                hook_name,
                hook,
                *args,
                **kwargs
            )

    def _call_module_hook[**Parameters, ReturnT](
        self,
        module: Module,
        hook_name: str,
        hook: Callable[Parameters, ReturnT],
        *args: Parameters.args,
        **kwargs: Parameters.kwargs
    ) -> ReturnT:
        action_name: str = f"[Module]{type(module).__name__}.{hook_name}"
        if not self._profiler_is_setup:
            with module._hook_context(hook_name):
                return hook(*args, **kwargs)
        with self.profiler.profile(action_name), module._hook_context(hook_name):
            return hook(*args, **kwargs)

    def _call_datamodule_hook[**Parameters, ReturnT](
        self,
        datamodule: DataModule,
        hook_name: str,
        hook: Callable[Parameters, ReturnT],
        *args: Parameters.args,
        **kwargs: Parameters.kwargs
    ) -> ReturnT:
        action_name: str = f"[DataModule]{type(datamodule).__name__}.{hook_name}"
        if not self._profiler_is_setup:
            return hook(*args, **kwargs)
        with self.profiler.profile(action_name):
            return hook(*args, **kwargs)

    def _call_strategy_hook[**Parameters, ReturnT](
        self,
        hook_name: str,
        hook: Callable[Parameters, ReturnT],
        *args: Parameters.args,
        **kwargs: Parameters.kwargs
    ) -> ReturnT:
        action_name: str = f"[Strategy]{type(self.strategy).__name__}.{hook_name}"
        if not self._profiler_is_setup:
            return hook(*args, **kwargs)
        with self.profiler.profile(action_name):
            return hook(*args, **kwargs)

    def _configure_model(self, module: Module) -> None:
        has_legacy_override: bool = is_overridden("configure_sharded_model", module, ModelHooks)
        has_configure_override: bool = is_overridden("configure_model", module, ModelHooks)
        if has_legacy_override and has_configure_override:
            module_name: str = type(module).__name__
            raise MisconfigurationError(
                f"{module_name} overrides both configure_model() and deprecated configure_sharded_model(). "
                f"Override only configure_model()."
            )
        hook_name: str
        hook: Callable[[], None]
        if has_legacy_override:
            warnings.warn(
                "configure_sharded_model() is deprecated; override configure_model() instead.",
                DeprecationWarning,
                stacklevel=3
            )
            hook_name: str = "configure_sharded_model"
            hook: Callable[[], None] = module.configure_sharded_model
        elif has_configure_override:
            hook_name: str = "configure_model"
            hook: Callable[[], None] = module.configure_model
        else:
            return
        self._call_module_hook(module, hook_name, hook)

    def _setup_model(self, module: Module, device: torch.device) -> Module:
        self.accelerator.setup_device(device)
        with self.precision_plugin.module_init_context():
            self._configure_model(module)
        layer_sync: LayerSync | None = (
            self.strategy.layer_sync
            if isinstance(self.strategy, DDPStrategy)
            else None
        )
        if (
            self.state.stage == "fit"
            and self.strategy.is_distributed
            and layer_sync is not None
        ):
            synchronized_model: torch.nn.Module = layer_sync.apply(module)
            if not isinstance(synchronized_model, Module):
                raise TypeError(
                    f"LayerSync.apply must return Module, got {type(synchronized_model).__name__}."
                )
            module = synchronized_model
            self._layer_sync_model = synchronized_model
        converted_model: torch.nn.Module = self.precision_plugin.convert_module(module)
        if not isinstance(converted_model, Module):
            raise TypeError(
                f"Precision.convert_module must return Module, got {type(converted_model).__name__}."
            )
        resolved_module: Module = converted_model
        if self._layer_sync_model is not None:
            self._layer_sync_model = resolved_module
        resolved_module.trainer = self
        self.module = resolved_module
        self.strategy.connect(resolved_module)
        self.profiler._attach_module(resolved_module)
        self._call_strategy_hook(
            "setup",
            self.strategy.setup,
            resolved_module,
            device
        )
        resolved_module._device = device
        return resolved_module

    def _connect_precision_plugin[SchedulerT](
        self,
        module: Module,
        optimizers: list[torch.optim.Optimizer],
        lr_schedulers: list[SchedulerT]
    ) -> tuple[Module, list[torch.optim.Optimizer], list[SchedulerT]]:
        strategy_model: torch.nn.Module | None = self.strategy.model
        if strategy_model is None:
            raise RuntimeError("Strategy has no model available for precision setup.")
        precision_connection: tuple[
            torch.nn.Module,
            list[torch.optim.Optimizer],
            list[SchedulerT]
        ] = (
            self.precision_plugin.connect(
                strategy_model,
                optimizers,
                lr_schedulers
            )
        )
        connected_model: torch.nn.Module = precision_connection[0]
        connected_optimizers: list[torch.optim.Optimizer] = precision_connection[1]
        connected_lr_schedulers: list[SchedulerT] = precision_connection[2]
        if not isinstance(connected_model, torch.nn.Module):
            raise TypeError(
                f"Precision.connect must return torch.nn.Module, got {type(connected_model).__name__}."
        )
        if not isinstance(connected_optimizers, list):
            raise TypeError(
                "Precision.connect must return an optimizer list."
            )
        connected_optimizer: torch.optim.Optimizer
        for connected_optimizer in connected_optimizers:
            if not isinstance(connected_optimizer, torch.optim.Optimizer):
                raise TypeError(
                    "Precision.connect must return torch.optim.Optimizer instances."
                )
        self.strategy.model: torch.nn.Module | None = connected_model
        self.strategy.optimizers: list[torch.optim.Optimizer] = list(
            connected_optimizers
        )
        self._checkpoint_connector.connect_optimizers(self.strategy.optimizers)
        return module, connected_optimizers, connected_lr_schedulers

    def _call_on_validation_model_zero_grad(self, module: Module) -> None:
        self._call_module_hook(
            module,
            "on_validation_model_zero_grad",
            module.on_validation_model_zero_grad
        )

    def _resolve_device(self) -> torch.device:
        return self.strategy.root_device

    def _sanity_check_validation(self, module: Module) -> None:
        self.sanity_checking = True
        module._running_stage = "sanity_checking"

        try:
            self._ensure_val_dataloader()
            self._call_callback_hooks(module, "on_sanity_check_start")
            self.validate_loop.run()
            if not self._validation_schedule.is_epoch_eligible(self.current_epoch):
                self._data_connector.last_val_reload_epoch = None
            self._validation_schedule.mark_validation_end()
            self.validate_loop.epoch_metrics = {}
            module._reset_logging_state()
            self._logger_connector.clear_metric_views()
            self._reset_progress_bar_metrics("validating")
            self._call_callback_hooks(module, "on_sanity_check_end")
        finally:
            self.sanity_checking = False
            self.training = True
            module._running_stage = "training"

    def _restore_checkpoint(
        self,
        path: str | Path,
        module: Module,
        weights_only: bool | None = None,
        checkpoint: CheckpointDict | None = None,
        restore_datamodule_state: bool = True
    ) -> None:
        loaded_checkpoint: CheckpointDict = (
            self._checkpoint_connector.load(path, weights_only)
            if checkpoint is None
            else checkpoint
        )
        self._call_module_hook(
            module,
            "on_load_checkpoint",
            module.on_load_checkpoint,
            loaded_checkpoint
        )
        self._call_callback_hooks(module, "on_load_checkpoint", loaded_checkpoint)
        self.precision_plugin.on_load_checkpoint(loaded_checkpoint)
        self._checkpoint_connector.restore_strategy_state(
            loaded_checkpoint,
            strict=module.strict_loading,
            restore_optimizers=self.state.stage == "fit"
        )
        epoch: int
        global_step: int
        completed_batches: int
        total_batch_idx: int
        epoch_rng_state: CheckpointDict | None
        epoch, global_step, completed_batches, total_batch_idx, epoch_rng_state = (
            self._checkpoint_connector.restore_training_state(
                checkpoint=loaded_checkpoint,
                scheduler=(self.scheduler if self.state.stage == "fit" else None),
                schedulers=(self.schedulers if self.state.stage == "fit" else []),
                datamodule=self.datamodule,
                precision_plugin=self.precision_plugin,
                restore_datamodule_state=restore_datamodule_state
            )
        )
        combined_loader_states: list[DataLoaderState] = (
            self._checkpoint_connector.combined_loader_states(loaded_checkpoint)
            if self.state.stage == "fit"
            else []
        )
        if combined_loader_states:
            self._checkpoint_connector.train_dataloader_states: list[DataLoaderState] = (
                combined_loader_states
            )
            self._checkpoint_connector.train_dataloader_state_active: bool = True
            self._checkpoint_connector.train_dataloader_batches_completed: int = completed_batches
            self._checkpoint_connector.train_total_batch_idx: int = total_batch_idx
            if self._data_connector.train_combined_loader is not None:
                self._data_connector.train_combined_loader._load_state_dicts(
                    combined_loader_states
                )
        self.state.load_state_dict({"current_epoch": epoch, "global_step": global_step})
        module._current_epoch = epoch
        module._global_step = global_step
        if self.fit_loop.epoch_loop is not None:
            loader_state_restored: bool = bool(
                combined_loader_states
                and self._data_connector.train_combined_loader is not None
            )
            self.fit_loop.epoch_loop._combined_loader_state_restored = (
                loader_state_restored
            )
            if completed_batches > 0:
                self.fit_loop.epoch_loop._completed_batches = completed_batches
            if total_batch_idx >= 0:
                self.fit_loop.epoch_loop._total_batches_processed = total_batch_idx + 1
            if epoch_rng_state is not None and not loader_state_restored:
                self.fit_loop.epoch_loop._resume_rng_state = epoch_rng_state
            checkpoint_rng_state: object | None = loaded_checkpoint.get("rng_state")
            if (
                (completed_batches > 0 or loader_state_restored)
                and isinstance(checkpoint_rng_state, dict)
            ):
                self.fit_loop.epoch_loop._resume_checkpoint_rng_state = cast(
                    CheckpointDict,
                    checkpoint_rng_state
                )

    def _select_checkpoint_path(
        self,
        ckpt_path: str | Path | None,
        *,
        stage: TrainerStage,
        model_provided: bool,
        ckpt_path_provided: bool
    ) -> str | Path | None:
        return self._checkpoint_connector.select_path(
            ckpt_path,
            stage,
            model_provided,
            ckpt_path_provided
        )

    def _resolve_checkpoint_callback_directories(self) -> None:
        for callback in self.checkpoint_callbacks:
            if isinstance(callback, ModelCheckpoint):
                resolved_dirpath: str = callback._resolve_dirpath(self)
                dirpath: str | None = (
                    resolved_dirpath
                    if self.is_global_zero
                    else None
                )
                synchronized_dirpath: str | None = self.strategy.broadcast(dirpath)
                if synchronized_dirpath is None:
                    raise RuntimeError(
                        "ModelCheckpoint directory synchronization returned no path."
                    )
                callback.dirpath = synchronized_dirpath

    def _build_checkpoint(self, weights_only: bool = False) -> CheckpointDict:
        module: Module | None = self.module
        if module is None:
            raise AttributeError(
                "Saving a checkpoint requires a model attached through a Trainer stage."
            )
        epoch_loop: TrainingEpochLoop | None = self.fit_loop.epoch_loop
        completed_batches: int = 0
        total_batch_idx: int = -1
        epoch_rng_state: CheckpointDict | None = None
        has_entered_batch_loop: bool = False
        if self.state.stage == "fit" and epoch_loop is not None:
            completed_batches = epoch_loop._completed_batches
            total_batch_idx = epoch_loop._total_batches_processed - 1
            has_entered_batch_loop = epoch_loop._has_entered_batch_loop
            if completed_batches > 0 or has_entered_batch_loop:
                epoch_rng_state = epoch_loop._epoch_start_rng_state
        has_batch_progress: bool = completed_batches > 0 or has_entered_batch_loop
        is_validation_checkpoint: bool = self.state.running_stage == "validating"
        is_mid_epoch: bool = has_batch_progress and (
            self._is_mid_epoch_validation or not is_validation_checkpoint
        )
        combined_loader_states: list[DataLoaderState] = []
        if (
            is_mid_epoch
            and self._data_connector.train_combined_loader is not None
        ):
            combined_loader_states: list[DataLoaderState] = cast(
                list[DataLoaderState],
                epoch_loop._completed_loader_states
                if epoch_loop is not None
                and epoch_loop._use_completed_loader_states
                else self._data_connector.train_combined_loader._state_dicts()
            )
        elif self._checkpoint_connector.train_dataloader_state_active:
            combined_loader_states: list[DataLoaderState] = cast(
                list[DataLoaderState],
                self._checkpoint_connector.train_dataloader_states
            )
        checkpoint_completed_batches: int = (
            completed_batches
            if is_mid_epoch
            else self._checkpoint_connector.train_dataloader_batches_completed
            if self._checkpoint_connector.train_dataloader_state_active
            else 0
        )
        checkpoint_total_batch_idx: int = (
            total_batch_idx
            if total_batch_idx >= 0
            else self._checkpoint_connector.train_total_batch_idx
            if self._checkpoint_connector.train_dataloader_state_active
            else -1
        )
        epoch: int = self.state.current_epoch
        if (
            self.state.stage == "fit"
            and not is_mid_epoch
            and module._current_fx_name in {
                "on_train_epoch_end",
                "on_validation_epoch_end",
                "on_validation_end"
            }
        ):
            epoch: int = epoch + 1
        optimizer_states: list[CheckpointValue] = []
        if not weights_only:
            optimizer: torch.optim.Optimizer
            for optimizer in self.optimizers:
                optimizer_states.append(
                    cast(
                        CheckpointValue,
                        self._checkpoint_connector.optimizer_state(optimizer)
                    )
                )
        checkpoint: CheckpointDict = self._checkpoint_connector.build(
            model=module,
            scheduler=self.scheduler,
            schedulers=self.schedulers,
            epoch=epoch,
            global_step=self.state.global_step,
            datamodule=self.datamodule,
            completed_batches=checkpoint_completed_batches,
            total_batch_idx=checkpoint_total_batch_idx,
            epoch_rng_state=epoch_rng_state if is_mid_epoch else None,
            combined_loader_states=combined_loader_states,
            precision_plugin=self.precision_plugin,
            weights_only=weights_only,
            model_state_dict=cast(
                CheckpointValue,
                self._checkpoint_connector.model_state_dict()
            ),
            optimizer_states=optimizer_states
        )
        if not weights_only:
            self.precision_plugin.on_save_checkpoint(checkpoint)
        if not weights_only:
            self._call_callback_hooks(module, "on_save_checkpoint", checkpoint)
        self._call_module_hook(
            module,
            "on_save_checkpoint",
            module.on_save_checkpoint,
            checkpoint
        )
        return checkpoint

    def save_checkpoint(
        self,
        filepath: str | Path,
        weights_only: bool | None = None,
        storage_options: CheckpointStorageOptions = None
    ) -> None:
        if not isinstance(filepath, str | Path):
            raise TypeError("`filepath` must be a string or Path.")
        if weights_only is not None and type(weights_only) is not bool:
            raise TypeError("`weights_only` must be a bool or None.")
        if self.module is None:
            raise AttributeError(
                "Saving a checkpoint requires a model attached through a Trainer stage."
            )
        self._save_checkpoint(
            filepath,
            bool(weights_only),
            storage_options,
            synchronize=True
        )

    def _save_checkpoint(
        self,
        filepath: str | Path,
        weights_only: bool,
        storage_options: CheckpointStorageOptions,
        synchronize: bool
    ) -> None:
        profile_context: AbstractContextManager[str] | AbstractContextManager[None] = (
            self.profiler.profile("save_checkpoint")
            if self._profiler_is_setup
            else nullcontext()
        )
        with profile_context:
            checkpoint: CheckpointDict = self._build_checkpoint(weights_only)
            self._checkpoint_connector.save(
                checkpoint,
                filepath,
                storage_options,
                synchronize
            )

    def _handle_exception(
        self,
        module: Module,
        datamodule: DataModule | None,
        exception: BaseException
    ) -> Never:
        self._active_exception = exception
        self._notify_exception(module, datamodule, exception)
        raise

    def _notify_exception(
        self,
        module: Module,
        datamodule: DataModule | None,
        exception: BaseException
    ) -> None:
        self.state.interrupt()
        for index, callback in enumerate(self.callbacks):
            try:
                self._call_callback_hook(
                    callback,
                    module,
                    "on_exception",
                    callback.on_exception,
                    exception
                )
            except BaseException as hook_exception:
                self._add_exception_note(
                    exception,
                    f"callback[{index}].on_exception",
                    hook_exception
                )
        if datamodule is not None:
            try:
                self._call_datamodule_hook(
                    datamodule,
                    "on_exception",
                    datamodule.on_exception,
                    exception
                )
            except BaseException as hook_exception:
                self._add_exception_note(
                    exception,
                    "datamodule.on_exception",
                    hook_exception
                )
        try:
            self._call_strategy_hook(
                "on_exception",
                self.strategy.on_exception,
                exception
            )
        except BaseException as hook_exception:
            self._add_exception_note(exception, "strategy.on_exception", hook_exception)

    def _teardown_stage(
        self,
        module: Module,
        datamodule: DataModule | None,
        stage: TrainerStage
    ) -> BaseException | None:
        from syntheticmind.callbacks.progress_bar import ProgressBarCallback

        failures: list[_TeardownFailure] = []
        progress_callbacks: list[tuple[int, ProgressBarCallback]] = []
        self._attempt_teardown(
            "signal_connector.teardown",
            self._signal_connector.teardown,
            failures
        )
        self._attempt_teardown("strategy.teardown", self.strategy.teardown, failures)
        self._attempt_teardown(
            "checkpoint.teardown",
            self._checkpoint_connector.teardown,
            failures
        )
        self._attempt_teardown(
            "layer_sync.revert",
            self._revert_layer_sync,
            failures
        )
        self._attempt_teardown(
            "precision_plugin.teardown",
            self.precision_plugin.teardown,
            failures
        )
        if isinstance(self.strategy, ParallelStrategy):
            self._attempt_teardown(
                "cluster_environment.teardown",
                self.strategy.cluster_environment.teardown,
                failures
            )
        combined_loaders: tuple[tuple[str, CombinedLoader | None], ...] = (
            (
                ("train", self._data_connector.train_combined_loader),
                ("val", self._data_connector.val_combined_loader)
            )
            if stage == "fit"
            else (
                ("val", self._data_connector.val_combined_loader),
            )
            if stage == "validate"
            else (
                (
                    stage,
                    self._data_connector.test_combined_loader
                    if stage == "test"
                    else self._data_connector.predict_combined_loader
                ),
            )
        )
        for loader_stage, combined_loader in combined_loaders:
            if combined_loader is not None:
                self._attempt_teardown(
                    f"{loader_stage}_combined_loader.reset",
                    combined_loader.reset,
                    failures
                )
        if datamodule is not None:
            self._attempt_teardown(
                "datamodule.teardown",
                lambda: self._call_datamodule_hook(
                    datamodule,
                    "teardown",
                    datamodule.teardown,
                    stage
                ),
                failures
            )
        for index, callback in enumerate(self.callbacks):
            if isinstance(callback, ProgressBarCallback):
                progress_callbacks.append((index, callback))
                continue
            self._attempt_teardown(
                f"callback[{index}].teardown",
                lambda callback=callback: self._call_callback_teardown(
                    callback,
                    module,
                    stage
                ),
                failures
            )
        self._attempt_teardown(
            "module.teardown",
            lambda: self._call_module_teardown(module, stage),
            failures
        )
        for index, configured_logger in enumerate(self.loggers):
            self._attempt_teardown(
                f"logger[{index}].finalize",
                configured_logger.finalize,
                failures
            )
        for index, callback in progress_callbacks:
            self._attempt_teardown(
                f"callback[{index}].teardown",
                lambda callback=callback: self._call_callback_teardown(
                    callback,
                    module,
                    stage
                ),
                failures
            )
        self._attempt_teardown(
            "profiler.describe",
            self._teardown_profiler,
            failures
        )

        exception_notified: bool = False
        if failures and self._active_exception is None:
            self._notify_exception(module, datamodule, failures[0][1])
            exception_notified = True

        if not failures:
            return None
        if self._active_exception is not None:
            for label, cleanup_exception in failures:
                self._add_exception_note(self._active_exception, label, cleanup_exception)
            return None

        primary_exception: BaseException = failures[0][1]
        for label, cleanup_exception in failures[1:]:
            self._add_exception_note(primary_exception, label, cleanup_exception)
        if not exception_notified:
            self._notify_exception(module, datamodule, primary_exception)
        return primary_exception

    def _revert_layer_sync(self) -> None:
        layer_sync: LayerSync | None = (
            self.strategy.layer_sync
            if isinstance(self.strategy, DDPStrategy)
            else None
        )
        synchronized_model: Module | None = self._layer_sync_model
        if synchronized_model is None or layer_sync is None:
            return
        reverted_model: torch.nn.Module = layer_sync.revert(synchronized_model)
        if not isinstance(reverted_model, Module):
            raise TypeError(
                f"LayerSync.revert must return Module, got {type(reverted_model).__name__}."
            )
        reverted_model.trainer = self
        self.module = reverted_model
        self.strategy.model = reverted_model
        self._layer_sync_model = None

    def _attempt_teardown(
        self,
        label: str,
        action: Callable[[], None],
        failures: list[_TeardownFailure]
    ) -> None:
        try:
            action()
        except BaseException as exception:
            failures.append((label, exception))

    def _call_callback_teardown(
        self,
        callback: Callback,
        module: Module,
        stage: TrainerStage
    ) -> None:
        self._call_callback_hook(
            callback,
            module,
            "teardown",
            callback.teardown,
            stage
        )

    def _call_module_teardown(self, module: Module, stage: TrainerStage) -> None:
        self._call_module_hook(
            module,
            "teardown",
            module.teardown,
            stage
        )

    def _add_exception_note(
        self,
        exception: BaseException,
        label: str,
        secondary_exception: BaseException
    ) -> None:
        detail: str = " ".join(str(secondary_exception).split())
        description: str = type(secondary_exception).__name__
        if detail:
            description = f"{description}: {detail}"
        exception.add_note(f"{label} raised while handling the stage failure: {description}")

    def _resolve_total_scheduler_steps(
        self,
        scheduler_definitions: list[LRSchedulerConfig]
    ) -> int:
        config_schedulers: list[LRSchedulerConfig] = list(scheduler_definitions)
        if not config_schedulers:
            raise MisconfigurationError(
                "Expected at least one scheduler configuration when resolving scheduler steps."
            )
        if any(scheduler_config.interval == "step" for scheduler_config in config_schedulers):
            assert self.fit_loop.epoch_loop is not None
            limited_batches: int | None = self._data_connector.total_train_batches
            if limited_batches is None:
                raise MisconfigurationError(
                    "Cannot determine total training steps for the learning rate scheduler. "
                    "With iterable datasets, set limit_train_batches to an integer value."
                )
            accum: int = self.fit_loop.epoch_loop.accumulate_grad_batches
            stepping_per_epoch: int = max(1, math.ceil(limited_batches / accum))
            if self.max_epochs == -1:
                if self.max_steps == -1:
                    raise MisconfigurationError(
                        "Cannot determine total training steps without an epoch or step limit."
                    )
                return self.max_steps
            total_steps: int = stepping_per_epoch * self.max_epochs
            return min(total_steps, self.max_steps) if self.max_steps != -1 else total_steps
        if self.max_epochs == -1:
            if self.max_steps == -1:
                raise MisconfigurationError(
                    "Cannot determine an epoch scheduler horizon without an epoch or step limit."
                )
            return self.max_steps
        return self.max_epochs

    def _metric_to_float(self, value: torch.Tensor | float) -> float:
        return self._logger_connector.metric_to_float(value)

    def _log_metrics(
        self,
        metrics: Mapping[str, torch.Tensor | float],
        step: int
    ) -> None:
        self._logger_connector.log_metrics(metrics, step)

    def _log_hyperparams(self, params: HyperparameterDict) -> None:
        self._logger_connector.log_hyperparams(params)

    def _autolog_hyperparameters(
        self,
        module: Module,
        datamodule: DataModule | None
    ) -> None:
        self._logger_connector.autolog_hyperparameters(module, datamodule)

    @property
    def callbacks(self) -> list[Callback]:
        return self._callback_connector.callbacks

    @callbacks.setter
    def callbacks(self, callbacks: list[Callback]) -> None:
        self._callback_connector.callbacks = callbacks

    @property
    def checkpoint_callback(self) -> Checkpoint | None:
        callbacks: list[Checkpoint] = self.checkpoint_callbacks
        return callbacks[0] if callbacks else None

    @property
    def checkpoint_callbacks(self) -> list[Checkpoint]:
        return [
            callback
            for callback in self.callbacks
            if isinstance(callback, Checkpoint)
        ]

    @property
    def early_stopping_callback(self) -> EarlyStopping | None:
        callbacks: list[EarlyStopping] = self.early_stopping_callbacks
        return callbacks[0] if callbacks else None

    @property
    def early_stopping_callbacks(self) -> list[EarlyStopping]:
        return [
            callback
            for callback in self.callbacks
            if isinstance(callback, EarlyStopping)
        ]

    @property
    def progress_bar_callback(self) -> ProgressBarCallback | None:
        return next(
            (
                callback
                for callback in self.callbacks
                if isinstance(callback, ProgressBarCallback)
            ),
            None
        )

    @property
    def ckpt_path(self) -> str | Path | None:
        return self._checkpoint_connector.path

    @ckpt_path.setter
    def ckpt_path(self, ckpt_path: str | Path | None) -> None:
        self._checkpoint_connector.path = ckpt_path

    @property
    def logger(self) -> Logger | None:
        return self._logger_connector.logger

    @logger.setter
    def logger(self, logger: Logger | None) -> None:
        self._logger_connector.logger = logger

    @property
    def loggers(self) -> list[Logger]:
        return self._logger_connector.loggers

    @loggers.setter
    def loggers(self, loggers: list[Logger] | None) -> None:
        self._logger_connector.loggers = loggers

    @property
    def default_root_dir(self) -> str:
        return normalize_path(self._default_root_dir)

    @property
    def log_dir(self) -> str:
        return self._logger_connector.log_dir

    @property
    def current_epoch(self) -> int:
        return self.state.current_epoch

    @property
    def global_rank(self) -> int:
        return self.strategy.global_rank

    @property
    def local_rank(self) -> int:
        return self.strategy.local_rank

    @property
    def node_rank(self) -> int:
        return self.strategy.node_rank

    @property
    def world_size(self) -> int:
        return self.strategy.world_size

    @property
    def num_nodes(self) -> int:
        return self.strategy.num_nodes

    @property
    def device_ids(self) -> list[int]:
        return [
            device.index if device.index is not None else index
            for index, device in enumerate(self.parallel_devices)
        ]

    @property
    def num_devices(self) -> int:
        return len(self.parallel_devices)

    @property
    def reload_dataloaders_every_n_epochs(self) -> int:
        return self._data_connector.reload_interval

    @property
    def precision_plugin(self) -> Precision:
        return self._precision_plugin

    @property
    def precision(self) -> PrecisionType:
        return self._precision_plugin.precision

    @property
    def scaler(self) -> torch.GradScaler | None:
        return self._precision_plugin.scaler

    @property
    def is_global_zero(self) -> bool:
        return self.strategy.is_global_zero

    @property
    def model(self) -> torch.nn.Module | None:
        return self.strategy.model

    @property
    def distributed_sampler_kwargs(self) -> dict[str, int] | None:
        return self.strategy.distributed_sampler_kwargs

    def print(
        self,
        *values: object,
        sep: str | None = " ",
        end: str | None = "\n",
        file: IO[str] | None = None,
        flush: bool = False
    ) -> None:
        if self.local_rank == 0:
            print(*values, sep=sep, end=end, file=file, flush=flush)

    @property
    def max_epochs(self) -> int:
        return self.fit_loop.max_epochs

    @property
    def min_epochs(self) -> int:
        return self.fit_loop.min_epochs

    @property
    def max_steps(self) -> int:
        assert self.fit_loop.epoch_loop is not None
        return self.fit_loop.epoch_loop.max_steps

    @property
    def min_steps(self) -> int | None:
        assert self.fit_loop.epoch_loop is not None
        return self.fit_loop.epoch_loop.min_steps

    @property
    def global_step(self) -> int:
        return self.state.global_step

    @property
    def callback_metrics(self) -> dict[str, torch.Tensor]:
        return dict(self._logger_connector.callback_metrics)

    @property
    def logged_metrics(self) -> dict[str, torch.Tensor]:
        return dict(self._logger_connector.logged_metrics)

    @property
    def progress_bar_metrics(self) -> dict[str, float]:
        running_stage: RunningStage | None = self.state.running_stage
        if running_stage == "sanity_checking":
            return {}
        resolved_stage: RunningStage | None = (
            running_stage if running_stage is not None else self._logger_connector.last_progress_bar_stage
        )
        if resolved_stage is None:
            return {}
        return self._get_progress_bar_metrics(resolved_stage)

    @property
    def interrupted(self) -> bool:
        return self.state.status == "interrupted"

    @property
    def received_sigterm(self) -> bool:
        return self._signal_connector.received_sigterm

    @property
    def is_last_batch(self) -> bool:
        if self.fit_loop.epoch_loop is None:
            return False
        return self.fit_loop.epoch_loop.is_last_batch

    def _raise_if_sigterm(self, synchronize: bool = True) -> None:
        if not synchronize and self.strategy.is_distributed:
            return
        received_sigterm: bool = (
            self.strategy.reduce_boolean_decision(self.received_sigterm)
            if synchronize
            else self.received_sigterm
        )
        if received_sigterm:
            self._signal_connector.received_sigterm = True
            raise SIGTERMException()

    @property
    def training(self) -> bool:
        return self.state.running_stage == "training"

    @training.setter
    def training(self, value: bool) -> None:
        if type(value) is not bool:
            raise TypeError("`training` must be a bool.")
        if value:
            self.state.set_running_stage("training")
        elif self.training:
            self.state.set_running_stage(None)

    @property
    def validating(self) -> bool:
        return self.state.running_stage == "validating"

    @validating.setter
    def validating(self, value: bool) -> None:
        if type(value) is not bool:
            raise TypeError("`validating` must be a bool.")
        if value:
            self.state.set_running_stage("validating")
        elif self.validating:
            self.state.set_running_stage(None)

    @property
    def testing(self) -> bool:
        return self.state.running_stage == "testing"

    @testing.setter
    def testing(self, value: bool) -> None:
        if type(value) is not bool:
            raise TypeError("`testing` must be a bool.")
        if value:
            self.state.set_running_stage("testing")
        elif self.testing:
            self.state.set_running_stage(None)

    @property
    def predicting(self) -> bool:
        return self.state.running_stage == "predicting"

    @predicting.setter
    def predicting(self, value: bool) -> None:
        if type(value) is not bool:
            raise TypeError("`predicting` must be a bool.")
        if value:
            self.state.set_running_stage("predicting")
        elif self.predicting:
            self.state.set_running_stage(None)

    @property
    def evaluating(self) -> bool:
        return self.validating or self.testing or self.sanity_checking

    @property
    def sanity_checking(self) -> bool:
        return self.state.running_stage == "sanity_checking"

    @sanity_checking.setter
    def sanity_checking(self, value: bool) -> None:
        if type(value) is not bool:
            raise TypeError("`sanity_checking` must be a bool.")
        if value:
            self.state.set_running_stage("sanity_checking")
        elif self.sanity_checking:
            self.state.set_running_stage(None)

    @property
    def should_stop(self) -> bool:
        return self.state.should_stop

    @should_stop.setter
    def should_stop(self, value: bool) -> None:
        self.state.should_stop = value

    @property
    def enable_validation(self) -> bool:
        return self._validation_schedule.enabled

    @property
    def train_dataloader(self) -> object | None:
        if self._data_connector.train_combined_loader is None:
            return None
        return self._data_connector.train_combined_loader.iterables

    @property
    def val_dataloaders(self) -> object | None:
        if self._data_connector.val_combined_loader is None:
            return None
        return self._data_connector.val_combined_loader.iterables

    @property
    def test_dataloaders(self) -> object | None:
        if self._data_connector.test_combined_loader is None:
            return None
        return self._data_connector.test_combined_loader.iterables

    @property
    def predict_dataloaders(self) -> object | None:
        if self._data_connector.predict_combined_loader is None:
            return None
        return self._data_connector.predict_combined_loader.iterables

    @property
    def num_training_batches(self) -> int | float:
        return (
            float("inf")
            if self._data_connector.total_train_batches is None
            else self._data_connector.total_train_batches
        )

    @property
    def num_val_batches(self) -> list[int | float]:
        if self._data_connector.total_val_batches is None:
            return []
        resolved_batches: list[int | float] = []
        for total_batches in self._data_connector.total_val_batches:
            effective_batches: int | None = resolve_batch_limit(
                total_batches,
                self.limit_val_batches,
                "val"
            )
            resolved_batches.append(
                float("inf")
                if effective_batches is None
                else effective_batches
            )
        return resolved_batches

    @property
    def num_sanity_val_batches(self) -> list[int | float]:
        return [
            min(self.num_sanity_val_steps, batches)
            for batches in self.num_val_batches
        ]

    @property
    def num_test_batches(self) -> list[int | float]:
        if self._data_connector.total_test_batches is None:
            return []
        resolved_batches: list[int | float] = []
        for total_batches in self._data_connector.total_test_batches:
            effective_batches: int | None = resolve_batch_limit(
                total_batches,
                self.limit_test_batches,
                "test"
            )
            resolved_batches.append(
                float("inf")
                if effective_batches is None
                else effective_batches
            )
        return resolved_batches

    @property
    def num_predict_batches(self) -> list[int | float]:
        if self._data_connector.total_predict_batches is None:
            return []
        resolved_batches: list[int | float] = []
        for total_batches in self._data_connector.total_predict_batches:
            effective_batches: int | None = resolve_batch_limit(
                total_batches,
                self.limit_predict_batches,
                "predict"
            )
            resolved_batches.append(
                float("inf")
                if effective_batches is None
                else effective_batches
            )
        return resolved_batches

    @property
    def estimated_stepping_batches(self) -> int | float:
        if self.fit_loop.epoch_loop is None:
            return float("inf")
        limited_batches: int | None = self._data_connector.total_train_batches
        if limited_batches is None:
            return float("inf") if self.max_steps == -1 else self.max_steps
        accum: int = self.fit_loop.epoch_loop.accumulate_grad_batches
        stepping_per_epoch: int = math.ceil(limited_batches / accum)
        if self.max_epochs == -1:
            return float("inf") if self.max_steps == -1 else self.max_steps
        estimated_steps: int = stepping_per_epoch * self.max_epochs
        return min(estimated_steps, self.max_steps) if self.max_steps != -1 else estimated_steps

    def __repr__(self) -> str:
        return (
            f"Trainer("
            f"max_epochs={self.max_epochs}, "
            f"max_steps={self.max_steps}, "
            f"accelerator={type(self.accelerator).__name__}, "
            f"strategy={type(self.strategy).__name__}, "
            f"devices={self.devices}, "
            f"precision={self.precision!r}, "
            f"callbacks={[type(callback).__name__ for callback in self.callbacks]})"
        )
