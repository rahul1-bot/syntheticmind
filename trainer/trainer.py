
import inspect
import math
import os
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import IO, Literal, Never, cast, override

import torch
from loguru import logger as log
from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.accelerators.cpu import CPUAccelerator
from syntheticmind.callbacks.callback import Callback
from syntheticmind.callbacks.checkpointing import Checkpoint, ModelCheckpoint
from syntheticmind.callbacks.early_stopping import EarlyStopping
from syntheticmind.callbacks.progress_bar import ProgressBarCallback
from syntheticmind.core.datamodule import DataModule
from syntheticmind.core.hooks import DataHooks, ModelHooks
from syntheticmind.core.module import Module
from syntheticmind.core.optimizer import (
    LRSchedulerConfig,
    OptimizerLRScheduler,
    SyntheticOptimizer,
    _OptimizerConfigurationParser,
    _SchedulerInstance
)
from syntheticmind.loggers.logger import Logger
from syntheticmind.loops.evaluation_loop import EvaluationLoop
from syntheticmind.loops.fit_loop import FitLoop
from syntheticmind.loops.prediction_loop import PredictionLoop
from syntheticmind.loops.training_epoch_loop import TrainingEpochLoop
from syntheticmind.loops.validation_schedule import ValidationSchedule
from syntheticmind.plugins.precision.precision import Precision
from syntheticmind.plugins import Plugin
from syntheticmind.plugins.checkpoint_io import CheckpointStorageOptions
from syntheticmind.plugins.layer_sync import LayerSync
from syntheticmind.profilers import (
    AdvancedProfiler,
    PassThroughProfiler,
    Profiler,
    PyTorchProfiler,
    SimpleProfiler
)
from syntheticmind.randomness import _Randomness
from syntheticmind.trainer.states import TrainerState
from syntheticmind.strategies.ddp import DDPStrategy
from syntheticmind.strategies.launchers.launcher import _ProcessLauncher
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
from syntheticmind.trainer.configuration import (
    _TrainerConfiguration,
    _TrainerConfigurationSnapshot
)
from syntheticmind.trainer.console import ConsoleRenderer
from syntheticmind.trainer.hook_dispatcher import _HookDispatcher
from syntheticmind.combined_loader import CombinedLoader
from syntheticmind.exceptions import MisconfigurationError, SIGTERMException
from syntheticmind.storage.filesystem import StoragePath
from syntheticmind.types import (
    Batch,
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
    TrainerStage
)

__all__: list[str] = ["Trainer"]

type _TeardownFailure = tuple[str, BaseException]
type _CheckpointLoopState = tuple[
    int,
    int,
    CheckpointDict | None,
    list[DataLoaderState],
    bool
]
type _RestoredTrainingState = tuple[
    int,
    int,
    int,
    int,
    CheckpointDict | None
]
type _PrintValue = (
    Batch
    | ModelOutput
    | Path
    | BaseException
    | torch.dtype
    | torch.device
    | torch.nn.Module
    | torch.optim.Optimizer
)


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
        seed: int | None = None
    ) -> None:
        configuration: _TrainerConfiguration = _TrainerConfiguration(
            fast_dev_run=fast_dev_run,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            max_steps=max_steps,
            min_steps=min_steps,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            limit_predict_batches=limit_predict_batches,
            overfit_batches=overfit_batches,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            num_sanity_val_steps=num_sanity_val_steps,
            log_every_n_steps=log_every_n_steps,
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            deterministic=deterministic,
            benchmark=benchmark,
            inference_mode=inference_mode,
            use_distributed_sampler=use_distributed_sampler,
            detect_anomaly=detect_anomaly,
            barebones=barebones,
            sync_batchnorm=sync_batchnorm,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            enable_autolog_hparams=enable_autolog_hparams
        )
        self._initialize_runtime(
            configuration=configuration,
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            plugins=plugins,
            default_root_dir=default_root_dir,
            seed=seed,
            logger=logger,
            callbacks=callbacks,
            max_time=max_time,
            profiler=profiler
        )

    def _initialize_runtime(
        self,
        configuration: _TrainerConfiguration,
        accelerator: str | Accelerator,
        strategy: str | Strategy,
        devices: int | str | list[int],
        num_nodes: int,
        precision: PrecisionInput | None,
        plugins: Plugin | list[Plugin] | None,
        default_root_dir: str | Path | None,
        seed: int | None,
        logger: bool | Logger | Iterable[Logger] | None,
        callbacks: Callback | list[Callback] | None,
        max_time: str | timedelta | dict[str, int] | None,
        profiler: Profiler | str | None
    ) -> None:
        if configuration.barebones and logger is not None and logger is not False:
            raise MisconfigurationError(
                "Trainer barebones mode does not permit experiment loggers."
            )
        if configuration.barebones and profiler is not None:
            raise MisconfigurationError(
                "Trainer barebones mode does not permit profiler instrumentation."
            )
        self._configuration: _TrainerConfiguration = configuration
        self._initialize_accelerator(
            configuration,
            accelerator,
            strategy,
            devices,
            num_nodes,
            precision,
            plugins
        )
        self._apply_configuration(configuration, default_root_dir, seed)
        self._initialize_connectors(
            configuration,
            logger,
            callbacks,
            max_time,
            profiler
        )
        self._initialize_loops(configuration)
        self._initialize_runtime_state()

    def _initialize_accelerator(
        self,
        configuration: _TrainerConfiguration,
        accelerator: str | Accelerator,
        strategy: str | Strategy,
        devices: int | str | list[int],
        num_nodes: int,
        precision: PrecisionInput | None,
        plugins: Plugin | list[Plugin] | None
    ) -> None:
        self._accelerator_connector: _AcceleratorConnector = _AcceleratorConnector(
            accelerator,
            strategy,
            devices,
            num_nodes,
            precision,
            plugins,
            configuration.sync_batchnorm
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
        self._randomness: _Randomness = _Randomness(
            [device.type for device in self.parallel_devices]
        )

    def _apply_configuration(
        self,
        configuration: _TrainerConfiguration,
        default_root_dir: str | Path | None,
        seed: int | None
    ) -> None:
        if default_root_dir is None:
            self._default_root_dir: str = os.getcwd()
        else:
            configured_root_value: str | bytes = os.fspath(default_root_dir)
            if not isinstance(configured_root_value, str):
                raise TypeError("`default_root_dir` must resolve to a string path.")
            configured_root_dir: str = configured_root_value
            self._default_root_dir: str = configured_root_dir or os.getcwd()
        self.limit_train_batches: int | float = (
            configuration.resolved_limit_train_batches
        )
        self.limit_val_batches: int | float = (
            configuration.resolved_limit_val_batches
        )
        self.limit_test_batches: int | float = (
            configuration.resolved_limit_test_batches
        )
        self.limit_predict_batches: int | float = (
            configuration.resolved_limit_predict_batches
        )
        self.gradient_clip_val: int | float | None = configuration.gradient_clip_val
        self.gradient_clip_algorithm: ClipAlgorithm | None = (
            configuration.gradient_clip_algorithm
        )
        self.log_every_n_steps: int = configuration.resolved_log_every_n_steps
        self.enable_progress_bar: bool = configuration.resolved_enable_progress_bar
        self.enable_model_summary: bool = configuration.resolved_enable_model_summary
        self.enable_autolog_hparams: bool = configuration.enable_autolog_hparams
        self.barebones: bool = configuration.barebones
        self.num_sanity_val_steps: int | float = (
            configuration.resolved_num_sanity_val_steps
        )
        self.fast_dev_run: int | bool = configuration.fast_dev_run
        self.overfit_batches: int | float = configuration.overfit_batches
        self.use_distributed_sampler: bool = configuration.use_distributed_sampler
        self.sync_batchnorm: bool = configuration.sync_batchnorm
        self._configure_torch_flags(
            configuration.deterministic,
            configuration.benchmark
        )

        self.deterministic: bool | Literal["warn"] | None = (
            configuration.deterministic
        )
        self.inference_mode: bool = configuration.inference_mode
        self._detect_anomaly: bool = configuration.detect_anomaly

        if seed is not None:
            self._randomness.seed(seed)

    def _initialize_connectors(
        self,
        configuration: _TrainerConfiguration,
        logger: bool | Logger | Iterable[Logger] | None,
        callbacks: Callback | list[Callback] | None,
        max_time: str | timedelta | dict[str, int] | None,
        profiler: Profiler | str | None
    ) -> None:
        self._callback_connector: _CallbackConnector = _CallbackConnector(
            callbacks,
            configuration.resolved_enable_checkpointing,
            configuration.resolved_enable_progress_bar,
            configuration.resolved_enable_model_summary,
            max_time
        )
        self._checkpoint_connector: _CheckpointConnector = _CheckpointConnector(
            self.strategy,
            self._callback_connector,
            configuration.fast_dev_run_enabled,
            self._randomness
        )

        configured_logger: bool | Logger | Iterable[Logger] | None = (
            False if configuration.barebones else logger
        )
        self._logger_connector: _LoggerConnector = _LoggerConnector(
            self.strategy,
            configured_logger,
            self.default_root_dir,
            configuration.fast_dev_run_enabled,
            configuration.enable_autolog_hparams
        )
        self.profiler: Profiler = self._resolve_profiler(profiler)
        self._hook_dispatcher: _HookDispatcher[Trainer] = _HookDispatcher(
            self,
            self._callback_connector,
            self.profiler,
            self.strategy
        )
        self._data_connector: _DataConnector = _DataConnector(
            self.strategy,
            self.profiler,
            self.limit_train_batches,
            self.limit_val_batches,
            self.limit_test_batches,
            self.limit_predict_batches,
            self.overfit_batches,
            self.use_distributed_sampler,
            configuration.reload_dataloaders_every_n_epochs
        )

    def _initialize_loops(self, configuration: _TrainerConfiguration) -> None:
        self.state: TrainerState = TrainerState()
        self._signal_connector: _SignalConnector = _SignalConnector(self.strategy)
        self._is_mid_epoch_validation: bool = False
        self._validation_schedule: ValidationSchedule = ValidationSchedule(
            configuration.resolved_val_check_interval,
            configuration.resolved_check_val_every_n_epoch
        )
        self.val_check_interval: int | float | str | timedelta | dict[str, int] = (
            self._validation_schedule.val_check_interval
        )
        self.check_val_every_n_epoch: int | None = (
            self._validation_schedule.check_val_every_n_epoch
        )
        self.val_check_batch: int | None = None

        self.fit_loop: FitLoop = FitLoop(
            min_epochs=configuration.resolved_min_epochs,
            max_epochs=configuration.resolved_max_epochs
        )
        self.fit_loop.epoch_loop: TrainingEpochLoop | None = TrainingEpochLoop(
            min_steps=configuration.min_steps,
            max_steps=configuration.resolved_max_steps,
            accumulate_grad_batches=configuration.accumulate_grad_batches,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm=self.gradient_clip_algorithm,
            log_every_n_steps=configuration.resolved_log_every_n_steps
        )
        self.validate_loop: EvaluationLoop = EvaluationLoop(
            stage="validate",
            inference_mode=configuration.inference_mode
        )
        self.test_loop: EvaluationLoop = EvaluationLoop(
            stage="test",
            inference_mode=configuration.inference_mode
        )
        self.predict_loop: PredictionLoop = PredictionLoop(
            inference_mode=configuration.inference_mode
        )

        self._connect_loops()

    def _initialize_runtime_state(self) -> None:
        self.module: Module | None = None
        self.datamodule: DataModule | None = None
        self.optimizers: list[torch.optim.Optimizer] = []
        self._synthetic_optimizers: list[SyntheticOptimizer] = []
        self.schedulers: list[_SchedulerInstance] = []
        self.lr_scheduler_configs: list[LRSchedulerConfig] = []
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: _SchedulerInstance | None = None
        self.scheduler_config: LRSchedulerConfig | None = None
        self._active_exception: BaseException | None = None
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
        resolved_datamodule: DataModule | None = (
            train_dataloaders
            if isinstance(train_dataloaders, DataModule)
            else datamodule
        )
        resolved_train_dataloaders: DataLoaderSource | CombinedLoader | None = (
            None
            if isinstance(train_dataloaders, DataModule)
            else train_dataloaders
        )
        if (
            resolved_datamodule is not None
            and (
                resolved_train_dataloaders is not None
                or val_dataloaders is not None
            )
        ):
            raise MisconfigurationError(
                "You cannot pass `train_dataloaders` or `val_dataloaders` "
                "to `Trainer.fit(datamodule=...)`."
            )
        if weights_only is not None and type(weights_only) is not bool:
            raise TypeError("`weights_only` must be a bool or None.")
        resume_path: str | Path | None = self._select_checkpoint_path(
            ckpt_path,
            stage="fit",
            model_provided=True,
            ckpt_path_provided=ckpt_path is not None
        )
        self._attach_model_callbacks(module, "fit")
        launcher: _ProcessLauncher | None = self._stage_launcher()
        if launcher is not None and launcher.should_launch:
            try:
                launcher.launch(
                    "fit",
                    self,
                    module,
                    resolved_datamodule,
                    {
                        "train_dataloaders": resolved_train_dataloaders,
                        "val_dataloaders": val_dataloaders,
                        "ckpt_path": resume_path,
                        "weights_only": weights_only
                    }
                )
            finally:
                self._clear_model_callback_configuration("fit")
                self.state.finish()
            return
        self._fit_impl(
            module,
            resolved_train_dataloaders,
            val_dataloaders,
            resolved_datamodule,
            resume_path,
            weights_only
        )

    def _fit_impl(
        self,
        module: Module,
        train_dataloaders: DataLoaderSource | CombinedLoader | None,
        val_dataloaders: DataLoaderSource | CombinedLoader | None,
        datamodule: DataModule | None,
        resume_path: str | Path | None,
        weights_only: bool | None
    ) -> None:
        active_module: Module = module
        self._begin_fit_stage(
            active_module,
            datamodule,
            train_dataloaders,
            val_dataloaders
        )
        training_modes: dict[str, bool] = active_module._capture_training_modes()
        fit_completed: bool = False
        try:
            fit_preparation: tuple[Module, CheckpointDict | None] = (
                self._prepare_fit_stage(
                    active_module,
                    datamodule,
                    resume_path,
                    weights_only
                )
            )
            active_module: Module = fit_preparation[0]
            preloaded_checkpoint: CheckpointDict | None = fit_preparation[1]
            self._configure_fit_optimization(active_module)
            self._restore_fit_checkpoint(
                active_module,
                resume_path,
                weights_only,
                preloaded_checkpoint
            )
            self._start_fit_stage(active_module, datamodule, resume_path)
            self._run_fit_stage(active_module)
            fit_completed: bool = True
        except BaseException as exception:
            attached_module: Module | None = self.module
            if attached_module is not None:
                active_module: Module = attached_module
            self._handle_exception(active_module, datamodule, exception)
        finally:
            self._finish_stage(
                active_module,
                datamodule,
                "fit",
                fit_completed,
                training_modes
            )

    def _begin_fit_stage(
        self,
        module: Module,
        datamodule: DataModule | None,
        train_dataloaders: DataLoaderSource | CombinedLoader | None,
        val_dataloaders: DataLoaderSource | CombinedLoader | None
    ) -> None:
        self.state.set_stage("fit")
        self._active_exception: BaseException | None = None
        self.state.reset_on_run()
        self._is_mid_epoch_validation: bool = False
        self._logger_connector.clear_metric_views()
        module._reset_logging_state()
        self._reset_progress_bar_metrics()
        self._data_connector.reset_fit()
        self._checkpoint_connector.reset_training_continuation()
        self.fit_loop.should_stop: bool = False
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
        module._current_epoch: int = 0
        module._global_step: int = 0

    def _prepare_fit_stage(
        self,
        module: Module,
        datamodule: DataModule | None,
        resume_path: str | Path | None,
        weights_only: bool | None
    ) -> tuple[Module, CheckpointDict | None]:
        self._signal_connector.register_signal_handlers()
        self.strategy.setup_environment()
        self._announce_barebones_stage("fit")
        self._setup_profiler("fit", module)
        self._prepare_data_and_setup(module, datamodule, "fit")
        preloaded_checkpoint: CheckpointDict | None = self._preload_fit_checkpoint(
            module,
            datamodule,
            resume_path,
            weights_only
        )
        if (
            self.limit_train_batches != 0
            and self._hook_dispatcher.is_overridden("training_step", module, Module)
        ):
            self._request_dataloaders("train")
        else:
            self._data_connector.total_train_batches: int | None = 0
        self._configure_validation_schedule(reset_time=True)
        self._warn_about_dataloader_workers()
        device: torch.device = self._resolve_device()
        configured_module: Module = self._setup_model(module, device)
        return configured_module, preloaded_checkpoint

    def _preload_fit_checkpoint(
        self,
        module: Module,
        datamodule: DataModule | None,
        resume_path: str | Path | None,
        weights_only: bool | None
    ) -> CheckpointDict | None:
        if resume_path is None:
            return None
        checkpoint: CheckpointDict = self._checkpoint_connector.load_metadata(
            resume_path,
            weights_only
        )
        resume_epoch: int = self._checkpoint_connector.integer_value(
            checkpoint.get("epoch"),
            "epoch",
            0
        )
        resume_global_step: int = self._checkpoint_connector.integer_value(
            checkpoint.get("global_step"),
            "global_step",
            0
        )
        self.state.load_state_dict(
            {
                "current_epoch": resume_epoch,
                "global_step": resume_global_step
            }
        )
        module._current_epoch: int = resume_epoch
        module._global_step: int = resume_global_step
        if datamodule is not None:
            self._checkpoint_connector.load_datamodule_state(checkpoint, datamodule)
        return checkpoint

    def _warn_about_dataloader_workers(self) -> None:
        if not self.is_global_zero:
            return
        combined_loader: CombinedLoader | None = (
            self._data_connector.train_combined_loader
        )
        if combined_loader is None:
            return
        for train_dataloader in combined_loader.flattened:
            if (
                isinstance(train_dataloader, torch.utils.data.DataLoader)
                and train_dataloader.num_workers == 0
                and not isinstance(self.accelerator, CPUAccelerator)
            ):
                log.warning(
                    "num_workers=0 with GPU accelerator. Consider increasing "
                    "for better throughput."
                )
                return

    def _configure_fit_optimization(self, module: Module) -> None:
        training_step_signature: inspect.Signature = inspect.signature(
            module.training_step
        )
        if "optimizer_idx" in training_step_signature.parameters:
            raise RuntimeError(
                "Training with multiple optimizers is only supported with manual "
                "optimization. Remove the optimizer_idx parameter from training_step, "
                "set automatic_optimization = False, and access optimizers through "
                "self.optimizers()."
            )
        optimization_configuration: OptimizerLRScheduler = self._hook_dispatcher.call_module(
            module,
            "configure_optimizers",
            module.configure_optimizers
        )
        optimizer_parser: _OptimizerConfigurationParser = (
            _OptimizerConfigurationParser(
                module.automatic_optimization,
                self._hook_dispatcher.is_overridden("lr_scheduler_step", module, Module)
            )
        )
        optimizer_initialization: tuple[
            list[torch.optim.Optimizer],
            list[LRSchedulerConfig]
        ] = optimizer_parser.resolve(optimization_configuration)
        self.optimizers: list[torch.optim.Optimizer] = optimizer_initialization[0]
        self.lr_scheduler_configs: list[LRSchedulerConfig] = (
            optimizer_initialization[1]
        )
        precision_connection: tuple[
            Module,
            list[torch.optim.Optimizer],
            list[LRSchedulerConfig]
        ] = (
            self._connect_precision_plugin(
                module,
                self.optimizers,
                self.lr_scheduler_configs
            )
        )
        self.optimizers: list[torch.optim.Optimizer] = precision_connection[1]
        self.lr_scheduler_configs: list[LRSchedulerConfig] = (
            precision_connection[2]
        )
        self._connect_synthetic_optimizers(module)
        self._validate_manual_optimization(module)
        self._cache_primary_optimization()

    def _connect_synthetic_optimizers(self, module: Module) -> None:
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
                partial(epoch_loop._automatic_optimizer_step, self, module),
                module.toggled_optimizer,
                self.strategy.defer_backward_sync
            )
            self._synthetic_optimizers.append(synthetic_optimizer)

    def _validate_manual_optimization(self, module: Module) -> None:
        if module.automatic_optimization:
            return
        epoch_loop: TrainingEpochLoop | None = self.fit_loop.epoch_loop
        if epoch_loop is None:
            raise RuntimeError("Trainer fit loop has no training epoch loop.")
        if epoch_loop.accumulate_grad_batches != 1:
            raise MisconfigurationError(
                "Manual optimization does not support accumulate_grad_batches != 1. "
                "Set accumulate_grad_batches = 1 and manage accumulation inside "
                "training_step if needed."
            )
        if self.gradient_clip_val is not None and self.gradient_clip_val > 0:
            raise MisconfigurationError(
                "Manual optimization does not support automatic gradient clipping. "
                "Call module.clip_gradients(...) explicitly inside training_step if needed."
            )

    def _cache_primary_optimization(self) -> None:
        self.optimizer: torch.optim.Optimizer | None = (
            self.optimizers[0] if self.optimizers else None
        )
        self.schedulers: list[_SchedulerInstance] = [
            configuration.scheduler for configuration in self.lr_scheduler_configs
        ]
        self.scheduler: _SchedulerInstance | None = (
            self.schedulers[0] if self.schedulers else None
        )
        self.scheduler_config: LRSchedulerConfig | None = (
            self.lr_scheduler_configs[0] if self.lr_scheduler_configs else None
        )

    def _restore_fit_checkpoint(
        self,
        module: Module,
        resume_path: str | Path | None,
        weights_only: bool | None,
        preloaded_checkpoint: CheckpointDict | None
    ) -> None:
        self._resolve_checkpoint_callback_directories()
        if resume_path is not None:
            if preloaded_checkpoint is None:
                raise RuntimeError("Fit checkpoint metadata was not preloaded.")
            self._restore_checkpoint(
                resume_path,
                module,
                weights_only=weights_only,
                checkpoint=None,
                restore_datamodule_state=False
            )
        device: torch.device = self._resolve_device()
        synthetic_optimizer: SyntheticOptimizer
        for synthetic_optimizer in self._synthetic_optimizers:
            synthetic_optimizer.move_state_to(device)

    def _start_fit_stage(
        self,
        module: Module,
        datamodule: DataModule | None,
        resume_path: str | Path | None
    ) -> None:
        self._callback_connector.validate_state_keys()
        self._hook_dispatcher.call_callbacks(module, "on_fit_start")
        self._hook_dispatcher.call_module(module, "on_fit_start", module.on_fit_start)
        self._autolog_hyperparameters(module, datamodule)
        if (
            self.enable_validation
            and self.num_sanity_val_steps > 0
            and resume_path is None
        ):
            self._sanity_check_validation(module)
        self.state.set_running_stage("training")
        module._running_stage: RunningStage | None = "training"

    def _run_fit_stage(self, module: Module) -> None:
        with torch.autograd.set_detect_anomaly(self._detect_anomaly):
            self.fit_loop.run()
        self._hook_dispatcher.call_callbacks(module, "on_fit_end")
        self._hook_dispatcher.call_module(module, "on_fit_end", module.on_fit_end)
        self._raise_if_sigterm()

    def validate(
        self,
        model: Module | None = None,
        dataloaders: DataLoaderSource | CombinedLoader | None = None,
        ckpt_path: str | Path | None = None,
        verbose: bool = True,
        datamodule: DataModule | None = None,
        weights_only: bool | None = None
    ) -> list[dict[str, float]]:
        return self._evaluate(
            "validate",
            self.validate_loop,
            model,
            dataloaders,
            ckpt_path,
            verbose,
            datamodule,
            weights_only
        )

    def test(
        self,
        model: Module | None = None,
        dataloaders: DataLoaderSource | CombinedLoader | None = None,
        ckpt_path: str | Path | None = None,
        verbose: bool = True,
        datamodule: DataModule | None = None,
        weights_only: bool | None = None
    ) -> list[dict[str, float]]:
        return self._evaluate(
            "test",
            self.test_loop,
            model,
            dataloaders,
            ckpt_path,
            verbose,
            datamodule,
            weights_only
        )

    def _evaluate(
        self,
        stage: Literal["validate", "test"],
        loop: EvaluationLoop,
        model: Module | None,
        dataloaders: DataLoaderSource | CombinedLoader | None,
        ckpt_path: str | Path | None,
        verbose: bool,
        datamodule: DataModule | None,
        weights_only: bool | None
    ) -> list[dict[str, float]]:
        model_provided: bool = model is not None
        module: Module = self._resolve_model(model, stage)
        resolved_datamodule: DataModule | None = (
            dataloaders if isinstance(dataloaders, DataModule) else datamodule
        )
        resolved_dataloaders: DataLoaderSource | CombinedLoader | None = (
            None if isinstance(dataloaders, DataModule) else dataloaders
        )
        if resolved_dataloaders is not None and resolved_datamodule is not None:
            raise MisconfigurationError(
                f"You cannot pass both `Trainer.{stage}(dataloaders=..., "
                "datamodule=...)`."
            )
        if type(verbose) is not bool:
            raise TypeError("`verbose` must be a bool.")
        if weights_only is not None and type(weights_only) is not bool:
            raise TypeError("`weights_only` must be a bool or None.")
        loop.verbose: bool = verbose
        selected_checkpoint_path: str | Path | None = self._select_checkpoint_path(
            ckpt_path,
            stage=stage,
            model_provided=model_provided,
            ckpt_path_provided=ckpt_path is not None
        )
        self._attach_model_callbacks(module, stage)
        launcher: _ProcessLauncher | None = self._stage_launcher()
        if launcher is not None and launcher.should_launch:
            try:
                return cast(
                    list[dict[str, float]],
                    launcher.launch(
                        stage,
                        self,
                        module,
                        resolved_datamodule,
                        {
                            "dataloaders": resolved_dataloaders,
                            "ckpt_path": selected_checkpoint_path,
                            "verbose": verbose,
                            "weights_only": weights_only
                        }
                    )
                )
            finally:
                self._clear_model_callback_configuration(stage)
                self.state.finish()
        metrics: list[dict[str, float]] = self._evaluation_impl(
            stage,
            loop,
            module,
            resolved_dataloaders,
            resolved_datamodule,
            selected_checkpoint_path,
            weights_only
        )
        if verbose and self.is_global_zero:
            display_stage: str = "validation" if stage == "validate" else "test"
            ConsoleRenderer().write_metrics(display_stage, metrics)
        return metrics

    def _evaluation_impl(
        self,
        stage: Literal["validate", "test"],
        loop: EvaluationLoop,
        module: Module,
        dataloaders: DataLoaderSource | CombinedLoader | None,
        datamodule: DataModule | None,
        checkpoint_path: str | Path | None,
        weights_only: bool | None
    ) -> list[dict[str, float]]:
        active_module: Module = module
        data_stage: Literal["val", "test"] = (
            "val" if stage == "validate" else "test"
        )
        running_stage: Literal["validating", "testing"] = (
            "validating" if stage == "validate" else "testing"
        )
        self._begin_inference_stage(
            active_module,
            datamodule,
            dataloaders,
            stage,
            data_stage,
            running_stage
        )
        training_modes: dict[str, bool] = active_module._capture_training_modes()
        stage_completed: bool = False
        metrics: list[dict[str, float]] = []
        try:
            active_module: Module = self._prepare_inference_stage(
                active_module,
                datamodule,
                stage,
                data_stage,
                checkpoint_path,
                weights_only
            )
            metrics: list[dict[str, float]] = loop.run()
            stage_completed: bool = True
        except BaseException as exception:
            attached_module: Module | None = self.module
            if attached_module is not None:
                active_module: Module = attached_module
            self._handle_exception(active_module, datamodule, exception)
        finally:
            self._finish_stage(
                active_module,
                datamodule,
                stage,
                stage_completed,
                training_modes
            )
        return metrics

    def _begin_inference_stage(
        self,
        module: Module,
        datamodule: DataModule | None,
        dataloaders: DataLoaderSource | CombinedLoader | None,
        stage: Literal["validate", "test", "predict"],
        data_stage: Literal["val", "test", "predict"],
        running_stage: Literal["validating", "testing", "predicting"]
    ) -> None:
        self.state.set_stage(stage)
        self._active_exception: BaseException | None = None
        self._logger_connector.clear_metric_views()
        module._reset_logging_state()
        self._reset_progress_bar_metrics(running_stage)
        self._data_connector.reset_stage(data_stage)
        self._attach_data(module, datamodule)
        self._data_connector.attach_source(
            data_stage,
            module,
            datamodule,
            dataloaders
        )

    def _prepare_inference_stage(
        self,
        module: Module,
        datamodule: DataModule | None,
        stage: Literal["validate", "test", "predict"],
        data_stage: Literal["val", "test", "predict"],
        checkpoint_path: str | Path | None,
        weights_only: bool | None
    ) -> Module:
        self._signal_connector.register_signal_handlers()
        self.strategy.setup_environment()
        self._announce_barebones_stage(stage)
        self._setup_profiler(stage, module)
        self._prepare_data_and_setup(module, datamodule, stage)
        self._request_dataloaders(data_stage)
        device: torch.device = self._resolve_device()
        configured_module: Module = self._setup_model(module, device)
        empty_schedulers: list[_SchedulerInstance] = []
        precision_connection: tuple[
            Module,
            list[torch.optim.Optimizer],
            list[_SchedulerInstance]
        ] = self._connect_precision_plugin(
            configured_module,
            [],
            empty_schedulers
        )
        configured_module: Module = precision_connection[0]
        if checkpoint_path is not None:
            self._resolve_checkpoint_callback_directories()
            self._restore_checkpoint(
                checkpoint_path,
                configured_module,
                weights_only=weights_only
            )
        self._autolog_hyperparameters(configured_module, datamodule)
        return configured_module

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
        resolved_datamodule: DataModule | None = (
            dataloaders if isinstance(dataloaders, DataModule) else datamodule
        )
        resolved_dataloaders: DataLoaderSource | CombinedLoader | None = (
            None if isinstance(dataloaders, DataModule) else dataloaders
        )
        if resolved_dataloaders is not None and resolved_datamodule is not None:
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
        launcher: _ProcessLauncher | None = self._stage_launcher()
        if launcher is not None and launcher.should_launch:
            try:
                return cast(
                    list[ModelOutput] | list[list[ModelOutput]] | None,
                    launcher.launch(
                        "predict",
                        self,
                        module,
                        resolved_datamodule,
                        {
                            "dataloaders": resolved_dataloaders,
                            "return_predictions": return_predictions,
                            "ckpt_path": selected_checkpoint_path,
                            "weights_only": weights_only
                        }
                    )
                )
            finally:
                self._clear_model_callback_configuration("predict")
                self.state.finish()
        return self._predict_impl(
            module,
            resolved_datamodule,
            resolved_dataloaders,
            selected_checkpoint_path,
            weights_only,
            return_predictions,
            launcher
        )

    def _predict_impl(
        self,
        module: Module,
        datamodule: DataModule | None,
        dataloaders: DataLoaderSource | CombinedLoader | None,
        checkpoint_path: str | Path | None,
        weights_only: bool | None,
        return_predictions: bool | None,
        launcher: _ProcessLauncher | None
    ) -> list[ModelOutput] | list[list[ModelOutput]] | None:
        active_module: Module = module
        self._begin_inference_stage(
            active_module,
            datamodule,
            dataloaders,
            "predict",
            "predict",
            "predicting"
        )
        training_modes: dict[str, bool] = active_module._capture_training_modes()
        stage_completed: bool = False
        predictions: list[ModelOutput] | list[list[ModelOutput]] | None = None
        try:
            active_module: Module = self._prepare_inference_stage(
                active_module,
                datamodule,
                "predict",
                "predict",
                checkpoint_path,
                weights_only
            )
            self.predict_loop.return_predictions: bool = (
                launcher is None or launcher.start_method == "popen"
                if return_predictions is None
                else return_predictions
            )
            predictions: list[ModelOutput] | list[list[ModelOutput]] | None = (
                self.predict_loop.run()
            )
            stage_completed: bool = True
        except BaseException as exception:
            attached_module: Module | None = self.module
            if attached_module is not None:
                active_module: Module = attached_module
            self._handle_exception(active_module, datamodule, exception)
        finally:
            self._finish_stage(
                active_module,
                datamodule,
                "predict",
                stage_completed,
                training_modes
            )
        return predictions

    def _stage_launcher(self) -> _ProcessLauncher | None:
        return (
            self.strategy.launcher
            if isinstance(self.strategy, ParallelStrategy)
            else None
        )

    def _finish_stage(
        self,
        module: Module,
        datamodule: DataModule | None,
        stage: TrainerStage,
        completed: bool,
        training_modes: dict[str, bool]
    ) -> None:
        modes_to_restore: dict[str, bool] | None = (
            None if completed else training_modes
        )
        cleanup_exception: BaseException | None = self._teardown_stage(
            module,
            datamodule,
            stage,
            modes_to_restore
        )
        module._running_stage: RunningStage | None = None
        module._current_fx_name: str | None = None
        self.state.finish()
        self._active_exception: BaseException | None = None
        self._is_mid_epoch_validation: bool = False
        self._clear_model_callback_configuration(stage)
        if stage == "fit" and self.fit_loop.epoch_loop is not None:
            self.fit_loop.epoch_loop.reset_on_teardown()
        if cleanup_exception is not None:
            raise cleanup_exception

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
        self.module: Module | None = module
        self.datamodule: DataModule | None = datamodule
        module.trainer: Trainer | None = self
        self.strategy.connect(module)
        if datamodule is not None:
            datamodule.trainer: Trainer | None = self

    def _attach_model_callbacks(
        self,
        module: Module,
        stage: TrainerStage
    ) -> None:
        self.state.set_stage(stage)
        try:
            self.module: Module | None = module
            module.trainer: Trainer | None = self
            self.strategy.connect(module)
            self._callback_connector.configure_model_callbacks(
                module,
                stage,
                partial(
                    self._hook_dispatcher.call_module,
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
        if self.is_global_zero and datamodule is not None:
            self._hook_dispatcher.call_datamodule(
                datamodule,
                "prepare_data",
                datamodule.prepare_data
            )
        if self.is_global_zero and self._hook_dispatcher.is_overridden(
            "prepare_data",
            module,
            DataHooks
        ):
            self._hook_dispatcher.call_module(
                module,
                "prepare_data",
                module.prepare_data
            )
        self._hook_dispatcher.call_strategy("barrier", self.strategy.barrier)
        if datamodule is not None:
            self._hook_dispatcher.call_datamodule(
                datamodule,
                "setup",
                datamodule.setup,
                stage
            )
        self._hook_dispatcher.call_callbacks(module, "setup", stage)
        self._hook_dispatcher.call_module(module, "setup", module.setup, stage)

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
            self._hook_dispatcher.call_module,
            self._hook_dispatcher.call_datamodule
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
            source_is_defined: bool = self._hook_dispatcher.is_overridden(
                "val_dataloader",
                source,
                DataModule
            )
        elif isinstance(source, Module):
            source_is_defined: bool = self._hook_dispatcher.is_overridden(
                "val_dataloader",
                source,
                DataHooks
            )
        validation_enabled: bool = (
            self.limit_val_batches != 0
            and module is not None
            and self._hook_dispatcher.is_overridden("validation_step", module, Module)
            and source_is_defined
        )
        if validation_enabled and self._data_connector.total_val_batches is not None:
            validation_enabled: bool = any(
                self._data_connector.resolve_batch_limit(total, "val") != 0
                for total in self._data_connector.total_val_batches
            )
        self._validation_schedule.configure(
            max_batches,
            validation_enabled,
            reset_time
        )
        self.val_check_batch: int | None = self._validation_schedule.batch_interval

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

    def _connect_loops(self) -> None:
        self.fit_loop.trainer: Trainer = self
        assert self.fit_loop.epoch_loop is not None
        self.fit_loop.epoch_loop.trainer: Trainer = self
        self.validate_loop.trainer: Trainer = self
        self.test_loop.trainer: Trainer = self
        self.predict_loop.trainer: Trainer = self

    def _announce_barebones_stage(self, stage: TrainerStage) -> None:
        if not self.barebones or not self.is_global_zero:
            return
        ConsoleRenderer().write_permanent(
            f"syntheticmind · {stage} · barebones"
        )

    def _resolve_profiler(self, profiler: Profiler | str | None) -> Profiler:
        if profiler is None:
            return PassThroughProfiler()
        if isinstance(profiler, Profiler):
            return profiler
        if not isinstance(profiler, str):
            raise TypeError(
                "`profiler` must be a Profiler instance, a supported string, or None."
            )
        match profiler.lower():
            case "simple":
                return SimpleProfiler()
            case "advanced":
                return AdvancedProfiler()
            case "pytorch":
                return PyTorchProfiler()
            case unsupported_name:
                raise MisconfigurationError(
                    "When passing a profiler name, expected 'simple', 'advanced', "
                    f"or 'pytorch', got {unsupported_name!r}."
                )

    def _setup_profiler(self, stage: TrainerStage, module: Module) -> None:
        if self.profiler.is_active:
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

    def _teardown_profiler(self) -> None:
        if not self.profiler.is_active:
            return
        self.profiler.describe()

    def _transfer_batch(
        self,
        module: Module,
        batch: Batch,
        device: torch.device,
        dataloader_idx: int
    ) -> Batch:
        datamodule: DataModule | None = self.datamodule
        if datamodule is not None and self._hook_dispatcher.is_overridden(
            "on_before_batch_transfer",
            datamodule,
            DataModule
        ):
            batch: Batch = self._hook_dispatcher.call_datamodule(
                datamodule,
                "on_before_batch_transfer",
                datamodule.on_before_batch_transfer,
                batch,
                dataloader_idx=dataloader_idx
            )
        else:
            batch: Batch = self._hook_dispatcher.call_module(
                module,
                "on_before_batch_transfer",
                module.on_before_batch_transfer,
                batch,
                dataloader_idx=dataloader_idx
            )
        if datamodule is not None and self._hook_dispatcher.is_overridden(
            "transfer_batch_to_device",
            datamodule,
            DataModule
        ):
            batch: Batch = self._hook_dispatcher.call_datamodule(
                datamodule,
                "transfer_batch_to_device",
                datamodule.transfer_batch_to_device,
                batch,
                device,
                dataloader_idx=dataloader_idx
            )
        else:
            batch: Batch = self._hook_dispatcher.call_module(
                module,
                "transfer_batch_to_device",
                module.transfer_batch_to_device,
                batch,
                device,
                dataloader_idx=dataloader_idx
            )
        if datamodule is not None and self._hook_dispatcher.is_overridden(
            "on_after_batch_transfer",
            datamodule,
            DataModule
        ):
            batch: Batch = self._hook_dispatcher.call_datamodule(
                datamodule,
                "on_after_batch_transfer",
                datamodule.on_after_batch_transfer,
                batch,
                dataloader_idx=dataloader_idx
            )
        else:
            batch: Batch = self._hook_dispatcher.call_module(
                module,
                "on_after_batch_transfer",
                module.on_after_batch_transfer,
                batch,
                dataloader_idx=dataloader_idx
            )
        return batch

    def _configure_model(self, module: Module) -> None:
        if not self._hook_dispatcher.is_overridden("configure_model", module, ModelHooks):
            return
        self._hook_dispatcher.call_module(
            module,
            "configure_model",
            module.configure_model
        )

    def _configure_torch_flags(
        self,
        deterministic: bool | Literal["warn"] | None,
        benchmark: bool | None
    ) -> None:
        if deterministic is not None and type(deterministic) is not bool and not (
            isinstance(deterministic, str) and deterministic == "warn"
        ):
            raise MisconfigurationError(
                "`deterministic` must be True, False, 'warn', or None."
            )
        if benchmark is not None and type(benchmark) is not bool:
            raise TypeError(f"`benchmark` must be a bool or None, got {benchmark!r}.")
        resolved_benchmark: bool | None = benchmark
        if deterministic:
            if resolved_benchmark is None:
                resolved_benchmark: bool | None = False
            elif resolved_benchmark:
                warnings.warn(
                    "`deterministic` and `benchmark=True` were both requested. "
                    "cuDNN benchmark selection can prevent deterministic execution.",
                    stacklevel=3
                )
        if resolved_benchmark is not None:
            torch.backends.cudnn.benchmark: bool = resolved_benchmark
        if deterministic == "warn":
            torch.use_deterministic_algorithms(True, warn_only=True)
        elif type(deterministic) is bool:
            torch.use_deterministic_algorithms(deterministic)
        if deterministic:
            os.environ.update({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})

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
            module: Module = synchronized_model
            self._layer_sync_model: Module | None = synchronized_model
        converted_model: torch.nn.Module = self.precision_plugin.convert_module(module)
        if not isinstance(converted_model, Module):
            raise TypeError(
                "Precision.convert_module must return Module, got "
                f"{type(converted_model).__name__}."
            )
        resolved_module: Module = converted_model
        if self._layer_sync_model is not None:
            self._layer_sync_model: Module | None = resolved_module
        resolved_module.trainer: Trainer | None = self
        self.module: Module | None = resolved_module
        self.strategy.connect(resolved_module)
        self.profiler._attach_module(resolved_module)
        self._hook_dispatcher.call_strategy(
            "setup",
            self.strategy.setup,
            resolved_module,
            device
        )
        resolved_module._device: torch.device = device
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
                "Precision.connect must return torch.nn.Module, got "
                f"{type(connected_model).__name__}."
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
        self._hook_dispatcher.call_module(
            module,
            "on_validation_model_zero_grad",
            module.on_validation_model_zero_grad
        )

    def _resolve_device(self) -> torch.device:
        return self.strategy.root_device

    def _sanity_check_validation(self, module: Module) -> None:
        self.sanity_checking: bool = True
        module._running_stage: RunningStage | None = "sanity_checking"

        try:
            self._ensure_val_dataloader()
            self._hook_dispatcher.call_callbacks(module, "on_sanity_check_start")
            self.validate_loop.run()
            if not self._validation_schedule.is_epoch_eligible(self.current_epoch):
                self._data_connector.last_val_reload_epoch: int | None = None
            self._validation_schedule.mark_validation_end()
            self.validate_loop.epoch_metrics: dict[str, float] = {}
            module._reset_logging_state()
            self._logger_connector.clear_metric_views()
            self._reset_progress_bar_metrics("validating")
            self._hook_dispatcher.call_callbacks(module, "on_sanity_check_end")
        finally:
            self.sanity_checking: bool = False
            self.training: bool = True
            module._running_stage: RunningStage | None = "training"

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
        if self.state.stage == "fit":
            current_configuration: _TrainerConfigurationSnapshot = (
                _TrainerConfigurationSnapshot.capture(
                    self._configuration,
                    self.precision
                )
            )
            self._checkpoint_connector.validate_fit_configuration(
                loaded_checkpoint,
                current_configuration
            )
        self._hook_dispatcher.call_module(
            module,
            "on_load_checkpoint",
            module.on_load_checkpoint,
            loaded_checkpoint
        )
        self._hook_dispatcher.call_callbacks(module, "on_load_checkpoint", loaded_checkpoint)
        self.precision_plugin.on_load_checkpoint(loaded_checkpoint)
        self._checkpoint_connector.restore_strategy_state(
            loaded_checkpoint,
            strict=module.strict_loading,
            restore_optimizers=self.state.stage == "fit"
        )
        restored_training_state: _RestoredTrainingState = (
            self._checkpoint_connector.restore_training_state(
                checkpoint=loaded_checkpoint,
                scheduler=(self.scheduler if self.state.stage == "fit" else None),
                schedulers=(self.schedulers if self.state.stage == "fit" else []),
                datamodule=self.datamodule,
                precision_plugin=self.precision_plugin,
                restore_datamodule_state=restore_datamodule_state,
                restore_schedulers=self.state.stage == "fit"
            )
        )
        self._apply_restored_training_state(
            module,
            loaded_checkpoint,
            restored_training_state
        )

    def _apply_restored_training_state(
        self,
        module: Module,
        checkpoint: CheckpointDict,
        restored_state: _RestoredTrainingState
    ) -> None:
        epoch: int = restored_state[0]
        global_step: int = restored_state[1]
        completed_batches: int = restored_state[2]
        total_batch_idx: int = restored_state[3]
        epoch_rng_state: CheckpointDict | None = restored_state[4]
        combined_loader_states: list[DataLoaderState] = (
            self._checkpoint_connector.combined_loader_states(checkpoint)
            if self.state.stage == "fit"
            else []
        )
        has_training_continuation: bool = self.state.stage == "fit" and (
            completed_batches > 0
            or total_batch_idx >= 0
            or bool(combined_loader_states)
        )
        if has_training_continuation:
            self._checkpoint_connector.train_dataloader_states: list[DataLoaderState] = (
                combined_loader_states
            )
            self._checkpoint_connector.train_dataloader_state_active: bool = True
            self._checkpoint_connector.train_dataloader_batches_completed: int = completed_batches
            self._checkpoint_connector.train_total_batch_idx: int = total_batch_idx
            self._checkpoint_connector.train_epoch_rng_state: CheckpointDict | None = (
                epoch_rng_state
            )
            if (
                combined_loader_states
                and self._data_connector.train_combined_loader is not None
            ):
                self._data_connector.train_combined_loader._load_state_dicts(
                    combined_loader_states
                )
        self.state.load_state_dict({"current_epoch": epoch, "global_step": global_step})
        module._current_epoch: int = epoch
        module._global_step: int = global_step
        if self.fit_loop.epoch_loop is not None:
            loader_state_restored: bool = bool(
                combined_loader_states
                and self._data_connector.train_combined_loader is not None
            )
            self.fit_loop.epoch_loop._combined_loader_state_restored: bool = (
                loader_state_restored
            )
            if completed_batches > 0:
                self.fit_loop.epoch_loop._completed_batches: int = completed_batches
            if total_batch_idx >= 0:
                self.fit_loop.epoch_loop._total_batches_processed: int = (
                    total_batch_idx + 1
                )
            if epoch_rng_state is not None and not loader_state_restored:
                self.fit_loop.epoch_loop._resume_rng_state: CheckpointDict | None = (
                    epoch_rng_state
                )
            checkpoint_rng_state: CheckpointValue | None = checkpoint.get(
                "rng_state"
            )
            if (
                has_training_continuation
                and isinstance(checkpoint_rng_state, dict)
            ):
                self.fit_loop.epoch_loop._resume_checkpoint_rng_state: CheckpointDict | None = (
                    cast(CheckpointDict, checkpoint_rng_state)
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
                callback.dirpath: str | None = synchronized_dirpath

    def _build_checkpoint(self, weights_only: bool = False) -> CheckpointDict:
        module: Module | None = self.module
        if module is None:
            raise AttributeError(
                "Saving a checkpoint requires a model attached through a Trainer stage."
            )
        loop_state: _CheckpointLoopState = self._checkpoint_loop_state()
        checkpoint_completed_batches: int = loop_state[0]
        checkpoint_total_batch_idx: int = loop_state[1]
        epoch_rng_state: CheckpointDict | None = loop_state[2]
        combined_loader_states: list[DataLoaderState] = loop_state[3]
        is_mid_epoch: bool = loop_state[4]
        epoch: int = self._checkpoint_epoch(module, is_mid_epoch)
        optimizer_states: list[CheckpointValue] = self._checkpoint_optimizer_states(
            weights_only
        )
        configuration_snapshot: _TrainerConfigurationSnapshot = (
            _TrainerConfigurationSnapshot.capture(
                self._configuration,
                self.precision
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
            epoch_rng_state=epoch_rng_state,
            combined_loader_states=combined_loader_states,
            precision_plugin=self.precision_plugin,
            weights_only=weights_only,
            model_state_dict=cast(
                CheckpointValue,
                self._checkpoint_connector.model_state_dict()
            ),
            optimizer_states=optimizer_states,
            trainer_configuration=cast(
                CheckpointValue,
                configuration_snapshot.model_dump(mode="python")
            )
        )
        if not weights_only:
            self.precision_plugin.on_save_checkpoint(checkpoint)
            self._hook_dispatcher.call_callbacks(
                module,
                "on_save_checkpoint",
                checkpoint
            )
        self._hook_dispatcher.call_module(
            module,
            "on_save_checkpoint",
            module.on_save_checkpoint,
            checkpoint
        )
        return checkpoint

    def _checkpoint_loop_state(self) -> _CheckpointLoopState:
        epoch_loop: TrainingEpochLoop | None = self.fit_loop.epoch_loop
        completed_batches: int = 0
        total_batch_idx: int = -1
        epoch_rng_state: CheckpointDict | None = None
        has_entered_batch_loop: bool = False
        if self.state.stage == "fit" and epoch_loop is not None:
            completed_batches: int = epoch_loop._completed_batches
            total_batch_idx: int = epoch_loop._total_batches_processed - 1
            has_entered_batch_loop: bool = epoch_loop._has_entered_batch_loop
            if completed_batches > 0 or has_entered_batch_loop:
                epoch_rng_state: CheckpointDict | None = (
                    epoch_loop._epoch_start_rng_state
                )
        has_live_batch_progress: bool = completed_batches > 0 or has_entered_batch_loop
        has_persisted_progress: bool = (
            self._checkpoint_connector.train_dataloader_state_active
        )
        has_batch_progress: bool = (
            has_live_batch_progress or has_persisted_progress
        )
        is_validation_checkpoint: bool = self.state.running_stage == "validating"
        is_mid_epoch: bool = has_batch_progress and (
            self._is_mid_epoch_validation or not is_validation_checkpoint
        )
        combined_loader_states: list[DataLoaderState] = []
        if (
            is_mid_epoch
            and has_live_batch_progress
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
            epoch_rng_state: CheckpointDict | None = (
                self._checkpoint_connector.train_epoch_rng_state
            )
        checkpoint_completed_batches: int = (
            completed_batches
            if is_mid_epoch and has_live_batch_progress
            else self._checkpoint_connector.train_dataloader_batches_completed
            if self._checkpoint_connector.train_dataloader_state_active
            else 0
        )
        checkpoint_total_batch_idx: int = (
            total_batch_idx
            if is_mid_epoch and total_batch_idx >= 0
            else self._checkpoint_connector.train_total_batch_idx
            if self._checkpoint_connector.train_dataloader_state_active
            else -1
        )
        return (
            checkpoint_completed_batches,
            checkpoint_total_batch_idx,
            epoch_rng_state if is_mid_epoch else None,
            combined_loader_states,
            is_mid_epoch
        )

    def _checkpoint_epoch(self, module: Module, is_mid_epoch: bool) -> int:
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
        return epoch

    def _checkpoint_optimizer_states(
        self,
        weights_only: bool
    ) -> list[CheckpointValue]:
        optimizer_states: list[CheckpointValue] = []
        if weights_only:
            return optimizer_states
        optimizer: torch.optim.Optimizer
        for optimizer in self.optimizers:
            optimizer_states.append(
                cast(
                    CheckpointValue,
                    self._checkpoint_connector.optimizer_state(optimizer)
                )
            )
        return optimizer_states

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
            if self.profiler.is_active
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
        self._active_exception: BaseException | None = exception
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
                self._hook_dispatcher.call_callback(
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
                self._hook_dispatcher.call_datamodule(
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
            self._hook_dispatcher.call_strategy(
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
        stage: TrainerStage,
        training_modes: dict[str, bool] | None = None
    ) -> BaseException | None:
        failures: list[_TeardownFailure] = []
        if training_modes is not None:
            self._attempt_teardown(
                "module.restore_training_modes",
                module._restore_training_modes,
                failures,
                training_modes
            )
        self._teardown_framework_resources(failures)
        self._teardown_stage_dataloaders(stage, failures)
        self._teardown_stage_hooks(module, datamodule, stage, failures)
        self._attempt_teardown(
            "profiler.describe",
            self._teardown_profiler,
            failures
        )
        return self._resolve_teardown_failure(module, datamodule, failures)

    def _teardown_framework_resources(
        self,
        failures: list[_TeardownFailure]
    ) -> None:
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

    def _teardown_stage_dataloaders(
        self,
        stage: TrainerStage,
        failures: list[_TeardownFailure]
    ) -> None:
        combined_loaders: list[tuple[str, CombinedLoader | None]] = (
            [
                ("train", self._data_connector.train_combined_loader),
                ("val", self._data_connector.val_combined_loader)
            ]
            if stage == "fit"
            else [("val", self._data_connector.val_combined_loader)]
            if stage == "validate"
            else [
                (
                    stage,
                    self._data_connector.test_combined_loader
                    if stage == "test"
                    else self._data_connector.predict_combined_loader
                )
            ]
        )
        for loader_stage, combined_loader in combined_loaders:
            if combined_loader is not None:
                self._attempt_teardown(
                    f"{loader_stage}_combined_loader.reset",
                    combined_loader.reset,
                    failures
                )

    def _teardown_stage_hooks(
        self,
        module: Module,
        datamodule: DataModule | None,
        stage: TrainerStage,
        failures: list[_TeardownFailure]
    ) -> None:
        if datamodule is not None:
            self._attempt_teardown(
                "datamodule.teardown",
                self._hook_dispatcher.call_datamodule,
                failures,
                datamodule,
                "teardown",
                datamodule.teardown,
                stage
            )
        progress_callbacks: list[tuple[int, ProgressBarCallback]] = []
        for index, callback in enumerate(self.callbacks):
            if isinstance(callback, ProgressBarCallback):
                progress_callbacks.append((index, callback))
                continue
            self._attempt_teardown(
                f"callback[{index}].teardown",
                self._call_callback_teardown,
                failures,
                callback,
                module,
                stage
            )
        self._attempt_teardown(
            "module.teardown",
            self._call_module_teardown,
            failures,
            module,
            stage
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
                self._call_callback_teardown,
                failures,
                callback,
                module,
                stage
            )

    def _resolve_teardown_failure(
        self,
        module: Module,
        datamodule: DataModule | None,
        failures: list[_TeardownFailure]
    ) -> BaseException | None:
        exception_notified: bool = False
        if failures and self._active_exception is None:
            self._notify_exception(module, datamodule, failures[0][1])
            exception_notified: bool = True

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
        reverted_model.trainer: Trainer | None = self
        self.module: Module | None = reverted_model
        self.strategy.model: torch.nn.Module | None = reverted_model
        self._layer_sync_model: Module | None = None

    def _attempt_teardown[**Parameters](
        self,
        label: str,
        action: Callable[Parameters, None],
        failures: list[_TeardownFailure],
        *args: Parameters.args,
        **kwargs: Parameters.kwargs
    ) -> None:
        try:
            action(*args, **kwargs)
        except BaseException as exception:
            failures.append((label, exception))

    def _call_callback_teardown(
        self,
        callback: Callback,
        module: Module,
        stage: TrainerStage
    ) -> None:
        self._hook_dispatcher.call_callback(
            callback,
            module,
            "teardown",
            callback.teardown,
            stage
        )

    def _call_module_teardown(self, module: Module, stage: TrainerStage) -> None:
        self._hook_dispatcher.call_module(
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
            description: str = f"{description}: {detail}"
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
        self._callback_connector.callbacks: list[Callback] = callbacks

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
        self._checkpoint_connector.path: str | Path | None = ckpt_path

    @property
    def logger(self) -> Logger | None:
        return self._logger_connector.logger

    @logger.setter
    def logger(self, logger: Logger | None) -> None:
        self._logger_connector.logger: Logger | None = logger

    @property
    def loggers(self) -> list[Logger]:
        return self._logger_connector.loggers

    @loggers.setter
    def loggers(self, loggers: list[Logger] | None) -> None:
        self._logger_connector.loggers: list[Logger] | None = loggers

    @property
    def default_root_dir(self) -> str:
        return StoragePath(self._default_root_dir).normalized

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
        *values: _PrintValue,
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
            running_stage
            if running_stage is not None
            else self._logger_connector.last_progress_bar_stage
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
            self._signal_connector.received_sigterm: bool = True
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
        self.state.should_stop: bool = value

    @property
    def enable_validation(self) -> bool:
        return self._validation_schedule.enabled

    @property
    def train_dataloader(self) -> DataLoaderSource | None:
        if self._data_connector.train_combined_loader is None:
            return None
        return cast(
            DataLoaderSource,
            self._data_connector.train_combined_loader.iterables
        )

    @property
    def val_dataloaders(self) -> DataLoaderSource | None:
        if self._data_connector.val_combined_loader is None:
            return None
        return cast(
            DataLoaderSource,
            self._data_connector.val_combined_loader.iterables
        )

    @property
    def test_dataloaders(self) -> DataLoaderSource | None:
        if self._data_connector.test_combined_loader is None:
            return None
        return cast(
            DataLoaderSource,
            self._data_connector.test_combined_loader.iterables
        )

    @property
    def predict_dataloaders(self) -> DataLoaderSource | None:
        if self._data_connector.predict_combined_loader is None:
            return None
        return cast(
            DataLoaderSource,
            self._data_connector.predict_combined_loader.iterables
        )

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
            effective_batches: int | None = self._data_connector.resolve_batch_limit(
                total_batches,
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
            effective_batches: int | None = self._data_connector.resolve_batch_limit(
                total_batches,
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
            effective_batches: int | None = self._data_connector.resolve_batch_limit(
                total_batches,
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

    @override
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
