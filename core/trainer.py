# This module:
# 1. Provides the Trainer, the harness's central orchestrator: it validates run
#    configuration, resolves the accelerator, strategy, and device, constructs
#    and connects the execution loops, and drives the fit, validate, test, and
#    predict entry points
# 2. Builds optimizers and schedulers from the module's
#    OptimizationConfiguration, including total-step resolution for
#    step-interval schedules
# 3. Manages distributed-sampler wrapping, sanity validation, checkpoint
#    restoration, exception handling, and stage teardown
#
# Design decisions:
# - Every entry point follows one skeleton: bind the module and datamodule,
#   prepare data on rank zero behind a barrier, set up per-process state,
#   resolve and bind the device, execute the stage loop inside a try block, and
#   perform teardown in a finally block, so end hooks, component teardown, and
#   logger finalization execute even when the stage fails
# - prepare_data runs only on rank zero with a barrier afterward, so single-copy
#   work such as dataset downloads cannot race across processes
# - The automatic-optimization contract is deliberately restrictive: multiple
#   optimizers require manual optimization, scheduler definitions must be
#   validated configurations so the loop can own stepping intervals, and manual
#   optimization excludes gradient accumulation and automatic clipping; each
#   violation raises a configuration error carrying corrective instructions
# - Prediction wraps its dataloader with an unrepeated distributed sampler that
#   partitions the dataset without padding shards to equal length, because
#   padding duplicates would corrupt prediction outputs, whereas in training
#   they merely repeat work
# - Step-interval scheduler horizons are computed before the first epoch from
#   the limited batch count and the accumulation factor, so schedules span the
#   true optimizer-step budget rather than the raw batch count
# - Sanity validation executes a bounded validation pass before training and
#   then discards its epoch metrics, resets the module's logging buffer, and
#   clears the callback metrics, so sanity results can never leak into real
#   monitoring or checkpoint decisions
# - Stateful callbacks must expose unique state keys, validated before fitting,
#   so two instances of one callback class cannot overwrite each other's
#   checkpoint state
#
# Author: Rahul Sawhney

import inspect
import math
import os
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Sized
from contextlib import AbstractContextManager, contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path
from typing import IO, Literal, Never, cast

import torch
from loguru import logger as log
from torch.utils.data import (
    BatchSampler,
    DistributedSampler,
    IterableDataset,
    Sampler,
    SequentialSampler,
)

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.accelerators.cpu import CPUAccelerator
from syntheticmind.accelerators.cuda import CUDAAccelerator
from syntheticmind.accelerators.mps import MPSAccelerator
from syntheticmind.accelerators.topology import DeviceTopology, resolve_device_topology
from syntheticmind.callbacks.callback import Callback
from syntheticmind.callbacks.checkpoint import Checkpoint
from syntheticmind.callbacks.early_stopping import EarlyStopping
from syntheticmind.callbacks.model_checkpoint import ModelCheckpoint
from syntheticmind.callbacks.model_summary import ModelSummary
from syntheticmind.callbacks.progress_bar import ProgressBarCallback
from syntheticmind.core.datamodule import DataModule
from syntheticmind.core.hooks import DataHooks, ModelHooks
from syntheticmind.core.module import Module
from syntheticmind.core.optimizer import (
    LRSchedulerConfig,
    SyntheticOptimizer,
    _SchedulerInstance,
    _init_optimizers_and_lr_schedulers,
)
from syntheticmind.core.protocols import OptimizerTrainerProtocol
from syntheticmind.loggers.configuration import resolve_loggers
from syntheticmind.loggers.csv_logger import CSVLogger
from syntheticmind.loggers.logger import Logger
from syntheticmind.loggers.tensorboard import TensorBoardLogger
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
from syntheticmind.precision.resolver import PrecisionResolver
from syntheticmind.plugins import Plugin, PluginConfiguration, PluginResolver
from syntheticmind.plugins.io import CheckpointStorageOptions
from syntheticmind.plugins.layer_sync import LayerSync
from syntheticmind.profilers import PassThroughProfiler, Profiler, _resolve_profiler
from syntheticmind.state.checkpoint_state import CheckpointState
from syntheticmind.state.trainer_state import TrainerState
from syntheticmind.strategies.ddp import DDPStrategy
from syntheticmind.strategies.launcher import _ProcessLauncher
from syntheticmind.strategies.parallel import ParallelStrategy
from syntheticmind.strategies.single_device import SingleDeviceStrategy
from syntheticmind.strategies.strategy import Strategy
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
from syntheticmind.utilities.optimizers import _optimizers_to_device
from syntheticmind.utilities.seed import SeedManager
from syntheticmind.utilities.signal_connector import _SignalConnector
from syntheticmind.utilities.torch_flags import configure_torch_flags
from syntheticmind.utilities.types import (
    CheckpointDict,
    CheckpointValue,
    ClipAlgorithm,
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


class _UnrepeatedDistributedSampler(DistributedSampler):
    # Distributed sampler that assigns each rank a strided slice of the dataset
    # without padding shards to equal length. The standard DistributedSampler
    # repeats leading samples so every rank yields the same count, which is
    # acceptable during training but would emit duplicate predictions; this
    # sampler accepts unequal shard sizes instead.
    def __iter__(self) -> Iterator[int]:
        # Produces this rank's index sequence: the optionally shuffled full
        # index list is strided by rank so the union across ranks covers every
        # sample exactly once. Shuffling seeds from the base seed plus the
        # epoch, matching the parent sampler's epoch-dependent permutation.
        dataset: Sized = cast(Sized, self.dataset)
        if self.shuffle:
            g: torch.Generator = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices: list[int] = torch.randperm(len(dataset), generator=g).tolist()
        else:
            indices: list[int] = list(range(len(dataset)))
        indices: list[int] = indices[self.rank::self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        # Length of this rank's shard under the strided partition, which may
        # differ across ranks by one when the dataset size is not divisible by
        # the replica count.
        dataset: Sized = cast(Sized, self.dataset)
        return max(0, math.ceil((len(dataset) - self.rank) / self.num_replicas))


class Trainer:
    # Central orchestrator of the harness. Construction validates the run
    # configuration, seeds the random sources when requested, resolves the
    # accelerator and strategy, and builds the four connected loops; the
    # fit, validate, test, and predict entry points then execute complete
    # stages against a module and datamodule pair. Collaborator state bound
    # during a stage (dataloaders, optimizers, schedulers, metrics) lives on
    # the trainer so loops and callbacks share one authoritative view.
    #
    # Integration: a client constructs one Trainer per run and passes its
    # syntheticmind.core.module.Module subclass together with a
    # syntheticmind.core.datamodule.DataModule to an entry point.
    #
    # Example::
    #
    #     from syntheticmind.callbacks.early_stopping import EarlyStopping
    #     from syntheticmind.callbacks.model_checkpoint import ModelCheckpoint
    #     from syntheticmind.core.trainer import Trainer
    #
    #     model: AutoEncoder = AutoEncoder()
    #     datamodule: SpeechDataModule = SpeechDataModule(corpus_root)
    #     trainer: Trainer = Trainer(
    #         max_epochs=100,
    #         callbacks=[
    #             ModelCheckpoint(monitor="val_loss", mode="min"),
    #             EarlyStopping(monitor="val_loss", mode="min", patience=10)
    #         ],
    #         precision="16-mixed",
    #         seed=42
    #     )
    #     trainer.fit(model, datamodule)
    #     test_metrics: dict[str, float] = trainer.test(model, datamodule)
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
        # Validates the configuration surface, optionally seeds every random
        # source, resolves the accelerator and strategy, installs the default
        # progress bar unless one is already present, constructs the four
        # loops with their limits, connects them to this trainer, and prepares
        # the empty collaborator slots that the entry points populate per run.
        #
        # Args:
        #     max_epochs: Number of complete training epochs the fit loop
        #         executes before stopping. Also enters the total-step
        #         estimate used to materialize step-interval schedulers.
        #         Default: ``100``.
        #     accelerator: Compute backend. Accepts ``"cpu"``, ``"cuda"``,
        #         ``"mps"``, ``"auto"``, or a constructed Accelerator
        #         instance; ``"auto"`` prefers CUDA, then MPS, then CPU by
        #         availability. Default: ``"auto"``.
        #     strategy: Execution strategy. Accepts ``"single_device"``,
        #         ``"ddp"``, ``"auto"``, or a constructed Strategy instance;
        #         ``"auto"`` selects distributed data parallel only for
        #         multi-device CUDA configurations and single-device
        #         execution otherwise. Default: ``"auto"``.
        #     devices: Declared device count, consulted by ``"auto"``
        #         strategy resolution; values above one on CUDA select
        #         distributed data parallel. The distributed world size and
        #         per-process device binding come from the launcher
        #         environment, not from this value. Default: ``1``.
        #     callbacks: Callbacks invoked at every hook point in list
        #         order. The list is extended with the default progress bar
        #         when enable_progress_bar is true and none is present.
        #         Default: ``None``.
        #     logger: Experiment logger receiving step-cadence metrics,
        #         reduced epoch metrics, and hyperparameters; ``None``
        #         disables experiment logging. Default: ``None``.
        #     accumulate_grad_batches: Number of consecutive batches whose
        #         gradients accumulate before one optimizer step; under
        #         distributed data parallel, gradient synchronization is
        #         suppressed on non-boundary batches. Must be at least one.
        #         Default: ``1``.
        #     val_check_interval: Mid-epoch validation cadence. An integer
        #         counts optimizer steps under the ``"global_step"`` scope
        #         and batches within the epoch under the ``"epoch"`` scope;
        #         a float in (0.0, 1.0] selects evenly spaced fractional
        #         positions of the epoch, with ``1.0`` meaning no mid-epoch
        #         validation. Default: ``1.0``.
        #     val_check_interval_scope: Unit for integer interval values:
        #         ``"epoch"`` counts batches within each epoch and excludes
        #         the final batch position, ``"global_step"`` counts
        #         optimizer steps across epoch boundaries and requires an
        #         integer interval. Default: ``"epoch"``.
        #     validate_at_epoch_end: Whether the fit loop schedules the
        #         epoch-boundary validation pass. Step-cadence training
        #         disables this so validation runs strictly on the step
        #         interval. Default: ``True``.
        #     precision: Numeric execution mode: ``"32-true"`` for full
        #         float32, ``"16-mixed"`` for float16 autocast with gradient
        #         scaling, ``"bf16-mixed"`` for bfloat16 autocast without
        #         scaling. Default: ``"32-true"``.
        #     gradient_clip_val: Clipping threshold applied to gradients
        #         before each optimizer step, after unscaling under
        #         ``"16-mixed"``; ``None`` disables clipping.
        #         Default: ``None``.
        #     gradient_clip_algorithm: Clipping method: ``"norm"`` rescales
        #         by total gradient norm, ``"value"`` clamps each component.
        #         Default: ``"norm"``.
        #     log_every_n_steps: Global-step cadence at which step-routed
        #         training metrics are emitted to the logger. Must be at
        #         least one. Default: ``10``.
        #     enable_progress_bar: Whether the default progress bar callback
        #         is appended when the callback list does not already
        #         contain one. Default: ``True``.
        #     deterministic: Forwarded to the seeding call when a seed is
        #         provided: enforces deterministic algorithm selection and
        #         disables cuDNN benchmark autotuning. Recorded on the
        #         trainer for collaborators to consult. Default: ``False``.
        #     num_sanity_val_steps: Number of validation batches run before
        #         training starts to surface evaluation defects early;
        #         every trace of the pass is discarded afterwards. Zero
        #         disables the sanity pass. Default: ``2``.
        #     limit_train_batches: Ceiling on training batches per epoch.
        #         An integer is an absolute count, a float is a fraction of
        #         the dataloader length (at least one batch), ``None`` runs
        #         the full loader. Fractions below one require a sized
        #         loader. Default: ``None``.
        #     limit_val_batches: Ceiling on validation batches per pass,
        #         with the same integer, float, and ``None`` semantics.
        #         Default: ``None``.
        #     limit_test_batches: Ceiling on test batches per pass, with
        #         the same semantics. Default: ``None``.
        #     limit_predict_batches: Ceiling on prediction batches per
        #         pass, with the same semantics. Default: ``None``.
        #     use_distributed_sampler: Whether distributed runs wrap sized
        #         map-style dataloaders with DistributedSampler, shuffled
        #         for training and unshuffled for evaluation; prediction
        #         uses the unrepeated sampler so no rank emits padded
        #         duplicate outputs. Iterable datasets are never wrapped.
        #         Default: ``True``.
        #     sync_batchnorm: Whether batch-normalization layers are
        #         converted to synchronized batch normalization before
        #         distributed training. Default: ``False``.
        #     resume_from_checkpoint: Default checkpoint path restored at
        #         the start of fit; the ckpt_path argument of fit overrides
        #         it per call. Default: ``None``.
        #     seed: When provided, seeds Python, NumPy, and torch random
        #         sources at construction through the seed manager.
        #         Default: ``None``.
        #     reload_dataloaders_every_n_epochs: Rebuild cadence for the
        #         training and validation dataloaders; epochs whose index
        #         is a positive multiple of this value request fresh
        #         loaders from the datamodule. Zero keeps the initial
        #         loaders for the whole run. Default: ``0``.
        #
        # Raises:
        #     MisconfigurationError: If accumulate_grad_batches is below
        #         one, an integer val_check_interval is below one, a float
        #         val_check_interval is outside (0.0, 1.0], the interval
        #         scope is invalid, the ``"global_step"`` scope receives a
        #         non-integer interval, or log_every_n_steps is below one.
        #     ValueError: If the accelerator or strategy name is unknown.
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
        plugin_configuration: PluginConfiguration = PluginResolver.resolve(
            plugins,
            sync_batchnorm
        )
        if callbacks is None:
            resolved_callbacks: list[Callback] = []
        elif isinstance(callbacks, Callback):
            resolved_callbacks = [callbacks]
        elif isinstance(callbacks, list) and all(
            isinstance(callback, Callback) for callback in callbacks
        ):
            resolved_callbacks = list(callbacks)
        else:
            raise TypeError(
                "`callbacks` must be a Callback, a list of Callback instances, or None."
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
        resolved_loggers: list[Logger] = resolve_loggers(
            logger,
            self.default_root_dir,
            fast_dev_run_enabled
        )
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
        self._ckpt_path: str | Path | None = None
        self._ckpt_path_user_managed: bool = False

        if accumulate_grad_batches < 1:
            raise MisconfigurationError("accumulate_grad_batches must be >= 1")
        if seed is not None:
            SeedManager.seed(seed)

        configured_parallel_devices: list[torch.device] | None = None
        resolved_accelerator_input: str | Accelerator = accelerator
        accelerator_is_auto: bool = (
            isinstance(accelerator, str) and accelerator.lower() == "auto"
        )
        if isinstance(strategy, Strategy):
            if strategy.accelerator is not None:
                if not accelerator_is_auto:
                    raise MisconfigurationError(
                        "Set the accelerator through either Trainer or the strategy instance, not both."
                    )
                resolved_accelerator_input = strategy.accelerator
            if strategy.parallel_devices:
                configured_parallel_devices = strategy.parallel_devices
                if accelerator_is_auto and strategy.accelerator is None:
                    resolved_accelerator_input = configured_parallel_devices[0].type

        self.accelerator: Accelerator = self._resolve_accelerator(
            resolved_accelerator_input
        )
        device_topology: DeviceTopology = resolve_device_topology(
            self.accelerator,
            devices,
            num_nodes,
            configured_parallel_devices
        )
        self.devices: int | list[int] = device_topology.devices
        self.parallel_devices: list[torch.device] = list(
            device_topology.parallel_devices
        )
        self.strategy: Strategy = self._resolve_strategy(
            strategy,
            self.accelerator,
            self.parallel_devices,
            device_topology.num_nodes
        )
        self.strategy.configure_topology(
            self.accelerator,
            self.parallel_devices,
            device_topology.num_nodes
        )
        if (
            plugin_configuration.checkpoint_io is not None
            and self.strategy.checkpoint.configured_checkpoint_io is not None
        ):
            raise MisconfigurationError(
                "CheckpointIO was configured through both Trainer(plugins=...) and the strategy. "
                "Configure it in exactly one place."
            )
        if plugin_configuration.checkpoint_io is not None:
            self.strategy.checkpoint.attach_checkpoint_io(
                plugin_configuration.checkpoint_io
            )
        if plugin_configuration.cluster_environment is not None:
            if not isinstance(self.strategy, ParallelStrategy):
                raise MisconfigurationError(
                    "ClusterEnvironment requires a parallel strategy."
                )
            if self.strategy.configured_cluster_environment is not None:
                raise MisconfigurationError(
                    "ClusterEnvironment was configured through both Trainer(plugins=...) and the strategy. "
                    "Configure it in exactly one place."
                )
            self.strategy.attach_cluster_environment(
                plugin_configuration.cluster_environment
            )
        if plugin_configuration.layer_sync is not None:
            if not isinstance(self.strategy, DDPStrategy):
                raise MisconfigurationError("LayerSync requires DDPStrategy.")
            if self.strategy.configured_layer_sync is not None:
                raise MisconfigurationError(
                    "LayerSync was configured through both Trainer(plugins=...) and the strategy. "
                    "Configure it in exactly one place."
                )
            self.strategy.attach_layer_sync(plugin_configuration.layer_sync)
        self._precision_plugin: Precision = PrecisionResolver.resolve(
            precision,
            self.accelerator.name(),
            plugin_configuration.precision,
            self.strategy.configured_precision_plugin
        )
        self.strategy.attach_precision_plugin(self._precision_plugin)

        self.callbacks: list[Callback] = resolved_callbacks
        configured_checkpoint_callbacks: list[Checkpoint] = [
            callback
            for callback in self.callbacks
            if isinstance(callback, Checkpoint)
        ]
        if configured_checkpoint_callbacks and not resolved_enable_checkpointing:
            raise MisconfigurationError(
                "Trainer was configured with `enable_checkpointing=False` but found "
                "a Checkpoint callback in `callbacks`."
            )
        if not configured_checkpoint_callbacks and resolved_enable_checkpointing:
            self.callbacks.append(ModelCheckpoint())
        progress_callbacks: list[ProgressBarCallback] = [
            callback
            for callback in self.callbacks
            if isinstance(callback, ProgressBarCallback)
        ]
        if len(progress_callbacks) > 1:
            raise MisconfigurationError(
                "Only one progress bar callback is supported. Remove duplicate "
                "ProgressBarCallback instances from callbacks."
            )
        if not resolved_enable_progress_bar and progress_callbacks:
            raise MisconfigurationError(
                "Trainer was configured with enable_progress_bar=False but received "
                "a ProgressBarCallback in callbacks."
            )
        from syntheticmind.callbacks.timer import Timer

        timer_callbacks: list[Timer] = [
            callback
            for callback in self.callbacks
            if isinstance(callback, Timer)
        ]
        if max_time is not None:
            if timer_callbacks:
                warnings.warn(
                    "Ignoring Trainer(max_time=...) because the callback list already contains a Timer.",
                    stacklevel=2
                )
            else:
                timer: Timer = Timer(duration=max_time, interval="step", verbose=False)
                self.callbacks.append(timer)
                timer_callbacks.append(timer)

        if resolved_enable_progress_bar:
            progress_index: int | None = next(
                (
                    index
                    for index, callback in enumerate(self.callbacks)
                    if isinstance(callback, ProgressBarCallback)
                ),
                None
            )
            if not timer_callbacks:
                timer = Timer(verbose=False)
                if progress_index is None:
                    self.callbacks.append(timer)
                else:
                    self.callbacks.insert(progress_index, timer)
            if not progress_callbacks:
                self.callbacks.append(ProgressBarCallback())

        if resolved_enable_model_summary and not any(
            isinstance(callback, ModelSummary)
            for callback in self.callbacks
        ):
            self.callbacks.append(ModelSummary())

        non_checkpoint_callbacks: list[Callback] = [
            callback
            for callback in self.callbacks
            if not isinstance(callback, Checkpoint)
        ]
        checkpoint_callbacks: list[Callback] = [
            callback
            for callback in self.callbacks
            if isinstance(callback, Checkpoint)
        ]
        self.callbacks = non_checkpoint_callbacks + checkpoint_callbacks
        self._model_callbacks: list[Callback] = []
        self._model_callback_configuration: tuple[TrainerStage, Module] | None = None

        self._loggers: list[Logger] = resolved_loggers
        self.profiler: Profiler = _resolve_profiler(profiler)
        self.state: TrainerState = TrainerState()
        self._signal_connector: _SignalConnector = _SignalConnector(self)
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
        self._train_combined_loader: CombinedLoader | None = None
        self._val_combined_loader: CombinedLoader | None = None
        self._test_combined_loader: CombinedLoader | None = None
        self._predict_combined_loader: CombinedLoader | None = None
        self._last_train_dataloader_states: list[dict[str, object]] = []
        self._train_dataloader_state_active: bool = False
        self._last_train_dataloader_batches_completed: int = 0
        self._last_train_total_batch_idx: int = -1
        self._train_dataloader_source: object | None = None
        self._val_dataloader_source: object | None = None
        self._test_dataloader_source: object | None = None
        self._predict_dataloader_source: object | None = None
        self._total_train_batches: int | None = None
        self._total_val_batches: list[int | None] | None = None
        self._total_test_batches: list[int | None] | None = None
        self._total_predict_batches: list[int | None] | None = None
        self.reload_dataloaders_every_n_epochs: int = reload_dataloaders_every_n_epochs
        self._last_train_dataloader_reload_epoch: int | None = None
        self._last_val_dataloader_reload_epoch: int | None = None
        self._callback_metrics: dict[str, torch.Tensor] = {}
        self._logged_metrics: dict[str, torch.Tensor] = {}
        self._progress_bar_metrics_by_stage: dict[RunningStage, dict[str, float]] = {
            "training": {},
            "validating": {},
            "testing": {},
            "predicting": {},
            "sanity_checking": {}
        }
        self._last_progress_bar_stage: RunningStage | None = None
        self._dataloader_fetch_time_seconds: dict[RunningStage, float] = {}
        self._dataloader_iterator_creation_time_seconds: dict[RunningStage, float] = {}
        self._active_exception: BaseException | None = None
        self._profiler_is_setup: bool = False
        self._layer_sync_model: Module | None = None

    def fit(
        self,
        model: Module,
        train_dataloaders: object | None = None,
        val_dataloaders: object | None = None,
        datamodule: DataModule | None = None,
        ckpt_path: str | Path | None = None,
        weights_only: bool | None = None
    ) -> None:
        # Executes a complete training run. The sequence is: reset run-local
        # state, bind the module and datamodule, prepare and set up the data,
        # wrap dataloaders for distributed execution, resolve and bind the
        # device, build optimizers and schedulers from the module's
        # optimization configuration under the automatic-optimization
        # contract, optionally restore a checkpoint, fire the fit-start and
        # train-start hook pairs around an optional sanity validation, run the
        # fit loop, and unwind through the end hooks, teardown, and logger
        # finalization in the finally block. An explicit ckpt_path overrides
        # the constructor-level resume path.
        #
        # Args:
        #     module: The model to train. The trainer attaches itself to
        #         the module, materializes its configure_optimizers
        #         declaration into live optimizers and schedulers, and
        #         drives its step methods through the loops.
        #     datamodule: Provider of the training and validation
        #         dataloaders through prepare_data, setup, and the
        #         dataloader hooks; each hook must return a single
        #         DataLoader.
        #     ckpt_path: Checkpoint to restore before training starts;
        #         overrides the constructor-level resume path for this
        #         call. ``None`` falls back to that path or a fresh start.
        #         Default: ``None``.
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
            and not self._ckpt_path_user_managed
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
        # Clear every run-local surface before anything executes, so a second
        # fit call on the same trainer inherits no stop request, cached
        # callback metrics, progress counters, or mid-epoch resume markers
        # from an earlier run.
        self.state.set_stage("fit")
        self._active_exception = None
        self.state.reset_on_run()
        self._is_mid_epoch_validation = False
        self._callback_metrics.clear()
        self._logged_metrics.clear()
        module._reset_logging_state()
        self._reset_progress_bar_metrics()
        self._total_train_batches: int | None = None
        self._total_val_batches: list[int | None] | None = None
        self._train_combined_loader: CombinedLoader | None = None
        self._val_combined_loader: CombinedLoader | None = None
        self._last_train_dataloader_states: list[dict[str, object]] = []
        self._train_dataloader_state_active: bool = False
        self._last_train_dataloader_batches_completed: int = 0
        self._last_train_total_batch_idx: int = -1
        self._last_train_dataloader_reload_epoch = None
        self._last_val_dataloader_reload_epoch = None
        self.fit_loop.should_stop = False
        if self.fit_loop.epoch_loop is not None:
            self.fit_loop.epoch_loop.reset_on_run()
        # Bind the run participants and zero the module's progress mirrors.
        # The trainer back-reference is what powers module.log, the optimizer
        # and scheduler accessors, and gradient clipping inside the step
        # methods.
        self._attach_data(module, datamodule)
        self._train_dataloader_source = self._select_dataloader_source(
            module,
            datamodule,
            train_dataloaders
        )
        self._val_dataloader_source = self._select_dataloader_source(
            module,
            datamodule,
            val_dataloaders
        )
        module._current_epoch = 0
        module._global_step = 0
        module_mode: ModuleMode = ModuleMode()
        module_mode.capture(module)
        fit_completed: bool = False

        # An explicit per-call checkpoint path overrides the constructor-level
        # resume path; with both absent the run starts fresh.
        try:
            self._signal_connector.register_signal_handlers()
            # Initialize the strategy's process context, run single-copy data
            # preparation on global rank zero only, and hold every rank at the
            # barrier so no process enters setup against partially prepared
            # data.
            self.strategy.setup_environment()
            self._announce_barebones_stage("fit")
            self._setup_profiler("fit", module)

            self._prepare_data_and_setup(module, datamodule, "fit")

            preloaded_checkpoint: CheckpointDict | None = None
            if resume_path is not None:
                preloaded_checkpoint: CheckpointDict | None = (
                    self.strategy.checkpoint.load_metadata(
                        resume_path,
                        weights_only
                    )
                )
                resume_epoch: int = CheckpointState._integer_value(
                    preloaded_checkpoint.get("epoch"),
                    0
                )
                resume_global_step: int = CheckpointState._integer_value(
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
                    CheckpointState.load_datamodule_state(
                        preloaded_checkpoint,
                        datamodule
                    )

            # Retrieve the fit-stage dataloaders. The harness supports exactly
            # one loader per stage, so collection returns are rejected here
            # rather than failing obscurely inside the loops.
            # Under a distributed strategy, map-style datasets are re-wrapped
            # with rank-aware samplers so every process sees a disjoint shard;
            # iterable datasets own their sharding and pass through with a
            # warning.
            # Record loader lengths where they are knowable; unsized iterable
            # loaders leave the totals as None and downstream cadence math
            # degrades to its unsized behavior.
            if (
                self.limit_train_batches != 0
                and is_overridden("training_step", module, Module)
            ):
                self._request_dataloaders("train")
            else:
                self._total_train_batches = 0
            self._configure_validation_schedule(reset_time=True)

            # Advisory only: single-process loading feeding a non-CPU
            # accelerator is a common throughput mistake worth one warning,
            # never an error.
            if is_rank_zero():
                combined_loader: CombinedLoader | None = self._train_combined_loader
                if combined_loader is not None:
                    for train_dataloader in combined_loader.flattened:
                        if (
                            isinstance(train_dataloader, torch.utils.data.DataLoader)
                            and train_dataloader.num_workers == 0
                            and not isinstance(self.accelerator, CPUAccelerator)
                        ):
                            log.warning("num_workers=0 with GPU accelerator. Consider increasing for better throughput.")
                            break

            # Resolve the concrete device, hand it to the accelerator, and let
            # the strategy place or wrap the module on it. BatchNorm
            # conversion precedes strategy setup so a distributed wrapper is
            # built around the converted layers, and the module's device
            # mirror is stamped once placement is final.
            device: torch.device = self._resolve_device()
            module = self._setup_model(module, device)

            # Materialize the module's declarative optimization contract. The
            # declaration is normalized to parallel definition lists first, so
            # single-optimizer and multi-optimizer modules flow through one
            # build path.
            training_step_signature: inspect.Signature = inspect.signature(module.training_step)
            if "optimizer_idx" in training_step_signature.parameters:
                raise RuntimeError(
                    "Training with multiple optimizers is only supported with manual optimization. Remove "
                    "the optimizer_idx parameter from training_step, set automatic_optimization = False, "
                    "and access optimizers through self.optimizers()."
                )
            optimization_configuration: object = self._call_module_hook(
                module,
                "configure_optimizers",
                module.configure_optimizers
            )
            optimizer_initialization: tuple[
                list[torch.optim.Optimizer],
                list[LRSchedulerConfig]
            ] = _init_optimizers_and_lr_schedulers(
                optimization_configuration,
                module.automatic_optimization,
                is_overridden("lr_scheduler_step", module, Module)
            )
            self.optimizers: list[torch.optim.Optimizer] = optimizer_initialization[0]
            self.lr_scheduler_configs: list[LRSchedulerConfig] = optimizer_initialization[1]
            module, self.optimizers, self.lr_scheduler_configs = (
                self._connect_precision_plugin(
                    module,
                    self.optimizers,
                    self.lr_scheduler_configs
                )
            )
            self._synthetic_optimizers: list[SyntheticOptimizer] = [
                SyntheticOptimizer._to_synthetic_optimizer(
                    optimizer,
                    cast(OptimizerTrainerProtocol, self),
                    module
                )
                for optimizer in self.optimizers
            ]

            # OptimizerConfig records are compiled into live torch.optim
            # objects against the module's parameters; pre-built
            # torch.optim.Optimizer instances pass through untouched.
            # Contract guards, enforced before any hook can observe the run:
            # multiple optimizers demand manual optimization; automatic
            # optimization demands SchedulerConfig records so the loop owns
            # stepping cadence; manual optimization excludes trainer-managed
            # accumulation and clipping because the module owns its backward
            # path.
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

            # The first optimizer doubles as the primary alias consumed by the
            # automatic-optimization path; the scheduler surfaces start empty
            # and are filled only when definitions exist.
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

            # Duplicate stateful-callback keys are rejected before callback
            # setup so two instances cannot silently overwrite each other's
            # checkpoint state later.

            # Schedulers pair positionally with optimizers, one definition per
            # optimizer. SchedulerConfig records are compiled against the
            # resolved total-step horizon; pre-built scheduler instances pass
            # through with no managed config, leaving their stepping to the
            # module.
            # Checkpoint restoration happens after optimizers and schedulers
            # exist, because their serialized state must land on live objects,
            # and before any hook fires, so every hook observes the resumed
            # run.
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
            _optimizers_to_device(self.optimizers, device)

            self._validate_callback_state_keys()

            # Start hooks fire callbacks-first, then the module. The optional
            # sanity pass runs inside the fit stage but before training
            # starts, surfacing validation-path defects before the first
            # training epoch is paid for; it discards every trace of itself.
            self._call_callback_hooks(module, "on_fit_start")
            self._call_module_hook(
                module,
                "on_fit_start",
                module.on_fit_start
            )

            # The run configuration is announced to the logger from rank zero
            # only, keeping multi-process runs single-writer.
            self._autolog_hyperparameters(module, datamodule)

            if (
                self.enable_validation
                and self.num_sanity_val_steps > 0
                and resume_path is None
            ):
                self._sanity_check_validation(module)

            # Enter the training running-stage and hand control to the fit
            # loop, which owns epochs, optimization, cadence-driven
            # validation, and stop requests from here.
            self.state.set_running_stage("training")
            module._running_stage = "training"

            with torch.autograd.set_detect_anomaly(self._detect_anomaly):
                self.fit_loop.run()

            # End hooks mirror the start order reversed: the module first,
            # then the callbacks.
            self._call_callback_hooks(module, "on_fit_end")
            self._call_module_hook(module, "on_fit_end", module.on_fit_end)
            self._raise_if_sigterm()
            fit_completed: bool = True
        # Interrupts and failures funnel through one handler that marks the
        # trainer state interrupted, offers the exception to the callback,
        # datamodule, and strategy hooks, and re-raises.
        except BaseException as exception:
            self._handle_exception(module, datamodule, exception)
        finally:
            if not fit_completed:
                module_mode.restore(module)
            # Unwind unconditionally, success or failure: end-of-fit hooks,
            # component teardown, and logger finalization all execute so a
            # failed run still releases its resources and flushes its records.
            cleanup_exception: BaseException | None = self._teardown_stage(module, datamodule, "fit")
            # The module's stage markers are cleared and the epoch loop's
            # resume markers reset, so the module and loops leave fit inert
            # rather than carrying live-run state into later stages.
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
        dataloaders: object | None = None,
        ckpt_path: str | Path | None = None,
        verbose: bool = True,
        datamodule: DataModule | None = None,
        weights_only: bool | None = None
    ) -> list[dict[str, float]]:
        # Executes a standalone validation stage and returns its reduced epoch
        # metrics. The module's training-mode flag is captured before the run
        # and restored in the finally block, so evaluating a model mid-workflow
        # does not permanently switch it out of training mode.
        #
        # Args:
        #     module: The model to evaluate; its validation steps run under
        #         inference mode.
        #     datamodule: Provider of the validation dataloader through
        #         prepare_data, setup, and the dataloader hook.
        #
        # Returns:
        #     The reduced epoch metrics of the validation pass, keyed by
        #     metric name.
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
        self._callback_metrics.clear()
        self._logged_metrics.clear()
        module._reset_logging_state()
        self._reset_progress_bar_metrics("validating")
        self._total_val_batches: list[int | None] | None = None
        self._val_combined_loader: CombinedLoader | None = None
        self._attach_data(module, datamodule)
        self._val_dataloader_source = self._select_dataloader_source(
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
        dataloaders: object | None = None,
        ckpt_path: str | Path | None = None,
        verbose: bool = True,
        datamodule: DataModule | None = None,
        weights_only: bool | None = None
    ) -> list[dict[str, float]]:
        # Executes a standalone test stage and returns its reduced epoch
        # metrics, following the same setup, execution, and teardown skeleton
        # as validate with the test dataloader and test loop.
        #
        # Args:
        #     module: The model to evaluate; its test steps run under
        #         inference mode.
        #     datamodule: Provider of the test dataloader through
        #         prepare_data, setup, and the dataloader hook.
        #
        # Returns:
        #     The reduced epoch metrics of the test pass, keyed by metric
        #     name.
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
        self._callback_metrics.clear()
        self._logged_metrics.clear()
        module._reset_logging_state()
        self._reset_progress_bar_metrics("testing")
        self._total_test_batches: list[int | None] | None = None
        self._test_combined_loader: CombinedLoader | None = None
        self._attach_data(module, datamodule)
        self._test_dataloader_source = self._select_dataloader_source(
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
        dataloaders: object | None = None,
        datamodule: DataModule | None = None,
        return_predictions: bool | None = None,
        ckpt_path: str | Path | None = None,
        weights_only: bool | None = None
    ) -> list[ModelOutput] | list[list[ModelOutput]] | None:
        # Executes a standalone prediction stage and returns the collected
        # step outputs. Distributed execution uses the unrepeated sampler so
        # no rank emits padded duplicate predictions, and the retention flag
        # is forwarded to the prediction loop before it runs.
        #
        # Args:
        #     module: The model to run inference with; its predict steps
        #         run under inference mode and metric logging is rejected
        #         by contract.
        #     datamodule: Provider of the prediction dataloader through
        #         prepare_data, setup, and the dataloader hook.
        #     return_predictions: Whether step outputs are retained and
        #         returned. Disable for measurement-only passes where
        #         retention would hold every output in memory for the whole
        #         stage. Default: ``True``.
        #
        # Returns:
        #     The prediction-step outputs in batch order, or an empty list
        #     when retention is disabled.
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
        self._callback_metrics.clear()
        self._logged_metrics.clear()
        module._reset_logging_state()
        self._reset_progress_bar_metrics("predicting")
        self._total_predict_batches: list[int | None] | None = None
        self._predict_combined_loader: CombinedLoader | None = None
        self._attach_data(module, datamodule)
        self._predict_dataloader_source = self._select_dataloader_source(
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
            active_configuration = self._model_callback_configuration
            if (
                active_configuration is not None
                and active_configuration[0] == stage
                and active_configuration[1] is module
            ):
                return
            self._model_callback_configuration = (stage, module)
            configured_callbacks: object = self._call_module_hook(
                module,
                "configure_callbacks",
                module.configure_callbacks
            )
            if configured_callbacks is None:
                return
            if isinstance(configured_callbacks, Callback):
                raw_callbacks: list[object] = [configured_callbacks]
            elif isinstance(configured_callbacks, Sequence):
                raw_callbacks = list(configured_callbacks)
            else:
                raise TypeError(
                    "configure_callbacks() must return a Callback, a sequence of Callback instances, or None."
                )
            if not all(isinstance(callback, Callback) for callback in raw_callbacks):
                raise TypeError(
                    "configure_callbacks() must return only Callback instances."
                )
            model_callbacks: list[Callback] = cast(
                list[Callback],
                raw_callbacks
            )
            if not model_callbacks:
                return
            previous_model_callback_ids: set[int] = {
                id(callback) for callback in self._model_callbacks
            }
            trainer_callbacks: list[Callback] = [
                callback
                for callback in self.callbacks
                if id(callback) not in previous_model_callback_ids
            ]
            model_callback_types: set[type[Callback]] = {
                type(callback) for callback in model_callbacks
            }
            trainer_callback_types: set[type[Callback]] = {
                type(callback) for callback in trainer_callbacks
            }
            trainer_callback_types.discard(Callback)
            override_types: set[type[Callback]] = {
                trainer_callback_type
                for model_callback_type in model_callback_types
                for trainer_callback_type in trainer_callback_types
                if issubclass(model_callback_type, trainer_callback_type)
            }
            merged_callbacks: list[Callback] = [
                callback
                for callback in trainer_callbacks
                if type(callback) not in override_types
            ]
            merged_callbacks.extend(model_callbacks)
            merged_callbacks = self._reorder_callbacks(merged_callbacks)
            self._validate_callback_state_keys(merged_callbacks)
            self.callbacks = merged_callbacks
            self._model_callbacks = model_callbacks
        except BaseException:
            self._model_callback_configuration = None
            self.state.finish()
            raise

    def _clear_model_callback_configuration(self, stage: TrainerStage) -> None:
        active_configuration = self._model_callback_configuration
        if active_configuration is not None and active_configuration[0] == stage:
            self._model_callback_configuration = None

    @staticmethod
    def _reorder_callbacks(callbacks: list[Callback]) -> list[Callback]:
        regular_callbacks: list[Callback] = [
            callback
            for callback in callbacks
            if not isinstance(callback, Checkpoint)
        ]
        checkpoint_callbacks: list[Callback] = [
            callback
            for callback in callbacks
            if isinstance(callback, Checkpoint)
        ]
        return regular_callbacks + checkpoint_callbacks

    def _select_dataloader_source(
        self,
        module: Module,
        datamodule: DataModule | None,
        dataloaders: object | None
    ) -> object:
        if dataloaders is not None:
            return dataloaders
        if datamodule is not None:
            return datamodule
        return module

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

    def _safe_len(
        self,
        dataloader: object
    ) -> int | None:
        # Returns the dataloader length, or None both for an absent dataloader
        # and for iterable datasets that define no length, so callers can
        # treat unknown totals uniformly.
        if not isinstance(dataloader, Sized):
            return None
        try:
            return len(dataloader)
        except TypeError:
            return None

    def _request_dataloaders(
        self,
        stage: Literal["train", "val", "test", "predict"]
    ) -> CombinedLoader:
        hook_name: str = f"{stage}_dataloader"
        source: object | None = (
            self._train_dataloader_source
            if stage == "train"
            else self._val_dataloader_source
            if stage == "val"
            else self._test_dataloader_source
            if stage == "test"
            else self._predict_dataloader_source
        )
        if source is None:
            raise RuntimeError(f"No {stage} dataloader source is attached.")
        with self.profiler.profile(f"setup_{stage}_dataloader"):
            if isinstance(source, DataModule):
                datamodule_hook: Callable[[], object] = cast(
                    Callable[[], object],
                    getattr(source, hook_name)
                )
                dataloader_result: object = self._call_datamodule_hook(
                    source,
                    hook_name,
                    datamodule_hook
                )
            elif isinstance(source, Module):
                module_hook: Callable[[], object] = cast(
                    Callable[[], object],
                    getattr(source, hook_name)
                )
                dataloader_result = self._call_module_hook(
                    source,
                    hook_name,
                    module_hook
                )
            else:
                dataloader_result = source
        combined_loader: CombinedLoader = (
            dataloader_result
            if isinstance(dataloader_result, CombinedLoader)
            else CombinedLoader(
                dataloader_result,
                "max_size_cycle" if stage == "train" else "sequential"
            )
        )
        if stage == "predict" and combined_loader.mode != "sequential":
            raise ValueError(
                "Trainer.predict() supports CombinedLoader only in sequential mode."
            )
        processed_dataloaders: list[object] = []
        totals: list[int | None] = []
        for dataloader in combined_loader.flattened:
            if not isinstance(dataloader, torch.utils.data.DataLoader):
                try:
                    iter(cast(Iterable[object], dataloader))
                except TypeError as exception:
                    argument_name: str = (
                        f"{stage}_dataloaders"
                        if self.state.stage == "fit"
                        else "dataloaders"
                    )
                    source_name: str = (
                        f"{type(source).__qualname__}.{hook_name}()"
                        if isinstance(source, Module | DataModule)
                        else f"Trainer.{self.state.stage}({argument_name}=...)"
                    )
                    raise MisconfigurationError(
                        f"{source_name} produced a non-iterable leaf of type "
                        f"{type(dataloader).__qualname__}."
                    ) from exception
            if isinstance(dataloader, torch.utils.data.DataLoader):
                if self.overfit_batches and (
                    stage == "train"
                    or (stage == "val" and self.state.stage == "fit")
                ):
                    dataloader = self._prepare_overfit_dataloader(dataloader, stage)
                if self.use_distributed_sampler and self.strategy.is_distributed:
                    if isinstance(dataloader.dataset, IterableDataset):
                        if is_rank_zero():
                            log.warning(
                                f"Skipping DistributedSampler for {stage} dataloader: "
                                "IterableDataset detected."
                            )
                    elif stage == "predict":
                        dataloader = self._wrap_with_unrepeated_sampler(dataloader)
                    else:
                        dataloader = self._wrap_with_distributed_sampler(
                            dataloader,
                            shuffle=(stage == "train" and not bool(self.overfit_batches))
                        )
                if (
                    self.reload_dataloaders_every_n_epochs > 0
                    and dataloader.persistent_workers
                    and dataloader.pin_memory
                ):
                    warnings.warn(
                        "The combination of `persistent_workers=True`, `pin_memory=True`, and "
                        "`reload_dataloaders_every_n_epochs > 0` can lead to instability due "
                        "to limitations in PyTorch.",
                        UserWarning,
                        stacklevel=3
                    )
            processed_dataloaders.append(dataloader)
            totals.append(self._safe_len(dataloader))
        combined_loader.flattened = processed_dataloaders
        combined_loader.limits = [
            resolve_batch_limit(total, self._batch_limit_for_stage(stage), stage)
            for total in totals
        ]
        self._call_strategy_hook("barrier", self.strategy.barrier)
        if stage == "train":
            self._train_combined_loader: CombinedLoader | None = combined_loader
            self._total_train_batches: int | None = combined_loader._length()
            self._last_train_dataloader_reload_epoch = self.current_epoch
        elif stage == "val":
            self._val_combined_loader: CombinedLoader | None = combined_loader
            self._total_val_batches: list[int | None] | None = totals
            self._last_val_dataloader_reload_epoch = self.current_epoch
        elif stage == "test":
            self._test_combined_loader: CombinedLoader | None = combined_loader
            self._total_test_batches: list[int | None] | None = totals
        else:
            self._predict_combined_loader: CombinedLoader | None = combined_loader
            self._total_predict_batches: list[int | None] | None = totals
        return combined_loader

    def _batch_limit_for_stage(
        self,
        stage: Literal["train", "val", "test", "predict"]
    ) -> int | float:
        if stage == "train":
            return self.limit_train_batches
        if stage == "val":
            return self.limit_val_batches
        if stage == "test":
            return self.limit_test_batches
        return self.limit_predict_batches

    def _ensure_train_dataloader(self) -> None:
        reload_interval: int = self.reload_dataloaders_every_n_epochs
        last_reload_epoch: int | None = self._last_train_dataloader_reload_epoch
        if self._train_combined_loader is not None and (
            reload_interval == 0
            or (
                last_reload_epoch is not None
                and self.current_epoch - last_reload_epoch < reload_interval
            )
        ):
            return
        self._request_dataloaders("train")
        self._configure_validation_schedule(reset_time=False)

    def _ensure_val_dataloader(self) -> None:
        reload_interval: int = self.reload_dataloaders_every_n_epochs
        last_reload_epoch: int | None = self._last_val_dataloader_reload_epoch
        if self._val_combined_loader is not None and (
            reload_interval == 0
            or (
                last_reload_epoch is not None
                and self.current_epoch - last_reload_epoch < reload_interval
            )
        ):
            return
        self._request_dataloaders("val")
        self._configure_validation_schedule(reset_time=False)

    def _configure_validation_schedule(self, reset_time: bool) -> None:
        assert self.fit_loop.epoch_loop is not None
        max_batches: int | None = self._total_train_batches
        module: Module | None = self.module
        source: object | None = self._val_dataloader_source
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
        if validation_enabled and self._total_val_batches is not None:
            validation_enabled = any(
                resolve_batch_limit(total, self.limit_val_batches, "val") != 0
                for total in self._total_val_batches
            )
        self._validation_schedule.configure(
            max_batches,
            validation_enabled,
            reset_time
        )
        self.val_check_batch: int | None = self._validation_schedule.batch_interval

    def _record_dataloader_fetch_time(self, running_stage: RunningStage, duration_seconds: float) -> None:
        # Records the most recent blocking dataloader fetch duration for the
        # stage; consumed by runtime diagnostics such as the profiler callback.
        self._dataloader_fetch_time_seconds[running_stage] = max(0.0, duration_seconds)

    def _reset_progress_bar_metrics(self, running_stage: RunningStage | None = None) -> None:
        if running_stage is None:
            self._progress_bar_metrics_by_stage: dict[RunningStage, dict[str, float]] = {
                "training": {},
                "validating": {},
                "testing": {},
                "predicting": {},
                "sanity_checking": {}
            }
            self._last_progress_bar_stage: RunningStage | None = None
            return
        self._progress_bar_metrics_by_stage[running_stage] = {}
        self._last_progress_bar_stage: RunningStage | None = running_stage

    def _update_progress_bar_metrics(
        self,
        running_stage: RunningStage,
        metrics: dict[str, float]
    ) -> None:
        if not metrics:
            return
        self._progress_bar_metrics_by_stage[running_stage].update(metrics)
        self._last_progress_bar_stage: RunningStage | None = running_stage

    def _update_logged_metrics(self, metrics: dict[str, torch.Tensor]) -> None:
        self._logged_metrics.update(metrics)

    def _preview_logged_metrics(
        self,
        module: Module,
        running_stage: RunningStage
    ) -> None:
        callback_metrics: dict[str, torch.Tensor] = {}
        logged_metrics: dict[str, torch.Tensor] = {}
        progress_bar_metrics: dict[str, float] = {}
        for metric in module.logged_metrics:
            if not metric.on_step:
                continue
            value: torch.Tensor | None = metric.compute_step(self.strategy.root_device)
            if value is None:
                continue
            metric_name: str = metric.output_name(on_step=True)
            if running_stage == "training":
                callback_metrics[metric.base_name()] = value
                callback_metrics[metric_name] = value
            if metric.logger:
                logged_metrics[metric_name] = value
            if metric.progress_bar:
                progress_bar_metrics[metric_name] = self._metric_to_float(value)
        self._callback_metrics.update(callback_metrics)
        if not self.sanity_checking:
            self._update_logged_metrics(logged_metrics)
            self._update_progress_bar_metrics(running_stage, progress_bar_metrics)

    def _get_progress_bar_metrics(self, running_stage: RunningStage) -> dict[str, float]:
        return dict(self._progress_bar_metrics_by_stage[running_stage])

    def _get_dataloader_fetch_time(self, running_stage: RunningStage) -> float | None:
        # Returns the most recent recorded fetch duration for the stage, or
        # None when no fetch has been recorded yet.
        return self._dataloader_fetch_time_seconds.get(running_stage)

    def _record_dataloader_iterator_creation_time(
        self, running_stage: RunningStage, duration_seconds: float
    ) -> None:
        # Records how long dataloader iterator construction took for the
        # stage. Iterator creation spawns worker processes, so this duration
        # exposes worker startup cost separately from per-batch fetch time.
        self._dataloader_iterator_creation_time_seconds[running_stage] = max(0.0, duration_seconds)

    def _consume_dataloader_iterator_creation_time(self, running_stage: RunningStage) -> float | None:
        # Returns and clears the recorded iterator-construction duration for
        # the stage. Consumption is destructive so the one-time startup cost
        # cannot be attributed to more than one measurement window.
        return self._dataloader_iterator_creation_time_seconds.pop(running_stage, None)

    def _connect_loops(self) -> None:
        # Attaches this trainer to every loop, giving each loop access to the
        # shared collaborators (strategy, state, callbacks, dataloaders,
        # metrics) through its trainer property.
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
        self.strategy.checkpoint.connect_optimizers(self.strategy.optimizers)
        return module, connected_optimizers, connected_lr_schedulers

    def _call_on_validation_model_zero_grad(self, module: Module) -> None:
        self._call_module_hook(
            module,
            "on_validation_model_zero_grad",
            module.on_validation_model_zero_grad
        )

    def _resolve_accelerator(self, accelerator: str | Accelerator) -> Accelerator:
        # Resolves the accelerator argument: a live instance passes through
        # unchanged, a backend name constructs the matching accelerator, and
        # "auto" prefers CUDA, then MPS, then CPU by availability.
        resolved_accelerator: Accelerator
        if isinstance(accelerator, Accelerator):
            resolved_accelerator = accelerator
        elif not isinstance(accelerator, str):
            raise TypeError(
                f"`accelerator` must be a string or Accelerator instance, got {accelerator!r}."
            )
        else:
            normalized_accelerator: str = accelerator.lower()
            match normalized_accelerator:
                case "cuda":
                    resolved_accelerator = CUDAAccelerator()
                case "mps":
                    resolved_accelerator = MPSAccelerator()
                case "cpu":
                    resolved_accelerator = CPUAccelerator()
                case "gpu":
                    if MPSAccelerator.is_available():
                        resolved_accelerator = MPSAccelerator()
                    elif CUDAAccelerator.is_available():
                        resolved_accelerator = CUDAAccelerator()
                    else:
                        raise MisconfigurationError(
                            "No supported GPU accelerator is available."
                        )
                case "auto":
                    if CUDAAccelerator.is_available():
                        resolved_accelerator = CUDAAccelerator()
                    elif MPSAccelerator.is_available():
                        resolved_accelerator = MPSAccelerator()
                    else:
                        resolved_accelerator = CPUAccelerator()
                case _:
                    raise ValueError(
                        f"Unknown accelerator {accelerator!r}. Available choices are "
                        "'auto', 'gpu', 'cpu', 'cuda', and 'mps'."
                    )
        if not type(resolved_accelerator).is_available():
            available_accelerators: list[str] = ["cpu"]
            if CUDAAccelerator.is_available():
                available_accelerators.append("cuda")
            if MPSAccelerator.is_available():
                available_accelerators.append("mps")
            raise MisconfigurationError(
                f"{type(resolved_accelerator).__name__} is not available on this system. "
                f"Available accelerators: {available_accelerators}."
            )
        return resolved_accelerator

    def _resolve_strategy(
        self,
        strategy: str | Strategy,
        accelerator: Accelerator,
        parallel_devices: list[torch.device],
        num_nodes: int
    ) -> Strategy:
        # Resolves the strategy argument: a live instance passes through
        # unchanged, a strategy name constructs the matching strategy, and
        # "auto" selects distributed data parallel only for multi-device CUDA
        # configurations, defaulting to single-device execution otherwise.
        resolved_strategy: Strategy
        if isinstance(strategy, Strategy):
            resolved_strategy = strategy
        elif not isinstance(strategy, str):
            raise TypeError(
                f"`strategy` must be a string or Strategy instance, got {strategy!r}."
            )
        else:
            normalized_strategy: str = strategy.lower()
            match normalized_strategy:
                case "ddp":
                    resolved_strategy = DDPStrategy()
                case "ddp_spawn":
                    resolved_strategy = DDPStrategy(start_method="spawn")
                case "ddp_fork" | "ddp_notebook":
                    resolved_strategy = DDPStrategy(start_method="fork")
                case "single_device":
                    resolved_strategy = SingleDeviceStrategy()
                case "auto":
                    if len(parallel_devices) > 1 or num_nodes > 1:
                        resolved_strategy = DDPStrategy()
                    else:
                        resolved_strategy = SingleDeviceStrategy()
                case _:
                    raise ValueError(
                        f"Unknown strategy {strategy!r}. Available choices are "
                        "'auto', 'single_device', 'ddp', 'ddp_spawn', "
                        "'ddp_fork', and 'ddp_notebook'."
                    )
        requires_distributed_strategy: bool = (
            len(parallel_devices) > 1 or num_nodes > 1
        )
        if isinstance(accelerator, MPSAccelerator) and resolved_strategy.is_distributed:
            raise MisconfigurationError(
                "Distributed strategies are not supported with the MPS accelerator. "
                "Use accelerator='cpu' or a single-device strategy."
            )
        if requires_distributed_strategy and not resolved_strategy.is_distributed:
            raise MisconfigurationError(
                f"{type(resolved_strategy).__name__} cannot manage num_devices="
                f"{len(parallel_devices)} and num_nodes={num_nodes}. Use strategy='ddp'."
            )
        return resolved_strategy

    def _resolve_device(self) -> torch.device:
        # Resolves the concrete device for this process. Distributed CUDA
        # execution derives the device index from the launcher-provided
        # LOCAL_RANK so each process binds its own GPU; single-process CUDA
        # uses device zero, and the remaining backends have one device each.
        return self.strategy.root_device

    def _wrap_with_distributed_sampler(
        self,
        dataloader: torch.utils.data.DataLoader,
        shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        # Rebuilds the dataloader around a DistributedSampler while carrying
        # over every construction parameter the DataLoader exposes, so the
        # wrapped loader differs from the original only in its sampling.
        sampler_kwargs: dict[str, int] | None = self.distributed_sampler_kwargs
        num_replicas: int | None = (
            sampler_kwargs["num_replicas"] if sampler_kwargs is not None else None
        )
        rank: int | None = sampler_kwargs["rank"] if sampler_kwargs is not None else None
        sampler: DistributedSampler = DistributedSampler(
            dataloader.dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        return self._replace_dataloader_sampler(dataloader, sampler)

    def _wrap_with_unrepeated_sampler(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> torch.utils.data.DataLoader:
        # Rebuilds the prediction dataloader around the unrepeated sampler,
        # which partitions without padding so no rank emits duplicate
        # predictions; all other construction parameters carry over unchanged.
        sampler_kwargs: dict[str, int] | None = self.distributed_sampler_kwargs
        num_replicas: int | None = (
            sampler_kwargs["num_replicas"] if sampler_kwargs is not None else None
        )
        rank: int | None = sampler_kwargs["rank"] if sampler_kwargs is not None else None
        sampler: _UnrepeatedDistributedSampler = _UnrepeatedDistributedSampler(
            dataloader.dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
        )
        return self._replace_dataloader_sampler(dataloader, sampler)

    def _prepare_overfit_dataloader(
        self,
        dataloader: torch.utils.data.DataLoader,
        stage: str
    ) -> torch.utils.data.DataLoader:
        if isinstance(dataloader.dataset, IterableDataset):
            return dataloader
        active_sampler: object = dataloader.sampler
        if dataloader.batch_size is None and dataloader.batch_sampler is not None:
            active_sampler = getattr(dataloader.batch_sampler, "sampler", dataloader.batch_sampler)
        if isinstance(active_sampler, SequentialSampler):
            return dataloader
        warnings.warn(
            f"`overfit_batches` requires deterministic {stage} batches; "
            f"disabling {stage} dataloader shuffling.",
            UserWarning,
            stacklevel=3
        )
        sampler: SequentialSampler = SequentialSampler(
            cast(Sized, dataloader.dataset)
        )
        return self._replace_dataloader_sampler(dataloader, sampler)

    def _replace_dataloader_sampler(
        self,
        dataloader: torch.utils.data.DataLoader,
        sampler: Sampler
    ) -> torch.utils.data.DataLoader:
        if dataloader.batch_size is None and dataloader.batch_sampler is not None:
            batch_sampler: object = dataloader.batch_sampler
            if not isinstance(batch_sampler, BatchSampler):
                raise MisconfigurationError(
                    "SyntheticMind cannot inject a sampler into a custom batch sampler that "
                    "does not subclass torch.utils.data.BatchSampler."
                )
            try:
                replaced_batch_sampler: BatchSampler = type(batch_sampler)(
                    sampler,
                    batch_sampler.batch_size,
                    batch_sampler.drop_last
                )
            except TypeError as exception:
                raise MisconfigurationError(
                    f"SyntheticMind cannot reconstruct custom batch sampler "
                    f"{type(batch_sampler).__qualname__}. Its constructor must accept "
                    "sampler, batch_size, and drop_last."
                ) from exception
            return self._reinstantiate_dataloader(
                dataloader,
                {
                    "dataset": dataloader.dataset,
                    "batch_sampler": replaced_batch_sampler,
                    "num_workers": dataloader.num_workers,
                    "collate_fn": dataloader.collate_fn,
                    "pin_memory": dataloader.pin_memory,
                    "timeout": dataloader.timeout,
                    "worker_init_fn": dataloader.worker_init_fn,
                    "multiprocessing_context": getattr(
                        dataloader,
                        "multiprocessing_context",
                        None
                    ),
                    "generator": getattr(dataloader, "generator", None),
                    "prefetch_factor": dataloader.prefetch_factor,
                    "persistent_workers": dataloader.persistent_workers,
                    "pin_memory_device": getattr(
                        dataloader,
                        "pin_memory_device",
                        ""
                    ),
                    "in_order": getattr(dataloader, "in_order", True)
                }
            )
        return self._reinstantiate_dataloader(
            dataloader,
            {
                "dataset": dataloader.dataset,
                "batch_size": dataloader.batch_size,
                "sampler": sampler,
                "drop_last": dataloader.drop_last,
                "num_workers": dataloader.num_workers,
                "collate_fn": dataloader.collate_fn,
                "pin_memory": dataloader.pin_memory,
                "timeout": dataloader.timeout,
                "worker_init_fn": dataloader.worker_init_fn,
                "multiprocessing_context": getattr(
                    dataloader,
                    "multiprocessing_context",
                    None
                ),
                "generator": getattr(dataloader, "generator", None),
                "prefetch_factor": dataloader.prefetch_factor,
                "persistent_workers": dataloader.persistent_workers,
                "pin_memory_device": getattr(
                    dataloader,
                    "pin_memory_device",
                    ""
                ),
                "in_order": getattr(dataloader, "in_order", True)
            }
        )

    def _reinstantiate_dataloader(
        self,
        dataloader: torch.utils.data.DataLoader,
        constructor_kwargs: dict[str, object]
    ) -> torch.utils.data.DataLoader:
        constructor: type[torch.utils.data.DataLoader] = type(dataloader)
        parameters: dict[str, inspect.Parameter] = dict(
            inspect.signature(constructor).parameters
        )
        base_constructor: type[object]
        for base_constructor in constructor.__mro__[1:]:
            if not issubclass(base_constructor, torch.utils.data.DataLoader):
                continue
            base_name: str
            base_parameter: inspect.Parameter
            for base_name, base_parameter in inspect.signature(
                base_constructor
            ).parameters.items():
                parameters.setdefault(base_name, base_parameter)
        dataloader_parameters: set[str] = set(
            inspect.signature(torch.utils.data.DataLoader).parameters
        )
        accepts_kwargs: bool = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        for name, parameter in parameters.items():
            if (
                name not in constructor_kwargs
                and parameter.kind
                in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY
                }
                and hasattr(dataloader, name)
                and (
                    parameter.default is inspect.Parameter.empty
                    or name not in dataloader_parameters
                )
            ):
                constructor_kwargs[name] = getattr(dataloader, name)
        missing: list[str] = [
            name
            for name, parameter in parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY
            }
            and name not in constructor_kwargs
        ]
        if missing:
            raise MisconfigurationError(
                f"SyntheticMind cannot reconstruct {constructor.__qualname__} with a "
                f"replacement sampler because required constructor values {missing!r} "
                "are not available as dataloader attributes."
            )
        positional_args: list[object] = []
        name: str
        parameter: inspect.Parameter
        for name, parameter in parameters.items():
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                positional_args.append(constructor_kwargs.pop(name))
        if not accepts_kwargs:
            unexpected: list[str] = [
                name
                for name in constructor_kwargs
                if name not in parameters
            ]
            if unexpected:
                raise MisconfigurationError(
                    f"SyntheticMind cannot reconstruct {constructor.__qualname__} with a "
                    f"replacement sampler because its constructor does not accept "
                    f"DataLoader parameters {unexpected!r}."
                )
        try:
            constructor_callable: Callable[..., object] = cast(
                Callable[..., object],
                constructor
            )
            rebuilt: object = constructor_callable(
                *positional_args,
                **constructor_kwargs
            )
        except TypeError as exception:
            raise MisconfigurationError(
                f"SyntheticMind could not reconstruct {constructor.__qualname__} with "
                "the replacement sampler. Its constructor must preserve the standard "
                "DataLoader parameter contract."
            ) from exception
        if not isinstance(rebuilt, torch.utils.data.DataLoader):
            raise TypeError(
                f"Reconstructing {constructor.__qualname__} returned "
                f"{type(rebuilt).__qualname__}, not a DataLoader."
            )
        return rebuilt

    def _sanity_check_validation(self, module: Module) -> None:
        # Runs a bounded validation pass before training to surface evaluation
        # defects immediately instead of after the first training epoch. The
        # validate loop's batch limit is temporarily replaced and restored in
        # a finally block, and every trace of the sanity pass (epoch metrics,
        # the module's logging buffer, callback metrics) is discarded so
        # sanity results cannot influence monitoring or checkpointing.
        self.sanity_checking = True
        module._running_stage = "sanity_checking"

        try:
            self._ensure_val_dataloader()
            self._call_callback_hooks(module, "on_sanity_check_start")
            self.validate_loop.run()
            if not self._validation_schedule.is_epoch_eligible(self.current_epoch):
                self._last_val_dataloader_reload_epoch = None
            self._validation_schedule.mark_validation_end()
            self.validate_loop.epoch_metrics = {}
            module._reset_logging_state()
            self._callback_metrics.clear()
            self._logged_metrics.clear()
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
        # Restores a full checkpoint onto the live objects: model, optimizer,
        # and scheduler state through CheckpointState, the progress counters
        # into the trainer state and module mirrors, and the mid-epoch resume
        # markers (completed batches and epoch-start random state) into the
        # training epoch loop. The module and callback load hooks fire last so
        # they observe the fully restored payload.
        loaded_checkpoint: CheckpointDict = (
            self.strategy.checkpoint.load(path, weights_only)
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
        self.strategy.checkpoint.restore(
            loaded_checkpoint,
            strict=module.strict_loading,
            restore_optimizers=self.state.stage == "fit"
        )
        epoch: int
        global_step: int
        completed_batches: int
        total_batch_idx: int
        epoch_rng_state: CheckpointDict | None
        epoch, global_step, completed_batches, total_batch_idx, epoch_rng_state = CheckpointState.restore(
            checkpoint=loaded_checkpoint,
            model=module,
            optimizers=(self.optimizers if self.state.stage == "fit" else []),
            scheduler=(self.scheduler if self.state.stage == "fit" else None),
            schedulers=(self.schedulers if self.state.stage == "fit" else []),
            callbacks=self.callbacks,
            datamodule=self.datamodule,
            precision_plugin=self.precision_plugin,
            restore_datamodule_state=restore_datamodule_state,
            strict=module.strict_loading,
            restore_model=False,
            restore_optimizers=False
        )
        combined_loader_states: list[dict[str, object]] = (
            CheckpointState.combined_loader_states(loaded_checkpoint)
            if self.state.stage == "fit"
            else []
        )
        if combined_loader_states:
            self._last_train_dataloader_states: list[dict[str, object]] = (
                combined_loader_states
            )
            self._train_dataloader_state_active: bool = True
            self._last_train_dataloader_batches_completed: int = completed_batches
            self._last_train_total_batch_idx: int = total_batch_idx
            if self._train_combined_loader is not None:
                self._train_combined_loader._load_state_dicts(
                    combined_loader_states
                )
        self.state.load_state_dict({"current_epoch": epoch, "global_step": global_step})
        module._current_epoch = epoch
        module._global_step = global_step
        if self.fit_loop.epoch_loop is not None:
            loader_state_restored: bool = bool(
                combined_loader_states
                and self._train_combined_loader is not None
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
        if self._ckpt_path_user_managed:
            if not ckpt_path_provided:
                ckpt_path = self._ckpt_path
            elif ckpt_path != self._ckpt_path:
                if self.is_global_zero:
                    warnings.warn(
                        f"`trainer.ckpt_path={self._ckpt_path!r}` is overridden by "
                        f"`Trainer.{stage}(ckpt_path={ckpt_path!r})`.",
                        UserWarning,
                        stacklevel=3
                    )
                self._ckpt_path_user_managed = False
        if ckpt_path is None and not model_provided and stage != "fit":
            ckpt_path = "best"
            if self.is_global_zero:
                warnings.warn(
                    f"`Trainer.{stage}(ckpt_path=None)` was called without a model; "
                    "the best checkpoint from the previous fit will be loaded. Pass "
                    "a model to use its current weights, or set `ckpt_path` explicitly.",
                    UserWarning,
                    stacklevel=3
                )
        if ckpt_path is None:
            self._ckpt_path = None
            return None
        if isinstance(ckpt_path, str) and ckpt_path == "best":
            model_checkpoint_callbacks: list[ModelCheckpoint] = [
                callback
                for callback in self.checkpoint_callbacks
                if isinstance(callback, ModelCheckpoint)
            ]
            if len(model_checkpoint_callbacks) > 1 and self.is_global_zero:
                warnings.warn(
                    f"`Trainer.{stage}(ckpt_path='best')` received multiple "
                    "ModelCheckpoint callbacks; the first callback owns best-checkpoint "
                    "selection.",
                    UserWarning,
                    stacklevel=3
                )
            if model_checkpoint_callbacks:
                best_model_path: str = model_checkpoint_callbacks[0].best_model_path
                if best_model_path:
                    self._ckpt_path = best_model_path
                    return best_model_path
            if self.fast_dev_run:
                raise MisconfigurationError(
                    f"`Trainer.{stage}(ckpt_path='best')` is unavailable with "
                    "`fast_dev_run=True`; pass an exact checkpoint path."
                )
            raise MisconfigurationError(
                f"`Trainer.{stage}(ckpt_path='best')` requires the first "
                "ModelCheckpoint callback to have a saved `best_model_path`."
            )
        if isinstance(ckpt_path, str) and ckpt_path == "last":
            candidates: list[str] = []
            for callback in self.checkpoint_callbacks:
                if not isinstance(callback, ModelCheckpoint):
                    continue
                if callback.last_model_path:
                    last_model_path: str = callback.last_model_path
                    if self.strategy.checkpoint.is_checkpoint_path(last_model_path):
                        candidates.append(last_model_path)
                directory: str = callback._resolve_dirpath(self)
                for candidate in list_directory(directory):
                    candidate_name: str = basename(candidate)
                    if (
                        candidate_name.startswith("last")
                        and candidate_name.endswith(".ckpt")
                        and self.strategy.checkpoint.is_checkpoint_path(candidate)
                    ):
                        candidates.append(candidate)
            unique_candidates: set[str] = set(candidates)
            if not unique_candidates:
                self._ckpt_path = None
                if self.is_global_zero:
                    warnings.warn(
                        f"`Trainer.{stage}(ckpt_path='last')` found no last checkpoint; "
                        "no checkpoint will be loaded. Configure "
                        "`ModelCheckpoint(save_last=True)` to enable it.",
                        UserWarning,
                        stacklevel=3
                    )
                return None
            path: str = max(
                unique_candidates,
                key=modified_time
            )
            self._ckpt_path = path
            return path
        if isinstance(ckpt_path, str) and ckpt_path in {"hpc", "registry"}:
            raise MisconfigurationError(
                f"`ckpt_path={ckpt_path!r}` is unavailable without its corresponding integration."
            )
        path: str = normalize_path(ckpt_path)
        if not self.strategy.checkpoint.is_checkpoint_path(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        self._ckpt_path = path
        return path

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
        combined_loader_states: list[dict[str, object]] = []
        if (
            is_mid_epoch
            and self._train_combined_loader is not None
        ):
            combined_loader_states: list[dict[str, object]] = (
                epoch_loop._completed_loader_states
                if epoch_loop is not None
                and epoch_loop._use_completed_loader_states
                else self._train_combined_loader._state_dicts()
            )
        elif self._train_dataloader_state_active:
            combined_loader_states: list[dict[str, object]] = (
                self._last_train_dataloader_states
            )
        checkpoint_completed_batches: int = (
            completed_batches
            if is_mid_epoch
            else self._last_train_dataloader_batches_completed
            if self._train_dataloader_state_active
            else 0
        )
        checkpoint_total_batch_idx: int = (
            total_batch_idx
            if total_batch_idx >= 0
            else self._last_train_total_batch_idx
            if self._train_dataloader_state_active
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
                        self.strategy.checkpoint.optimizer_state(optimizer)
                    )
                )
        checkpoint: CheckpointDict = CheckpointState.build(
            model=module,
            optimizers=self.optimizers,
            scheduler=self.scheduler,
            schedulers=self.schedulers,
            epoch=epoch,
            global_step=self.state.global_step,
            callbacks=self.callbacks,
            datamodule=self.datamodule,
            completed_batches=checkpoint_completed_batches,
            total_batch_idx=checkpoint_total_batch_idx,
            epoch_rng_state=epoch_rng_state if is_mid_epoch else None,
            combined_loader_states=combined_loader_states,
            strategy_name=type(self.strategy).__name__,
            world_size=self.world_size,
            precision_plugin=self.precision_plugin,
            weights_only=weights_only,
            model_state_dict=cast(
                CheckpointValue,
                self.strategy.checkpoint.model_state_dict()
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
            self.strategy.checkpoint.save(
                checkpoint,
                filepath,
                storage_options,
                self.global_rank
            )
            if synchronize:
                if self._profiler_is_setup:
                    self._call_strategy_hook("barrier", self.strategy.barrier)
                else:
                    self.strategy.barrier()

    def _handle_exception(
        self,
        module: Module,
        datamodule: DataModule | None,
        exception: BaseException
    ) -> Never:
        # Central exception path for every entry point: marks the trainer
        # state interrupted, offers the exception to the callback, datamodule,
        # and strategy exception hooks so they can persist state or release
        # resources, and then re-raises. KeyboardInterrupt re-raises through
        # the bare raise so its traceback is preserved unchanged.
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
            self.strategy.checkpoint.teardown,
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
                ("train", self._train_combined_loader),
                ("val", self._val_combined_loader)
            )
            if stage == "fit"
            else (
                ("val", self._val_combined_loader),
            )
            if stage == "validate"
            else (
                (
                    stage,
                    self._test_combined_loader
                    if stage == "test"
                    else self._predict_combined_loader
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

    def _validate_callback_state_keys(
        self,
        callback_group: Iterable[Callback] | None = None
    ) -> None:
        # Enforces state-key uniqueness across stateful callbacks before
        # fitting. Only callbacks that override the state methods participate,
        # so stateless callbacks of the same class may coexist freely, while
        # two stateful instances sharing a key would silently overwrite each
        # other inside checkpoints and are rejected instead.
        seen: dict[str, str] = {}
        callbacks: Iterable[Callback] = (
            self.callbacks if callback_group is None else callback_group
        )
        for cb in callbacks:
            if is_overridden("state_dict", cb, Callback) or is_overridden("load_state_dict", cb, Callback):
                key: str = cb.state_key
                if key in seen:
                    raise MisconfigurationError(
                        f"Duplicate callback state_key {key!r} between "
                        f"{seen[key]} and {cb!r}. "
                        f"Each stateful callback must have a unique state_key."
                    )
                seen[key] = repr(cb)

    def _resolve_total_scheduler_steps(
        self,
        scheduler_definitions: list[LRSchedulerConfig]
    ) -> int:
        # Computes the horizon passed to the scheduler builder. When any
        # configured schedule advances per step, the horizon is the number of
        # optimizer steps across the whole run: the limited per-epoch batch
        # count divided by the accumulation factor, times the epoch budget. An
        # unknown batch total with a step-interval schedule is a configuration
        # error because the schedule cannot be sized. Epoch-interval schedules
        # use the epoch budget directly.
        config_schedulers: list[LRSchedulerConfig] = list(scheduler_definitions)
        if not config_schedulers:
            raise MisconfigurationError(
                "Expected at least one scheduler configuration when resolving scheduler steps."
            )
        if any(scheduler_config.interval == "step" for scheduler_config in config_schedulers):
            assert self.fit_loop.epoch_loop is not None
            limited_batches: int | None = self._total_train_batches
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

    @staticmethod
    def _metric_to_float(value: torch.Tensor | float) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Logged metric tensors must contain exactly one element.")
            return float(value.detach().cpu().item())
        return float(value)

    def _log_metrics(
        self,
        metrics: Mapping[str, torch.Tensor | float],
        step: int
    ) -> None:
        scalar_metrics: dict[str, float] = {
            name: self._metric_to_float(value)
            for name, value in metrics.items()
        }
        for configured_logger in self._loggers:
            configured_logger.log_metrics(scalar_metrics, step=step)

    def _log_hyperparams(self, params: HyperparameterDict) -> None:
        for configured_logger in self._loggers:
            configured_logger.log_hyperparams(params)

    def _autolog_hyperparameters(
        self,
        module: Module,
        datamodule: DataModule | None
    ) -> None:
        if (
            not self.enable_autolog_hparams
            or not self.loggers
            or self.fast_dev_run
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
            inconsistent_keys: list[object] = []
            key: object
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
            merged_hyperparameters: HyperparameterDict | None = (
                module_hyperparameters
            )
        elif datamodule_hyperparameters is not None:
            merged_hyperparameters: HyperparameterDict | None = (
                datamodule_hyperparameters
            )
        if merged_hyperparameters is None:
            return
        filtered_hyperparameters: HyperparameterDict = {
            key: value
            for key, value in merged_hyperparameters.items()
            if key != "_class_path"
        }
        if is_rank_zero():
            self._log_hyperparams(filtered_hyperparameters)

    def _hyperparameters_equal(self, first: object, second: object) -> bool:
        if type(first) is not type(second):
            return False
        if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
            return first is second
        if first is second:
            return True
        try:
            comparison: object = first == second
        except Exception:
            return False
        if isinstance(comparison, bool):
            return comparison
        try:
            return bool(comparison)
        except (RuntimeError, TypeError, ValueError):
            return False

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
        return self._ckpt_path

    @ckpt_path.setter
    def ckpt_path(self, ckpt_path: str | Path | None) -> None:
        if ckpt_path is not None and not isinstance(ckpt_path, str | Path):
            raise TypeError("`ckpt_path` must be a string, Path, or None.")
        self._ckpt_path = ckpt_path
        self._ckpt_path_user_managed = ckpt_path is not None

    @property
    def logger(self) -> Logger | None:
        return self._loggers[0] if self._loggers else None

    @logger.setter
    def logger(self, logger: Logger | None) -> None:
        if logger is not None and not isinstance(logger, Logger):
            raise TypeError("`logger` must be a Logger or None.")
        self._loggers = [] if logger is None else [logger]

    @property
    def loggers(self) -> list[Logger]:
        return self._loggers

    @loggers.setter
    def loggers(self, loggers: list[Logger] | None) -> None:
        resolved_loggers: list[Logger] = [] if loggers is None else list(loggers)
        if not all(isinstance(logger, Logger) for logger in resolved_loggers):
            raise TypeError("`loggers` must contain only Logger instances.")
        self._loggers = resolved_loggers

    @property
    def default_root_dir(self) -> str:
        return normalize_path(self._default_root_dir)

    @property
    def log_dir(self) -> str:
        logger: Logger | None = self.logger
        if isinstance(logger, CSVLogger | TensorBoardLogger):
            dirpath: str = logger.log_dir
        elif logger is not None:
            dirpath: str = logger.save_dir
        else:
            dirpath: str = self.default_root_dir
        return self.strategy.broadcast(dirpath)

    @property
    def current_epoch(self) -> int:
        # Zero-based epoch counter mirrored from the trainer state.
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
        # Count of optimizer steps executed, mirrored from the trainer state.
        return self.state.global_step

    @property
    def callback_metrics(self) -> dict[str, torch.Tensor]:
        # Copy of the metric mapping monitored by callbacks; returned as a new
        # dictionary so callers cannot mutate the trainer's internal state.
        return dict(self._callback_metrics)

    @property
    def logged_metrics(self) -> dict[str, torch.Tensor]:
        # Reduced epoch metrics from the most recent validation pass, exposed
        # for inspection after validate() or a fit-scheduled validation.
        return dict(self._logged_metrics)

    @property
    def progress_bar_metrics(self) -> dict[str, float]:
        running_stage: RunningStage | None = self.state.running_stage
        if running_stage == "sanity_checking":
            return {}
        resolved_stage: RunningStage | None = (
            running_stage if running_stage is not None else self._last_progress_bar_stage
        )
        if resolved_stage is None:
            return {}
        return self._get_progress_bar_metrics(resolved_stage)

    @property
    def interrupted(self) -> bool:
        # Whether the most recent run terminated through an interruption.
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
        if self._train_combined_loader is None:
            return None
        return self._train_combined_loader.iterables

    @property
    def val_dataloaders(self) -> object | None:
        if self._val_combined_loader is None:
            return None
        return self._val_combined_loader.iterables

    @property
    def test_dataloaders(self) -> object | None:
        if self._test_combined_loader is None:
            return None
        return self._test_combined_loader.iterables

    @property
    def predict_dataloaders(self) -> object | None:
        if self._predict_combined_loader is None:
            return None
        return self._predict_combined_loader.iterables

    @property
    def num_training_batches(self) -> int | float:
        return (
            float("inf")
            if self._total_train_batches is None
            else self._total_train_batches
        )

    @property
    def num_val_batches(self) -> list[int | float]:
        if self._total_val_batches is None:
            return []
        resolved_batches: list[int | float] = []
        for total_batches in self._total_val_batches:
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
        if self._total_test_batches is None:
            return []
        resolved_batches: list[int | float] = []
        for total_batches in self._total_test_batches:
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
        if self._total_predict_batches is None:
            return []
        resolved_batches: list[int | float] = []
        for total_batches in self._total_predict_batches:
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
        # Estimated total optimizer steps for the configured run, derived from
        # the limited batch count and the accumulation factor. Returns
        # infinity when the batch total is unknown, which callers such as
        # one-cycle schedule sizing must handle explicitly.
        if self.fit_loop.epoch_loop is None:
            return float("inf")
        limited_batches: int | None = self._total_train_batches
        if limited_batches is None:
            return float("inf") if self.max_steps == -1 else self.max_steps
        accum: int = self.fit_loop.epoch_loop.accumulate_grad_batches
        stepping_per_epoch: int = math.ceil(limited_batches / accum)
        if self.max_epochs == -1:
            return float("inf") if self.max_steps == -1 else self.max_steps
        estimated_steps: int = stepping_per_epoch * self.max_epochs
        return min(estimated_steps, self.max_steps) if self.max_steps != -1 else estimated_steps

    def __repr__(self) -> str:
        # Compact configuration line for logs and interactive debugging.
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
