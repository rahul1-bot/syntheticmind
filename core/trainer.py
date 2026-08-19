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
import warnings
from collections.abc import Callable, Iterator, Sized
from datetime import timedelta
from pathlib import Path
from typing import Never, cast

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
from syntheticmind.core.datamodule import DataModule
from syntheticmind.core.hooks import ModelHooks
from syntheticmind.core.module import Module
from syntheticmind.core.optimizer import (
    LRSchedulerConfig,
    SyntheticOptimizer,
    _SchedulerInstance,
    _init_optimizers_and_lr_schedulers,
)
from syntheticmind.loggers.logger import Logger, _NoOpLogger
from syntheticmind.loops.batch_limits import (
    normalize_batch_limit,
    normalize_fast_dev_run,
    normalize_sanity_steps,
)
from syntheticmind.loops.evaluation_loop import EvaluationLoop
from syntheticmind.loops.fit_loop import FitLoop
from syntheticmind.loops.prediction_loop import PredictionLoop
from syntheticmind.loops.training_epoch_loop import TrainingEpochLoop
from syntheticmind.loops.validation_schedule import ValidationSchedule
from syntheticmind.precision.precision import Precision
from syntheticmind.precision.resolver import PrecisionResolver
from syntheticmind.state.checkpoint_state import CheckpointState
from syntheticmind.state.trainer_state import TrainerState
from syntheticmind.strategies.ddp import DDPStrategy
from syntheticmind.strategies.single_device import SingleDeviceStrategy
from syntheticmind.strategies.strategy import Strategy
from syntheticmind.utilities.checkpoint import load_checkpoint
from syntheticmind.utilities.distributed import is_rank_zero
from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.model_helpers import ModuleMode, is_overridden
from syntheticmind.utilities.optimizers import _optimizers_to_device
from syntheticmind.utilities.seed import SeedManager
from syntheticmind.utilities.types import (
    CheckpointDict,
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
        if self.shuffle:
            g: torch.Generator = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices: list[int] = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices: list[int] = list(range(len(self.dataset)))
        indices: list[int] = indices[self.rank::self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        # Length of this rank's shard under the strided partition, which may
        # differ across ranks by one when the dataset size is not divisible by
        # the replica count.
        return max(0, math.ceil((len(self.dataset) - self.rank) / self.num_replicas))


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
        logger: Logger | None = None,
        callbacks: list[Callback] | None = None,
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
        log_every_n_steps: int = 10,
        enable_progress_bar: bool = True,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
        deterministic: bool = False,
        use_distributed_sampler: bool = True,
        plugins: Precision | list[Precision] | None = None,
        sync_batchnorm: bool = False,
        reload_dataloaders_every_n_epochs: int = 0,
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
        effective_logger: Logger | None = logger
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
            if effective_logger is not None:
                effective_logger = _NoOpLogger()
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
        self.log_every_n_steps: int = log_every_n_steps
        self.num_sanity_val_steps: int | float = resolved_num_sanity_val_steps
        self.fast_dev_run: int | bool = resolved_fast_dev_run
        self.overfit_batches: int | float = resolved_overfit_batches
        self.use_distributed_sampler: bool = use_distributed_sampler
        self.sync_batchnorm: bool = sync_batchnorm
        self.deterministic: bool = deterministic
        self.resume_from_checkpoint: Path | None = (
            Path(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        )

        if accumulate_grad_batches < 1:
            raise MisconfigurationError("accumulate_grad_batches must be >= 1")
        if log_every_n_steps < 1:
            raise MisconfigurationError("log_every_n_steps must be >= 1")

        if seed is not None:
            SeedManager.seed_everything(seed, deterministic=deterministic)

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
        self._precision_plugin: Precision = PrecisionResolver.resolve(
            precision,
            self.accelerator.name(),
            plugins,
            self.strategy.configured_precision_plugin
        )
        self.strategy._attach_precision_plugin(self._precision_plugin)

        self.callbacks: list[Callback] = list(callbacks) if callbacks is not None else []
        from syntheticmind.callbacks.progress_bar import ProgressBarCallback

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
        if not enable_progress_bar and progress_callbacks:
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

        if enable_progress_bar:
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

        self.logger: Logger | None = effective_logger
        self.state: TrainerState = TrainerState()
        self._sanity_checking: bool = False
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
        self.fit_loop.epoch_loop: TrainingEpochLoop | None = TrainingEpochLoop(
            min_steps=min_steps,
            max_steps=effective_max_steps,
            accumulate_grad_batches=accumulate_grad_batches,
            precision=self.precision,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm=self.gradient_clip_algorithm,
            log_every_n_steps=log_every_n_steps
        )
        self.validate_loop: EvaluationLoop = EvaluationLoop(stage="validate")
        self.test_loop: EvaluationLoop = EvaluationLoop(stage="test")
        self.predict_loop: PredictionLoop = PredictionLoop()

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
        self.train_dataloader: torch.utils.data.DataLoader | None = None
        self.val_dataloader: torch.utils.data.DataLoader | None = None
        self.test_dataloader: torch.utils.data.DataLoader | None = None
        self.predict_dataloader: torch.utils.data.DataLoader | None = None
        self._total_train_batches: int | None = None
        self._total_val_batches: int | None = None
        self._total_test_batches: int | None = None
        self._total_predict_batches: int | None = None
        self.reload_dataloaders_every_n_epochs: int = reload_dataloaders_every_n_epochs
        self._callback_metrics: dict[str, float] = {}
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

    def fit(
        self,
        module: Module,
        datamodule: DataModule,
        ckpt_path: str | Path | None = None
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
        launcher = self.strategy.launcher
        if launcher is not None and launcher.should_launch:
            launcher.launch(
                "fit",
                self,
                module,
                datamodule,
                {"ckpt_path": ckpt_path}
            )
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
        self._reset_progress_bar_metrics()
        self._total_train_batches: int | None = None
        self._total_val_batches: int | None = None
        self.fit_loop.should_stop: bool = False
        if self.fit_loop.epoch_loop is not None:
            self.fit_loop.epoch_loop.reset_on_run()
        # Bind the run participants and zero the module's progress mirrors.
        # The trainer back-reference is what powers module.log, the optimizer
        # and scheduler accessors, and gradient clipping inside the step
        # methods.
        self.module: Module | None = module
        self.datamodule: DataModule | None = datamodule
        module._trainer: Trainer | None = self
        module._current_epoch: int = 0
        module._global_step: int = 0
        module_mode: ModuleMode = ModuleMode()
        module_mode.capture(module)
        fit_completed: bool = False

        # An explicit per-call checkpoint path overrides the constructor-level
        # resume path; with both absent the run starts fresh.
        resume_path: Path | None = (
            Path(ckpt_path) if ckpt_path is not None else self.resume_from_checkpoint
        )

        try:
            # Initialize the strategy's process context, run single-copy data
            # preparation on global rank zero only, and hold every rank at the
            # barrier so no process enters setup against partially prepared
            # data.
            self.strategy.setup_environment()

            if is_rank_zero():
                datamodule.prepare_data()
            self.strategy.barrier()

            datamodule.setup("fit")

            # Retrieve the fit-stage dataloaders. The harness supports exactly
            # one loader per stage, so collection returns are rejected here
            # rather than failing obscurely inside the loops.
            self.train_dataloader: torch.utils.data.DataLoader | None = datamodule.train_dataloader()

            if isinstance(self.train_dataloader, (list, tuple, dict)):
                raise MisconfigurationError(
                    "Multiple train dataloaders are not supported. train_dataloader() must return a single DataLoader."
                )

            self.val_dataloader: torch.utils.data.DataLoader | None = datamodule.val_dataloader()

            if isinstance(self.val_dataloader, (list, tuple, dict)):
                raise MisconfigurationError(
                    "Multiple val dataloaders are not supported. val_dataloader() must return a single DataLoader."
                )

            if self.overfit_batches:
                self.train_dataloader = self._prepare_overfit_dataloader(
                    self.train_dataloader,
                    "train"
                )
                if self.val_dataloader is not None:
                    self.val_dataloader = self._prepare_overfit_dataloader(
                        self.val_dataloader,
                        "val"
                    )

            # Under a distributed strategy, map-style datasets are re-wrapped
            # with rank-aware samplers so every process sees a disjoint shard;
            # iterable datasets own their sharding and pass through with a
            # warning.
            if self.use_distributed_sampler and self.strategy.is_distributed:
                if not isinstance(self.train_dataloader.dataset, IterableDataset):
                    self.train_dataloader: torch.utils.data.DataLoader | None = self._wrap_with_distributed_sampler(
                        self.train_dataloader,
                        shuffle=not bool(self.overfit_batches)
                    )
                elif is_rank_zero():
                    log.warning("Skipping DistributedSampler for train dataloader: IterableDataset detected.")
                if self.val_dataloader is not None:
                    if not isinstance(self.val_dataloader.dataset, IterableDataset):
                        self.val_dataloader: torch.utils.data.DataLoader | None = self._wrap_with_distributed_sampler(self.val_dataloader, shuffle=False)
                    elif is_rank_zero():
                        log.warning("Skipping DistributedSampler for val dataloader: IterableDataset detected.")

            # Record loader lengths where they are knowable; unsized iterable
            # loaders leave the totals as None and downstream cadence math
            # degrades to its unsized behavior.
            self._total_train_batches: int | None = self._safe_len(self.train_dataloader)
            if self.val_dataloader is not None:
                self._total_val_batches: int | None = self._safe_len(self.val_dataloader)
            self._configure_validation_schedule(reset_time=True)

            # Advisory only: single-process loading feeding a non-CPU
            # accelerator is a common throughput mistake worth one warning,
            # never an error.
            if is_rank_zero():
                if hasattr(self.train_dataloader, 'num_workers') and self.train_dataloader.num_workers == 0:
                    if not isinstance(self.accelerator, CPUAccelerator):
                        log.warning("num_workers=0 with GPU accelerator. Consider increasing for better throughput.")

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
            with module._hook_context("configure_optimizers"):
                optimization_configuration: object = module.configure_optimizers()
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
                SyntheticOptimizer._to_synthetic_optimizer(optimizer, self, module)
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

            # Schedulers pair positionally with optimizers, one definition per
            # optimizer. SchedulerConfig records are compiled against the
            # resolved total-step horizon; pre-built scheduler instances pass
            # through with no managed config, leaving their stepping to the
            # module.
            # Checkpoint restoration happens after optimizers and schedulers
            # exist, because their serialized state must land on live objects,
            # and before any hook fires, so every hook observes the resumed
            # run.
            if resume_path is not None:
                self._restore_checkpoint(resume_path, module)
            _optimizers_to_device(self.optimizers, device)

            # Duplicate stateful-callback keys are rejected before callback
            # setup so two instances cannot silently overwrite each other's
            # checkpoint state later.
            self._validate_callback_state_keys()

            with module._hook_context("setup"):
                for callback in self.callbacks:
                    callback.setup(self, module, "fit")
                module.setup("fit")

            # The run configuration is announced to the logger from rank zero
            # only, keeping multi-process runs single-writer.
            if self.logger is not None and is_rank_zero():
                config_dict: HyperparameterDict = {
                    "max_epochs": self.max_epochs,
                    "max_steps": self.max_steps,
                    "precision": self.precision,
                    "accelerator": type(self.accelerator).__name__,
                    "strategy": type(self.strategy).__name__
                }
                self.logger.log_hyperparams(config_dict)

            # Start hooks fire callbacks-first, then the module. The optional
            # sanity pass runs inside the fit stage but before training
            # starts, surfacing validation-path defects before the first
            # training epoch is paid for; it discards every trace of itself.
            with module._hook_context("on_fit_start"):
                for callback in self.callbacks:
                    callback.on_fit_start(self, module)
                module.on_fit_start()

            if (
                self.enable_validation
                and self.num_sanity_val_steps > 0
                and self._validation_schedule.is_epoch_eligible(self.current_epoch)
                and resume_path is None
            ):
                self._sanity_check_validation(module)

            # Enter the training running-stage and hand control to the fit
            # loop, which owns epochs, optimization, cadence-driven
            # validation, and stop requests from here.
            self.state.set_running_stage("training")
            module._running_stage: str | None = "training"

            self.fit_loop.run()

            # End hooks mirror the start order reversed: the module first,
            # then the callbacks.
            with module._hook_context("on_fit_end"):
                for callback in self.callbacks:
                    callback.on_fit_end(self, module)
                module.on_fit_end()
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
            module._running_stage: str | None = None
            module._current_fx_name: str | None = None
            self.state.finish()
            self._active_exception = None
            self._is_mid_epoch_validation = False
            if self.fit_loop.epoch_loop is not None:
                self.fit_loop.epoch_loop.reset_on_run()
            if cleanup_exception is not None:
                raise cleanup_exception

    def validate(
        self,
        module: Module,
        datamodule: DataModule
    ) -> dict[str, float]:
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
        launcher = self.strategy.launcher
        if launcher is not None and launcher.should_launch:
            return cast(
                dict[str, float],
                launcher.launch("validate", self, module, datamodule, {})
            )
        self.state.set_stage("validate")
        self._active_exception = None
        self._callback_metrics.clear()
        self._reset_progress_bar_metrics("validating")
        self._total_val_batches: int | None = None
        self.module: Module | None = module
        self.datamodule: DataModule | None = datamodule
        module._trainer: Trainer | None = self

        module_mode: ModuleMode = ModuleMode()
        module_mode.capture(module)
        stage_completed: bool = False
        metrics: dict[str, float] = {}
        try:
            self.strategy.setup_environment()

            if is_rank_zero():
                datamodule.prepare_data()
            self.strategy.barrier()

            datamodule.setup("validate")
            self.val_dataloader: torch.utils.data.DataLoader | None = datamodule.val_dataloader()

            if isinstance(self.val_dataloader, (list, tuple, dict)):
                raise MisconfigurationError(
                    "Multiple val dataloaders are not supported. val_dataloader() must return a single DataLoader."
                )

            if self.overfit_batches:
                self.val_dataloader = self._prepare_overfit_dataloader(
                    self.val_dataloader,
                    "val"
                )

            if self.use_distributed_sampler and self.strategy.is_distributed:
                if not isinstance(self.val_dataloader.dataset, IterableDataset):
                    self.val_dataloader: torch.utils.data.DataLoader | None = self._wrap_with_distributed_sampler(self.val_dataloader, shuffle=False)
                elif is_rank_zero():
                    log.warning("Skipping DistributedSampler for val dataloader: IterableDataset detected.")

            self._total_val_batches: int | None = self._safe_len(self.val_dataloader) if self.val_dataloader is not None else None

            device: torch.device = self._resolve_device()
            module = self._setup_model(module, device)
            module, _, _ = self._connect_precision_plugin(module, [], [])

            with module._hook_context("setup"):
                for callback in self.callbacks:
                    callback.setup(self, module, "validate")
                module.setup("validate")

            metrics: dict[str, float] = self.validate_loop.run()
            stage_completed: bool = True
        except BaseException as exception:
            self._handle_exception(module, datamodule, exception)
        finally:
            if not stage_completed:
                module_mode.restore(module)
            cleanup_exception: BaseException | None = self._teardown_stage(module, datamodule, "validate")
            module._running_stage: str | None = None
            module._current_fx_name: str | None = None
            self.state.finish()
            self._active_exception = None
            if cleanup_exception is not None:
                raise cleanup_exception

        return metrics

    def test(
        self,
        module: Module,
        datamodule: DataModule
    ) -> dict[str, float]:
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
        launcher = self.strategy.launcher
        if launcher is not None and launcher.should_launch:
            return cast(
                dict[str, float],
                launcher.launch("test", self, module, datamodule, {})
            )
        self.state.set_stage("test")
        self._active_exception = None
        self._callback_metrics.clear()
        self._reset_progress_bar_metrics("testing")
        self._total_test_batches: int | None = None
        self.module: Module | None = module
        self.datamodule: DataModule | None = datamodule
        module._trainer: Trainer | None = self

        module_mode: ModuleMode = ModuleMode()
        module_mode.capture(module)
        stage_completed: bool = False
        metrics: dict[str, float] = {}
        try:
            self.strategy.setup_environment()

            if is_rank_zero():
                datamodule.prepare_data()
            self.strategy.barrier()

            datamodule.setup("test")
            self.test_dataloader: torch.utils.data.DataLoader | None = datamodule.test_dataloader()

            if isinstance(self.test_dataloader, (list, tuple, dict)):
                raise MisconfigurationError(
                    "Multiple test dataloaders are not supported. test_dataloader() must return a single DataLoader."
                )

            if self.overfit_batches:
                self.test_dataloader = self._prepare_overfit_dataloader(
                    self.test_dataloader,
                    "test"
                )

            if self.use_distributed_sampler and self.strategy.is_distributed:
                if not isinstance(self.test_dataloader.dataset, IterableDataset):
                    self.test_dataloader: torch.utils.data.DataLoader | None = self._wrap_with_distributed_sampler(self.test_dataloader, shuffle=False)
                elif is_rank_zero():
                    log.warning("Skipping DistributedSampler for test dataloader: IterableDataset detected.")

            self._total_test_batches: int | None = self._safe_len(self.test_dataloader) if self.test_dataloader is not None else None

            device: torch.device = self._resolve_device()
            module = self._setup_model(module, device)
            module, _, _ = self._connect_precision_plugin(module, [], [])

            with module._hook_context("setup"):
                for callback in self.callbacks:
                    callback.setup(self, module, "test")
                module.setup("test")

            metrics: dict[str, float] = self.test_loop.run()
            stage_completed: bool = True
        except BaseException as exception:
            self._handle_exception(module, datamodule, exception)
        finally:
            if not stage_completed:
                module_mode.restore(module)
            cleanup_exception: BaseException | None = self._teardown_stage(module, datamodule, "test")
            module._running_stage: str | None = None
            module._current_fx_name: str | None = None
            self.state.finish()
            self._active_exception = None
            if cleanup_exception is not None:
                raise cleanup_exception

        return metrics

    def predict(
        self,
        module: Module,
        datamodule: DataModule,
        return_predictions: bool = True
    ) -> list[ModelOutput]:
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
        launcher = self.strategy.launcher
        if launcher is not None and launcher.should_launch:
            return cast(
                list[ModelOutput],
                launcher.launch(
                    "predict",
                    self,
                    module,
                    datamodule,
                    {"return_predictions": return_predictions}
                )
            )
        self.state.set_stage("predict")
        self._active_exception = None
        self._callback_metrics.clear()
        self._reset_progress_bar_metrics("predicting")
        self._total_predict_batches: int | None = None
        self.module: Module | None = module
        self.datamodule: DataModule | None = datamodule
        module._trainer: Trainer | None = self

        module_mode: ModuleMode = ModuleMode()
        module_mode.capture(module)
        stage_completed: bool = False
        predictions: list[ModelOutput] = []
        try:
            self.strategy.setup_environment()

            if is_rank_zero():
                datamodule.prepare_data()
            self.strategy.barrier()

            datamodule.setup("predict")
            self.predict_dataloader: torch.utils.data.DataLoader | None = datamodule.predict_dataloader()

            if isinstance(self.predict_dataloader, (list, tuple, dict)):
                raise MisconfigurationError(
                    "Multiple predict dataloaders are not supported. predict_dataloader() must return a single DataLoader."
                )

            if self.overfit_batches:
                self.predict_dataloader = self._prepare_overfit_dataloader(
                    self.predict_dataloader,
                    "predict"
                )

            if self.use_distributed_sampler and self.strategy.is_distributed:
                if not isinstance(self.predict_dataloader.dataset, IterableDataset):
                    self.predict_dataloader: torch.utils.data.DataLoader | None = self._wrap_with_unrepeated_sampler(self.predict_dataloader)
                elif is_rank_zero():
                    log.warning("Skipping DistributedSampler for predict dataloader: IterableDataset detected.")

            self._total_predict_batches: int | None = self._safe_len(self.predict_dataloader)

            device: torch.device = self._resolve_device()
            module = self._setup_model(module, device)
            module, _, _ = self._connect_precision_plugin(module, [], [])

            with module._hook_context("setup"):
                for callback in self.callbacks:
                    callback.setup(self, module, "predict")
                module.setup("predict")

            self.predict_loop.return_predictions: bool = return_predictions
            predictions: list[ModelOutput] = self.predict_loop.run()
            stage_completed: bool = True
        except BaseException as exception:
            self._handle_exception(module, datamodule, exception)
        finally:
            if not stage_completed:
                module_mode.restore(module)
            cleanup_exception: BaseException | None = self._teardown_stage(module, datamodule, "predict")
            module._running_stage: str | None = None
            module._current_fx_name: str | None = None
            self.state.finish()
            self._active_exception = None
            if cleanup_exception is not None:
                raise cleanup_exception

        return predictions

    def _safe_len(
        self,
        dataloader: torch.utils.data.DataLoader | None
    ) -> int | None:
        # Returns the dataloader length, or None both for an absent dataloader
        # and for iterable datasets that define no length, so callers can
        # treat unknown totals uniformly.
        if dataloader is None:
            return None
        try:
            return len(dataloader)
        except TypeError:
            return None

    def _configure_validation_schedule(self, reset_time: bool) -> None:
        assert self.fit_loop.epoch_loop is not None
        max_batches: int | None = self.fit_loop.epoch_loop._resolve_limit_batches(
            self._total_train_batches
        )
        validation_enabled: bool = self.val_dataloader is not None
        if validation_enabled:
            if self.limit_val_batches == 0:
                validation_enabled = False
            else:
                validation_enabled = (
                    self.validate_loop._resolve_limit_batches(self._total_val_batches) != 0
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
        self.fit_loop.trainer: Trainer = self
        assert self.fit_loop.epoch_loop is not None
        self.fit_loop.epoch_loop.trainer: Trainer = self
        self.validate_loop.trainer: Trainer = self
        self.test_loop.trainer: Trainer = self
        self.predict_loop.trainer: Trainer = self

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
        with module._hook_context(hook_name):
            hook()

    def _setup_model(self, module: Module, device: torch.device) -> Module:
        self.accelerator.setup_device(device)
        with self.precision_plugin.module_init_context():
            self._configure_model(module)
        converted_model: torch.nn.Module = self.precision_plugin.convert_module(module)
        if not isinstance(converted_model, Module):
            raise TypeError(
                f"Precision.convert_module must return Module, got {type(converted_model).__name__}."
            )
        resolved_module: Module = converted_model
        if self.sync_batchnorm and self.strategy.is_distributed:
            synchronized_model: torch.nn.Module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                resolved_module
            )
            if not isinstance(synchronized_model, Module):
                raise TypeError(
                    f"SyncBatchNorm conversion must preserve Module, got {type(synchronized_model).__name__}."
                )
            resolved_module: Module = synchronized_model
        resolved_module._trainer = self
        self.module = resolved_module
        self.strategy.setup(resolved_module, device)
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
        connected_model: torch.nn.Module
        connected_optimizers: list[torch.optim.Optimizer]
        connected_lr_schedulers: list[SchedulerT]
        connected_model, connected_optimizers, connected_lr_schedulers = (
            self.precision_plugin.connect(
                strategy_model,
                optimizers,
                lr_schedulers
            )
        )
        if not isinstance(connected_model, torch.nn.Module):
            raise TypeError(
                f"Precision.connect must return torch.nn.Module, got {type(connected_model).__name__}."
            )
        self.strategy.model = connected_model
        return module, connected_optimizers, connected_lr_schedulers

    def _call_on_validation_model_zero_grad(self, module: Module) -> None:
        with module._hook_context("on_validation_model_zero_grad"):
            module.on_validation_model_zero_grad()

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
        sampler: DistributedSampler = DistributedSampler(
            dataloader.dataset, shuffle=shuffle
        )
        return self._replace_dataloader_sampler(dataloader, sampler)

    def _wrap_with_unrepeated_sampler(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> torch.utils.data.DataLoader:
        # Rebuilds the prediction dataloader around the unrepeated sampler,
        # which partitions without padding so no rank emits duplicate
        # predictions; all other construction parameters carry over unchanged.
        sampler: _UnrepeatedDistributedSampler = _UnrepeatedDistributedSampler(
            dataloader.dataset, shuffle=False
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
            return torch.utils.data.DataLoader(
                dataset=dataloader.dataset,
                batch_sampler=replaced_batch_sampler,
                num_workers=dataloader.num_workers,
                collate_fn=dataloader.collate_fn,
                pin_memory=dataloader.pin_memory,
                timeout=dataloader.timeout,
                worker_init_fn=dataloader.worker_init_fn,
                multiprocessing_context=getattr(dataloader, "multiprocessing_context", None),
                generator=getattr(dataloader, "generator", None),
                prefetch_factor=dataloader.prefetch_factor,
                persistent_workers=dataloader.persistent_workers,
                pin_memory_device=getattr(dataloader, "pin_memory_device", ""),
                in_order=getattr(dataloader, "in_order", True)
            )
        return torch.utils.data.DataLoader(
            dataset=dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=sampler,
            drop_last=dataloader.drop_last,
            num_workers=dataloader.num_workers,
            collate_fn=dataloader.collate_fn,
            pin_memory=dataloader.pin_memory,
            timeout=dataloader.timeout,
            worker_init_fn=dataloader.worker_init_fn,
            multiprocessing_context=getattr(dataloader, "multiprocessing_context", None),
            generator=getattr(dataloader, "generator", None),
            prefetch_factor=dataloader.prefetch_factor,
            persistent_workers=dataloader.persistent_workers,
            pin_memory_device=getattr(dataloader, "pin_memory_device", ""),
            in_order=getattr(dataloader, "in_order", True)
        )

    def _sanity_check_validation(self, module: Module) -> None:
        # Runs a bounded validation pass before training to surface evaluation
        # defects immediately instead of after the first training epoch. The
        # validate loop's batch limit is temporarily replaced and restored in
        # a finally block, and every trace of the sanity pass (epoch metrics,
        # the module's logging buffer, callback metrics) is discarded so
        # sanity results cannot influence monitoring or checkpointing.
        with module._hook_context("on_sanity_check_start"):
            for callback in self.callbacks:
                callback.on_sanity_check_start(self, module)

        self.state.set_running_stage("sanity_checking")
        module._running_stage: str | None = "sanity_checking"
        self._sanity_checking: bool = True

        try:
            self.validate_loop.run()
        finally:
            self._sanity_checking: bool = False

        self._validation_schedule.mark_validation_end()

        self.validate_loop.epoch_metrics: dict[str, float] = {}
        module._reset_logged_metrics()
        self._callback_metrics.clear()
        self._reset_progress_bar_metrics("validating")

        with module._hook_context("on_sanity_check_end"):
            for callback in self.callbacks:
                callback.on_sanity_check_end(self, module)

        module._running_stage: str | None = "training"

    def _restore_checkpoint(self, path: Path, module: Module) -> None:
        # Restores a full checkpoint onto the live objects: model, optimizer,
        # and scheduler state through CheckpointState, the progress counters
        # into the trainer state and module mirrors, and the mid-epoch resume
        # markers (completed batches and epoch-start random state) into the
        # training epoch loop. The module and callback load hooks fire last so
        # they observe the fully restored payload.
        checkpoint: CheckpointDict = load_checkpoint(path, map_location=self.strategy.root_device)
        self.precision_plugin.on_load_checkpoint(checkpoint)
        assert self.optimizer is not None
        epoch: int
        global_step: int
        completed_batches: int
        total_batch_idx: int
        epoch_rng_state: CheckpointDict | None
        epoch, global_step, completed_batches, total_batch_idx, epoch_rng_state = CheckpointState.restore(
            checkpoint=checkpoint,
            model=module,
            optimizers=self.optimizers,
            scheduler=self.scheduler,
            schedulers=self.schedulers,
            callbacks=self.callbacks,
            datamodule=self.datamodule,
            precision_plugin=self.precision_plugin
        )
        self.state.load_state_dict({"current_epoch": epoch, "global_step": global_step})
        module._current_epoch: int = epoch
        module._global_step: int = global_step
        if self.fit_loop.epoch_loop is not None:
            if completed_batches > 0:
                self.fit_loop.epoch_loop._completed_batches: int = completed_batches
            if total_batch_idx >= 0:
                self.fit_loop.epoch_loop._total_batches_processed: int = total_batch_idx + 1
            if epoch_rng_state is not None:
                self.fit_loop.epoch_loop._resume_rng_state: CheckpointDict | None = epoch_rng_state
        with module._hook_context("on_load_checkpoint"):
            if hasattr(module, "on_load_checkpoint"):
                module.on_load_checkpoint(checkpoint)
            for callback in self.callbacks:
                callback.on_load_checkpoint(self, module, checkpoint)

    def _handle_exception(
        self,
        module: Module,
        datamodule: DataModule,
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
        datamodule: DataModule,
        exception: BaseException
    ) -> None:
        self.state.interrupt()
        for index, callback in enumerate(self.callbacks):
            try:
                with module._hook_context("on_exception"):
                    callback.on_exception(self, module, exception)
            except BaseException as hook_exception:
                self._add_exception_note(
                    exception,
                    f"callback[{index}].on_exception",
                    hook_exception
                )
        try:
            datamodule.on_exception(exception)
        except BaseException as hook_exception:
            self._add_exception_note(exception, "datamodule.on_exception", hook_exception)
        try:
            self.strategy.on_exception(exception)
        except BaseException as hook_exception:
            self._add_exception_note(exception, "strategy.on_exception", hook_exception)

    def _teardown_stage(
        self,
        module: Module,
        datamodule: DataModule,
        stage: TrainerStage
    ) -> BaseException | None:
        from syntheticmind.callbacks.progress_bar import ProgressBarCallback

        failures: list[_TeardownFailure] = []
        progress_callbacks: list[tuple[int, ProgressBarCallback]] = []
        self._attempt_teardown("strategy.teardown", self.strategy.teardown, failures)
        self._attempt_teardown(
            "precision_plugin.teardown",
            self.precision_plugin.teardown,
            failures
        )
        self._attempt_teardown(
            "datamodule.teardown",
            lambda: datamodule.teardown(stage),
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
        if self.logger is not None:
            self._attempt_teardown("logger.finalize", self.logger.finalize, failures)

        exception_notified: bool = False
        if failures and self._active_exception is None:
            self._notify_exception(module, datamodule, failures[0][1])
            exception_notified = True

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
        with module._hook_context("teardown"):
            callback.teardown(self, module, stage)

    def _call_module_teardown(self, module: Module, stage: TrainerStage) -> None:
        with module._hook_context("teardown"):
            module.teardown(stage)

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

    def _validate_callback_state_keys(self) -> None:
        # Enforces state-key uniqueness across stateful callbacks before
        # fitting. Only callbacks that override the state methods participate,
        # so stateless callbacks of the same class may coexist freely, while
        # two stateful instances sharing a key would silently overwrite each
        # other inside checkpoints and are rejected instead.
        seen: dict[str, str] = {}
        for cb in self.callbacks:
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
            limited_batches: int | None = self.fit_loop.epoch_loop._resolve_limit_batches(
                self._total_train_batches
            )
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
    def callback_metrics(self) -> dict[str, float]:
        # Copy of the metric mapping monitored by callbacks; returned as a new
        # dictionary so callers cannot mutate the trainer's internal state.
        return dict(self._callback_metrics)

    @property
    def logged_metrics(self) -> dict[str, float]:
        # Reduced epoch metrics from the most recent validation pass, exposed
        # for inspection after validate() or a fit-scheduled validation.
        metrics: dict[str, float] = {}
        metrics.update(self.validate_loop.epoch_metrics)
        return metrics

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
    def should_stop(self) -> bool:
        return self.state.should_stop

    @should_stop.setter
    def should_stop(self, value: bool) -> None:
        self.state.should_stop = value

    @property
    def enable_validation(self) -> bool:
        return self._validation_schedule.enabled

    @property
    def num_training_batches(self) -> int | float:
        assert self.fit_loop.epoch_loop is not None
        resolved: int | None = self.fit_loop.epoch_loop._resolve_limit_batches(
            self._total_train_batches
        )
        return float("inf") if resolved is None else resolved

    @property
    def num_val_batches(self) -> list[int | float]:
        if self.val_dataloader is None:
            return []
        resolved: int | None = self.validate_loop._resolve_limit_batches(
            self._total_val_batches
        )
        return [float("inf") if resolved is None else resolved]

    @property
    def num_sanity_val_batches(self) -> list[int | float]:
        return [
            min(self.num_sanity_val_steps, batches)
            for batches in self.num_val_batches
        ]

    @property
    def num_test_batches(self) -> list[int | float]:
        if self.test_dataloader is None:
            return []
        resolved: int | None = self.test_loop._resolve_limit_batches(
            self._total_test_batches
        )
        return [float("inf") if resolved is None else resolved]

    @property
    def num_predict_batches(self) -> list[int | float]:
        if self.predict_dataloader is None:
            return []
        resolved: int | None = self.predict_loop._resolve_limit_batches(
            self._total_predict_batches
        )
        return [float("inf") if resolved is None else resolved]

    @property
    def estimated_stepping_batches(self) -> int | float:
        # Estimated total optimizer steps for the configured run, derived from
        # the limited batch count and the accumulation factor. Returns
        # infinity when the batch total is unknown, which callers such as
        # one-cycle schedule sizing must handle explicitly.
        if self.fit_loop.epoch_loop is None:
            return float("inf")
        limited_batches: int | None = self.fit_loop.epoch_loop._resolve_limit_batches(
            self._total_train_batches
        )
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
