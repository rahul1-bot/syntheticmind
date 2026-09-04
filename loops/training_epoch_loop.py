# This module:
# 1. Defines TrainingEpochLoop, one logical training-loader epoch composed from
#    resume coordinates, data fetch, transfer, hooks, automatic or manual
#    optimization, accumulation, validation, scheduler cadence, metrics,
#    checkpoint durability, and exact continuation state
# 2. Owns the rank-local scalar and TorchMetrics windows required to resume a
#    partial epoch without fabricating an epoch boundary
#
# Design decisions:
# - A logical epoch ends only when its CombinedLoader exhausts. A max-step,
#   callback stop, or time boundary inside the loader preserves the current epoch
#   and does not run epoch-end hooks or epoch schedulers
# - Stateful loaders restore their physical cursor and continue directly;
#   stateless loaders replay the completed prefix from the epoch-start RNG state,
#   then restore checkpoint-time RNG before the first new batch
# - Unsized lookahead captures loader state after the returned batch and before
#   the unseen prefetch, preventing checkpoints from skipping data on resume
# - Automatic optimization owns closure, zeroing, backward, accumulation,
#   clipping, managed step, progress, and scheduler cadence. Manual optimization
#   executes only the client step and lets Module methods drive mutation
# - DDP synchronization is deferred through Strategy during non-boundary
#   microbatches. If loader exhaustion leaves a partial accumulation tail,
#   gradients synchronize and step exactly once before epoch completion
# - Checkpoints are durable only outside an incomplete batch and, for automatic
#   optimization, after a complete optimizer boundary; callbacks may record save
#   intent earlier but persistence waits for this state
# - Standard schedulers and plateau schedulers are separate operations because a
#   plateau step requires a settled monitor. Epoch schedulers run only at true
#   loader completion
# - Epoch metric identity, accumulator values, TorchMetrics state, update count,
#   loader cursor, and RNG coordinates persist together and validate before any
#   resumed live state mutates
#
# Author: Rahul Sawhney

import warnings
from collections.abc import Callable, Iterator
from functools import partial
from typing import Literal, cast, override

import syntheticmind
import torch
from torch import nn
from torchmetrics import Metric

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.metric import LoggedMetric, _LoggedMetricCheckpoint
from syntheticmind.core.module import Module
from syntheticmind.core.optimizer import SyntheticOptimizer
from syntheticmind.loops.fetchers import _DataFetcher
from syntheticmind.loops.loop import Loop
from syntheticmind.loops.optimization.automatic import _AutomaticOptimization
from syntheticmind.loops.optimization.manual import _ManualOptimization
from syntheticmind.trainer.states import TrainerState
from syntheticmind.strategies.strategy import Strategy
from syntheticmind.data.combined_loader import CombinedLoader
from syntheticmind.primitives.exceptions import MisconfigurationError
from syntheticmind.trainer.connectors.logger_connector.metric_accumulator import (
    MetricAccumulator
)
from syntheticmind.primitives.types import (
    Batch,
    BackwardInputs,
    CheckpointDict,
    CheckpointValue,
    ClipAlgorithm,
    DataLoaderState,
    OptimizerClosure,
    OptimizerStepOptions,
    OptimizerStepOutput,
    RunningStage,
    SchedulerInterval,
    StateDict,
    StepOutput,
    TrainingStepOutput
)

__all__: list[str] = ["TrainingEpochLoop"]

type _BatchOptimizationResult = tuple[StepOutput, bool]
type _BatchIteration = tuple[
    _DataFetcher[tuple[Batch, int, int]],
    Iterator[tuple[Batch, int, int]],
    int,
    int
]
type _BatchLoopResult = tuple[int, bool, bool]
type _EpochMetricEntry = tuple[LoggedMetric, MetricAccumulator]


class TrainingEpochLoop(Loop):
    # Stateful training epoch reused by FitLoop. Client code interacts through
    # Module training and optimization methods; Trainer and checkpoint callbacks
    # inspect durability and continuation state through this owner.
    @override
    def __init__(
        self,
        min_steps: int | None = None,
        max_steps: int = -1,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: ClipAlgorithm | None = None,
        log_every_n_steps: int = 10
    ) -> None:
        # Stores resolved step, accumulation, clipping, and logging policy and
        # initializes optimization helpers plus empty continuation and metric state.
        super().__init__()
        self.min_steps: int | None = min_steps
        self.max_steps: int = max_steps
        self.accumulate_grad_batches: int = accumulate_grad_batches
        self.gradient_clip_val: int | float | None = gradient_clip_val
        self.gradient_clip_algorithm: ClipAlgorithm | None = gradient_clip_algorithm
        self.log_every_n_steps: int = log_every_n_steps
        self.automatic_optimization: _AutomaticOptimization = (
            _AutomaticOptimization(accumulate_grad_batches)
        )
        self.manual_optimization: _ManualOptimization = _ManualOptimization()
        self.validation_ran_this_epoch: bool = False
        self._completed_batches: int = 0
        self._total_batches_processed: int = 0
        self._has_entered_batch_loop: bool = False
        self._epoch_start_rng_state: CheckpointDict | None = None
        self._resume_rng_state: CheckpointDict | None = None
        self._resume_checkpoint_rng_state: CheckpointDict | None = None
        self._combined_loader_state_restored: bool = False
        self._completed_loader_states: list[DataLoaderState] = []
        self._use_completed_loader_states: bool = False
        self._checkpoint_is_durable: bool = True
        self._epoch_accum: dict[str, _EpochMetricEntry] = {}
        self.epoch_completed: bool = True
        self.is_last_batch: bool = False

    def reset_on_run(self) -> None:
        # Clears prior lifecycle state before a new fit while resetting the
        # externally visible last-batch marker.
        self.reset_on_teardown()
        self.is_last_batch: bool = False

    def reset_on_teardown(self) -> None:
        # Releases epoch accumulators, cursor coordinates, replay RNG, loader
        # snapshots, and durability state after fit ownership ends.
        self._reset_epoch_metrics()
        self._completed_batches: int = 0
        self._total_batches_processed: int = 0
        self._has_entered_batch_loop: bool = False
        self._resume_rng_state: CheckpointDict | None = None
        self._resume_checkpoint_rng_state: CheckpointDict | None = None
        self._combined_loader_state_restored: bool = False
        self._completed_loader_states: list[DataLoaderState] = []
        self._use_completed_loader_states: bool = False
        self._checkpoint_is_durable: bool = True
        self.epoch_completed: bool = True

    def run(self) -> None:
        # Initializes epoch RNG, executes remaining batches, closes any
        # accumulation tail, stores continuation when incomplete, and publishes
        # epoch metrics only after genuine loader exhaustion.
        self._start_epoch()

        trainer: syntheticmind.Trainer = self.trainer
        module: Module | None = trainer.module
        assert module is not None
        strategy: Strategy = trainer.strategy
        state: TrainerState = trainer.state
        if not trainer.optimizers:
            raise RuntimeError("Trainer has no optimizers configured for training.")
        optimizer: torch.optim.Optimizer | None = trainer.optimizer
        combined_loader: CombinedLoader | None = trainer._data_connector.train_combined_loader
        assert combined_loader is not None

        if self._max_steps_reached(state):
            return

        epoch_accum: dict[str, _EpochMetricEntry] = self._epoch_accum
        batch_result: _BatchLoopResult = self._run_batches(
            trainer,
            module,
            combined_loader,
            epoch_accum
        )
        batch_idx: int = batch_result[0]
        tail_is_accumulating: bool = batch_result[1]
        epoch_completed: bool = batch_result[2]
        self.epoch_completed: bool = epoch_completed
        self._restore_checkpoint_random_state()
        self._finish_optimization(
            trainer,
            module,
            strategy,
            state,
            optimizer,
            batch_idx,
            tail_is_accumulating,
            epoch_completed
        )
        self._store_training_continuation(
            trainer,
            combined_loader,
            epoch_completed
        )
        self._has_entered_batch_loop: bool = False
        self._completed_batches: int = 0
        if epoch_completed:
            self._dispatch_epoch_metrics(epoch_accum, state)
            self._epoch_accum: dict[str, _EpochMetricEntry] = {}

    def _start_epoch(self) -> None:
        # Restores the epoch-start RNG needed for stateless prefix replay, then
        # captures the start state of the active logical epoch for future pause.
        self.validation_ran_this_epoch: bool = False
        self._checkpoint_is_durable: bool = True
        if self._resume_rng_state is not None:
            self.trainer._checkpoint_connector.restore_random_state(
                self._resume_rng_state
            )
            self._resume_rng_state: CheckpointDict | None = None
        self._epoch_start_rng_state: CheckpointDict | None = (
            self.trainer._checkpoint_connector.capture_random_state()
        )

    def _prepare_batch_iteration(
        self,
        trainer: syntheticmind.Trainer,
        combined_loader: CombinedLoader
    ) -> _BatchIteration:
        # Distinguishes physical stateful cursor restoration from stateless prefix
        # replay, validates offsets against sized horizons, configures unsized
        # snapshots, and returns the live fetcher plus resume coordinates.
        total_batches: int | None = combined_loader._length()
        loader_state_restored: bool = self._combined_loader_state_restored
        self._combined_loader_state_restored: bool = False
        resume_batch_offset: int = (
            self._completed_batches if loader_state_restored else 0
        )
        if total_batches is not None and resume_batch_offset > total_batches:
            raise MisconfigurationError(
                f"Checkpoint progress contains {resume_batch_offset} completed batches, "
                f"but the resumed training loader has only {total_batches} batches."
            )
        max_batches: int | None = (
            None
            if total_batches is None
            else total_batches - resume_batch_offset
            if loader_state_restored
            else total_batches
        )
        skip_batches: int = 0 if loader_state_restored else self._completed_batches
        self._has_entered_batch_loop: bool = True
        loader_is_sized: bool = combined_loader._has_sized_iterables()
        self._use_completed_loader_states: bool = (
            combined_loader._has_stateful_iterables()
            and not loader_is_sized
        )
        if self._use_completed_loader_states:
            self._completed_loader_states: list[DataLoaderState] = (
                combined_loader._state_dicts()
            )
        with trainer.profiler.profile("setup_train_dataloader"):
            data_fetcher: _DataFetcher[tuple[Batch, int, int]] = _DataFetcher(
                combined_loader,
                max_batches,
                sized=loader_is_sized,
                state_snapshot=(
                    combined_loader._state_dicts
                    if self._use_completed_loader_states
                    else None
                )
            )
            train_iterator: Iterator[tuple[Batch, int, int]] = iter(data_fetcher)
        return data_fetcher, train_iterator, resume_batch_offset, skip_batches

    def _run_batches(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        combined_loader: CombinedLoader,
        epoch_accum: dict[str, _EpochMetricEntry]
    ) -> _BatchLoopResult:
        # Fetches, optionally replays, converts, and advances batches until
        # natural exhaustion or a terminal control boundary, returning final
        # index, accumulation state, and true epoch completion.
        state: TrainerState = trainer.state
        iteration: _BatchIteration = self._prepare_batch_iteration(
            trainer,
            combined_loader
        )
        data_fetcher: _DataFetcher[tuple[Batch, int, int]] = iteration[0]
        train_iterator: Iterator[tuple[Batch, int, int]] = iteration[1]
        resume_batch_offset: int = iteration[2]
        skip_batches: int = iteration[3]
        batch_idx: int = resume_batch_offset - 1
        tail_is_accumulating: bool = False
        epoch_completed: bool = data_fetcher.done
        while not data_fetcher.done:
            trainer._raise_if_sigterm()
            if self._should_stop_batch_loop(trainer, state):
                break
            try:
                with trainer.profiler.profile(
                    f"[{type(self).__name__}].train_dataloader_next"
                ):
                    fetched: tuple[Batch, int, int] = next(train_iterator)
            except StopIteration:
                epoch_completed: bool = True
                break
            batch_idx: int = batch_idx + 1
            self.is_last_batch: bool = data_fetcher.done
            batch: Batch = trainer.precision_plugin.convert_input(fetched[0])
            if batch_idx < skip_batches:
                continue
            accumulation_state: bool | None = self._advance_batch(
                trainer,
                module,
                batch,
                batch_idx,
                data_fetcher,
                epoch_accum
            )
            if accumulation_state is None:
                self.is_last_batch: bool = True
                epoch_completed: bool = True
                break
            if self.is_last_batch:
                epoch_completed: bool = True
            tail_is_accumulating: bool = accumulation_state
            if self._should_stop_batch_loop(trainer, state):
                break
        return batch_idx, tail_is_accumulating, epoch_completed

    def _advance_batch(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        batch: Batch,
        batch_idx: int,
        data_fetcher: _DataFetcher[tuple[Batch, int, int]],
        epoch_accum: dict[str, _EpochMetricEntry]
    ) -> bool | None:
        # Marks the checkpoint unsafe, restores checkpoint-time RNG after replay,
        # transfers data, dispatches start control, executes optimization,
        # records progress, dispatches end and metrics, runs cadence, and flushes
        # deferred checkpoint intent at the next durable boundary.
        self._checkpoint_is_durable: bool = False
        self._restore_checkpoint_random_state()
        batch: Batch = trainer._transfer_batch(
            module,
            batch,
            trainer.strategy.root_device,
            0
        )
        if self._batch_start_requests_stop(
            trainer,
            module,
            trainer.callbacks,
            batch,
            batch_idx
        ):
            self._checkpoint_is_durable: bool = True
            return None
        model: nn.Module | None = trainer.strategy.model
        assert model is not None
        optimization_result: _BatchOptimizationResult = (
            self._run_batch_optimization(
                trainer,
                module,
                trainer.strategy,
                model,
                trainer.optimizer,
                trainer.state,
                batch,
                batch_idx
            )
        )
        outputs: StepOutput = optimization_result[0]
        is_accumulating: bool = optimization_result[1]
        self._record_batch_progress(data_fetcher, batch_idx)
        self._run_batch_end(
            trainer,
            module,
            trainer.callbacks,
            outputs,
            batch,
            batch_idx
        )
        self._dispatch_step_metrics(
            module,
            trainer.state,
            epoch_accum,
            batch,
            is_accumulating
        )
        self._run_batch_cadence(
            trainer,
            module,
            trainer.state,
            batch_idx,
            is_accumulating
        )
        self._checkpoint_is_durable: bool = (
            not module.automatic_optimization or not is_accumulating
        )
        epoch_will_end: bool = self.is_last_batch or self._should_stop_batch_loop(
            trainer,
            trainer.state
        )
        trainer._save_batch_checkpoints(module, epoch_will_end)
        return is_accumulating

    def _should_stop_batch_loop(
        self,
        trainer: syntheticmind.Trainer,
        state: TrainerState
    ) -> bool:
        # Combines terminal max steps with cooperative stop requests that satisfy
        # FitLoop minimum epoch and step constraints.
        return self._max_steps_reached(state) or (
            trainer.should_stop and trainer.fit_loop._can_stop_early(state)
        )

    def _restore_checkpoint_random_state(self) -> None:
        # After stateless prefix replay, restores the exact RNG captured at the
        # interruption so the first new batch follows the uninterrupted trajectory.
        if self._resume_checkpoint_rng_state is None:
            return
        self.trainer._checkpoint_connector.restore_random_state(
            self._resume_checkpoint_rng_state
        )
        self._resume_checkpoint_rng_state: CheckpointDict | None = None

    def _finish_optimization(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        strategy: Strategy,
        state: TrainerState,
        optimizer: torch.optim.Optimizer | None,
        batch_idx: int,
        tail_is_accumulating: bool,
        epoch_completed: bool
    ) -> None:
        # Synchronizes and steps an incomplete accumulation tail at loader
        # exhaustion, flushes now-durable batch checkpoints, and advances standard
        # epoch schedulers only when the epoch completed.
        if module.automatic_optimization and tail_is_accumulating:
            assert optimizer is not None
            strategy.sync_gradients(optimizer)
            self._do_optimizer_step(
                trainer,
                module,
                optimizer,
                state,
                batch_idx
            )
            self.update_plateau_schedulers(
                trainer,
                module,
                "step",
                batch_idx + 1
            )
            self._checkpoint_is_durable: bool = True
            trainer._save_batch_checkpoints(module, True)
        if module.automatic_optimization and epoch_completed:
            self.update_lr_schedulers(
                trainer,
                module,
                "epoch",
                state.current_epoch + 1
            )

    def _store_training_continuation(
        self,
        trainer: syntheticmind.Trainer,
        combined_loader: CombinedLoader,
        epoch_completed: bool
    ) -> None:
        # Clears continuation after completion or stores loader state, completed
        # logical batches, global batch coordinate, and epoch-start RNG for pause.
        combined_loader_states: list[DataLoaderState] = list(
            self._completed_loader_states
            if self._use_completed_loader_states
            else combined_loader._state_dicts()
        )
        if epoch_completed:
            trainer._checkpoint_connector.reset_training_continuation()
            return
        trainer._checkpoint_connector.train_dataloader_states: list[
            DataLoaderState
        ] = combined_loader_states
        trainer._checkpoint_connector.train_dataloader_state_active: bool = True
        trainer._checkpoint_connector.train_dataloader_batches_completed: int = (
            self._completed_batches
        )
        trainer._checkpoint_connector.train_total_batch_idx: int = (
            self._total_batches_processed - 1
        )
        trainer._checkpoint_connector.train_epoch_rng_state: CheckpointDict | None = (
            self._epoch_start_rng_state
        )

    def _batch_start_requests_stop(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        callbacks: list[Callback],
        batch: Batch,
        batch_idx: int
    ) -> bool:
        # Dispatches callback and Module batch-start hooks and interprets only the
        # established Module return value ``-1`` as stop-before-computation.
        trainer._hook_dispatcher.call_callbacks(
            module,
            "on_train_batch_start",
            batch,
            batch_idx,
            callback_group=callbacks
        )
        hook_response: int | None = trainer._hook_dispatcher.call_module(
            module,
            "on_train_batch_start",
            module.on_train_batch_start,
            batch,
            batch_idx
        )
        return hook_response == -1

    def _run_batch_optimization(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        strategy: Strategy,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        state: TrainerState,
        batch: Batch,
        batch_idx: int
    ) -> _BatchOptimizationResult:
        # Selects manual step normalization or automatic closure execution,
        # suppressing distributed backward synchronization on non-boundary
        # accumulation microbatches.
        with trainer.profiler.profile("run_training_batch"):
            if not module.automatic_optimization:
                manual_step: Callable[[], TrainingStepOutput] = partial(
                    self._run_training_step,
                    strategy,
                    module,
                    model,
                    batch,
                    batch_idx
                )
                outputs: StepOutput = self.manual_optimization.run(manual_step)
                return outputs, False
            assert optimizer is not None
            is_accumulating: bool = (
                (batch_idx + 1) % self.accumulate_grad_batches != 0
                and not self.is_last_batch
            )
            automatic_closure: OptimizerClosure = self._make_automatic_closure(
                trainer,
                module,
                strategy,
                model,
                batch,
                batch_idx
            )
            if is_accumulating:
                with strategy.defer_backward_sync():
                    automatic_closure()
            else:
                self._do_optimizer_step(
                    trainer,
                    module,
                    optimizer,
                    state,
                    batch_idx,
                    automatic_closure
                )
            outputs: StepOutput = self.automatic_optimization.outputs
            return outputs, is_accumulating

    def _record_batch_progress(
        self,
        data_fetcher: _DataFetcher[tuple[Batch, int, int]],
        batch_idx: int
    ) -> bool:
        # Advances logical and cross-epoch batch coordinates and adopts the
        # unsized fetcher's post-return loader snapshot as the durable cursor.
        self._completed_batches: int = batch_idx + 1
        self._total_batches_processed: int = self._total_batches_processed + 1
        if self._use_completed_loader_states:
            if not isinstance(data_fetcher.state, list):
                raise RuntimeError(
                    "Stateful training data did not produce a loader-state snapshot."
                )
            fetched_loader_states: list[DataLoaderState] = cast(
                list[DataLoaderState],
                data_fetcher.state
            )
            self._completed_loader_states: list[DataLoaderState] = list(
                fetched_loader_states
            )
        return self.is_last_batch

    def _run_batch_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        callbacks: list[Callback],
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int
    ) -> None:
        # Previews pending metrics for progress, then dispatches callback and
        # Module batch-end hooks before final metric accumulation.
        trainer._preview_logged_metrics(module, "training")
        trainer._hook_dispatcher.call_callbacks(
            module,
            "on_train_batch_end",
            outputs,
            batch,
            batch_idx,
            callback_group=callbacks
        )
        trainer._hook_dispatcher.call_module(
            module,
            "on_train_batch_end",
            module.on_train_batch_end,
            outputs,
            batch,
            batch_idx
        )

    def _run_batch_cadence(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        state: TrainerState,
        batch_idx: int,
        is_accumulating: bool
    ) -> None:
        # Evaluates validation schedule against batch and optimizer coordinates,
        # restores training stage afterward, and advances step-interval plateau
        # schedulers only at optimizer boundaries.
        is_last_batch: bool = self.is_last_batch
        if trainer._validation_schedule.should_validate(
            state.current_epoch,
            batch_idx,
            self._total_batches_processed - 1,
            is_last_batch,
            trainer.should_stop,
            trainer.fit_loop._can_stop_early(state)
        ):
            self._run_scheduled_validation(
                trainer,
                module,
                state,
                is_accumulating,
                is_last_batch
            )
        if module.automatic_optimization and not is_accumulating:
            self.update_plateau_schedulers(
                trainer,
                module,
                "step",
                batch_idx + 1
            )

    def _run_training_step(
        self,
        strategy: Strategy,
        module: Module,
        model: nn.Module,
        batch: Batch,
        batch_idx: int
    ) -> TrainingStepOutput:
        # Enters precision and hook context, then executes the client training
        # step through the Strategy-managed model identity.
        with self.trainer.precision_plugin.train_step_context():
            with module._hook_context("training_step"):
                outputs: TrainingStepOutput = self.trainer._hook_dispatcher.call_strategy(
                    "training_step",
                    strategy.training_step,
                    model,
                    batch,
                    batch_idx
                )
        return outputs

    def _make_automatic_closure(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        strategy: Strategy,
        model: nn.Module,
        batch: Batch,
        batch_idx: int
    ) -> OptimizerClosure:
        # Composes client step, precision-aware backward, and accumulation-window
        # zeroing into the closure consumed by managed optimizer stepping.
        step_operation: Callable[[], TrainingStepOutput] = partial(
            self._run_training_step,
            strategy,
            module,
            model,
            batch,
            batch_idx
        )
        backward_operation: Callable[[torch.Tensor], None] = partial(
            self.manual_backward,
            trainer,
            module
        )
        zero_grad_operation: Callable[[], None] | None = None
        if batch_idx % self.accumulate_grad_batches == 0:
            zero_grad_operation: Callable[[], None] | None = partial(
                self._optimizer_zero_grad,
                trainer,
                module,
                trainer.optimizers[0],
                batch_idx
            )
        automatic_closure: OptimizerClosure = partial(
            self.automatic_optimization.run,
            strategy,
            step_operation,
            backward_operation,
            zero_grad_operation
        )
        return automatic_closure

    def clip_gradients(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None
    ) -> None:
        # Resolves loop or explicit clipping policy, validates the algorithm, and
        # delegates numerical handling to the active precision plugin.
        resolved_gradient_clip_val: int | float | None = (
            self.gradient_clip_val if gradient_clip_val is None else gradient_clip_val
        )
        if resolved_gradient_clip_val is None or resolved_gradient_clip_val <= 0:
            return
        resolved_gradient_clip_algorithm: str = (
            self.gradient_clip_algorithm or "norm"
            if gradient_clip_algorithm is None
            else gradient_clip_algorithm.lower()
        )
        if resolved_gradient_clip_algorithm not in {"norm", "value"}:
            raise MisconfigurationError(
                "gradient_clip_algorithm must be 'norm' or 'value'."
            )
        validated_gradient_clip_algorithm: ClipAlgorithm = (
            "norm" if resolved_gradient_clip_algorithm == "norm" else "value"
        )
        self.trainer.precision_plugin.clip_gradients(
            optimizer,
            clip_val=resolved_gradient_clip_val,
            gradient_clip_algorithm=validated_gradient_clip_algorithm
        )

    def manual_backward(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        loss: torch.Tensor,
        gradient: torch.Tensor | None = None,
        retain_graph: bool | None = None,
        create_graph: bool = False,
        inputs: BackwardInputs = None
    ) -> None:
        # Routes client or automatic backward through Strategy and precision;
        # Module ownership is already carried by the connected Strategy.
        del module
        trainer._hook_dispatcher.call_strategy(
            "backward",
            trainer.strategy.backward,
            loss,
            None,
            gradient,
            retain_graph,
            create_graph,
            inputs
        )

    def optimizer_step(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        optimizer: torch.optim.Optimizer,
        closure: OptimizerClosure | None = None,
        kwargs: OptimizerStepOptions | None = None
    ) -> OptimizerStepOutput:
        # Executes one client-requested managed optimizer step and advances global
        # progress exactly once for manual optimization.
        output: OptimizerStepOutput = self._optimizer_step_only(
            trainer,
            module,
            optimizer,
            closure=closure,
            kwargs=kwargs
        )
        self._increment_global_step(trainer.state, module)
        return output

    def _automatic_optimizer_step(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        optimizer: torch.optim.Optimizer,
        closure: OptimizerClosure | None = None,
        kwargs: OptimizerStepOptions | None = None
    ) -> OptimizerStepOutput:
        # Executes an automatic managed step without incrementing progress here;
        # the enclosing epoch operation owns that transition.
        return self._optimizer_step_only(
            trainer,
            module,
            optimizer,
            closure,
            kwargs
        )

    def optimizer_zero_grad(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        optimizer: torch.optim.Optimizer,
        batch_idx: int
    ) -> None:
        # Public loop capability used by Module manual optimization to preserve
        # zero-grad hooks and model customization.
        self._optimizer_zero_grad(trainer, module, optimizer, batch_idx)

    def _do_optimizer_step(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        optimizer: torch.optim.Optimizer,
        state: TrainerState,
        batch_idx: int,
        closure: OptimizerClosure | None = None
    ) -> None:
        # Resolves the matching SyntheticOptimizer, selects automatic context,
        # dispatches Module.optimizer_step, advances global progress, and steps
        # standard step schedulers.
        synthetic_optimizer: SyntheticOptimizer = next(
            wrapped_optimizer
            for wrapped_optimizer in trainer._synthetic_optimizers
            if wrapped_optimizer.optimizer is optimizer
        )
        with synthetic_optimizer.automatic_step_context():
            trainer._hook_dispatcher.call_module(
                module,
                "optimizer_step",
                module.optimizer_step,
                module.current_epoch,
                batch_idx,
                synthetic_optimizer,
                closure
            )
        self._increment_global_step(state, module)
        self.update_lr_schedulers(
            trainer,
            module,
            "step",
            batch_idx + 1
        )

    def update_lr_schedulers(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        interval: SchedulerInterval,
        current_index: int
    ) -> None:
        # Advances non-plateau schedulers matching interval and frequency.
        self._update_lr_schedulers(
            trainer,
            module,
            interval,
            current_index,
            "standard"
        )

    def update_plateau_schedulers(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        interval: SchedulerInterval,
        current_index: int
    ) -> None:
        # Advances plateau schedulers only after their monitored metric may exist.
        self._update_lr_schedulers(
            trainer,
            module,
            interval,
            current_index,
            "plateau"
        )

    def _update_lr_schedulers(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        interval: SchedulerInterval,
        current_index: int,
        scheduler_kind: Literal["standard", "plateau"]
    ) -> None:
        # Filters by scheduler kind, interval, and frequency; resolves plateau
        # monitor under strict or warning policy; then dispatches the Module hook.
        for configuration in trainer.lr_scheduler_configs:
            is_plateau: bool = scheduler_kind == "plateau"
            if configuration.reduce_on_plateau != is_plateau:
                continue
            if configuration.interval != interval:
                continue
            if current_index % configuration.frequency != 0:
                continue
            monitor_value: torch.Tensor | None = None
            if configuration.reduce_on_plateau:
                monitor: str | None = configuration.monitor
                assert monitor is not None
                monitor_value: torch.Tensor | None = trainer.callback_metrics.get(
                    monitor
                )
                if monitor_value is None:
                    message: str = (
                        f"ReduceLROnPlateau monitor {monitor!r} is unavailable. "
                        f"Available metrics: {list(trainer.callback_metrics)}"
                    )
                    if configuration.strict:
                        raise MisconfigurationError(message)
                    warnings.warn(message, RuntimeWarning, stacklevel=3)
                    continue
            trainer._hook_dispatcher.call_module(
                module,
                "lr_scheduler_step",
                module.lr_scheduler_step,
                configuration.scheduler,
                monitor_value
            )

    def _optimizer_step_only(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        optimizer: torch.optim.Optimizer,
        closure: OptimizerClosure | None = None,
        kwargs: OptimizerStepOptions | None = None
    ) -> OptimizerStepOutput:
        # Normalizes optional closure and keyword state and invokes Strategy
        # optimizer stepping without changing Trainer progress.
        step_kwargs: OptimizerStepOptions = {} if kwargs is None else kwargs
        resolved_closure: OptimizerClosure = closure or self._empty_closure
        return trainer._hook_dispatcher.call_strategy(
            "optimizer_step",
            trainer.strategy.optimizer_step,
            optimizer,
            resolved_closure,
            model=module,
            **step_kwargs
        )

    def _empty_closure(self) -> None:
        return None

    def _optimizer_zero_grad(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        optimizer: torch.optim.Optimizer,
        batch_idx: int
    ) -> None:
        # Dispatches callback and Module pre-zero hooks before the overridable
        # Module.optimizer_zero_grad operation.
        callbacks: list[Callback] = trainer.callbacks
        trainer._hook_dispatcher.call_callbacks(
            module,
            "on_before_zero_grad",
            optimizer,
            callback_group=callbacks
        )
        trainer._hook_dispatcher.call_module(
            module,
            "on_before_zero_grad",
            module.on_before_zero_grad,
            optimizer
        )
        trainer._hook_dispatcher.call_module(
            module,
            "optimizer_zero_grad",
            module.optimizer_zero_grad,
            module.current_epoch,
            batch_idx,
            optimizer
        )

    def _increment_global_step(
        self,
        state: TrainerState,
        module: Module
    ) -> None:
        # Advances TrainerState and mirrors the completed optimizer-step count onto
        # the client Module.
        state.increment_global_step()
        module._global_step: int = state.global_step

    def _run_scheduled_validation(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        state: TrainerState,
        is_accumulating: bool,
        is_last_batch: bool
    ) -> None:
        # Optionally clears gradients at a completed automatic boundary, marks
        # mid-epoch ownership, executes validation, advances time cadence, and
        # restores training stage identity.
        if module.automatic_optimization and not is_accumulating:
            trainer._call_on_validation_model_zero_grad(module)
        trainer._is_mid_epoch_validation: bool = not is_last_batch
        trainer.validate_loop.run()
        trainer._is_mid_epoch_validation: bool = False
        self.validation_ran_this_epoch: bool = True
        trainer._validation_schedule.mark_validation_end()
        state.set_running_stage("training")
        module._running_stage: RunningStage | None = "training"

    def _max_steps_reached(self, state: TrainerState) -> bool:
        return self.max_steps != -1 and state.global_step >= self.max_steps

    @property
    def checkpoint_is_durable(self) -> bool:
        # Whether public checkpoint persistence can represent the complete current
        # optimization and batch transition without replay ambiguity.
        return self._checkpoint_is_durable

    def epoch_metric_state(self) -> CheckpointDict:
        # Serializes every active epoch observation identity, scalar accumulator,
        # TorchMetrics tensor state, and update count for rank-local gathering.
        state: CheckpointDict = {}
        storage_key: str
        metric_entry: _EpochMetricEntry
        for storage_key, metric_entry in self._epoch_accum.items():
            metric: LoggedMetric = metric_entry[0]
            accumulator: MetricAccumulator = metric_entry[1]
            metric_state: tuple[StateDict, int] = metric._metric_checkpoint_state()
            state[storage_key]: CheckpointValue = {
                "identity": metric._checkpoint_state(),
                "accumulator": accumulator._checkpoint_state(),
                "metric_state": metric_state[0],
                "metric_update_count": metric_state[1]
            }
        return state

    def load_epoch_metric_state(
        self,
        state: CheckpointDict,
        module: Module
    ) -> None:
        # Rebuilds metric observations around registered live Metric attributes or
        # scalar placeholders, validates identity and state kind, restores
        # accumulator windows, and marks the epoch incomplete.
        self._reset_epoch_metrics()
        storage_key: str
        entry_value: CheckpointValue
        for storage_key, entry_value in state.items():
            if not isinstance(entry_value, dict):
                raise TypeError("Checkpoint epoch metric entry must be a mapping.")
            identity_value: CheckpointValue | None = entry_value.get("identity")
            accumulator_value: CheckpointValue | None = entry_value.get("accumulator")
            metric_state_value: CheckpointValue | None = entry_value.get("metric_state")
            update_count_value: CheckpointValue | None = entry_value.get(
                "metric_update_count"
            )
            if not isinstance(identity_value, dict):
                raise TypeError("Checkpoint epoch metric identity must be a mapping.")
            if not isinstance(accumulator_value, dict):
                raise TypeError("Checkpoint epoch metric accumulator must be a mapping.")
            if not isinstance(metric_state_value, dict):
                raise TypeError("Checkpoint epoch Metric state must be a mapping.")
            identity: _LoggedMetricCheckpoint = LoggedMetric._parse_checkpoint_state(
                cast(StateDict, identity_value)
            )
            metric_value: torch.Tensor | Metric
            restored_metric: Metric | None = None
            if identity.is_metric:
                metric_attribute: str | None = identity.metric_attribute
                if not metric_attribute:
                    raise ValueError("Checkpoint TorchMetric identity requires an attribute.")
                candidate: nn.Module = module.get_submodule(metric_attribute)
                if not isinstance(candidate, Metric):
                    raise TypeError(
                        f"Checkpoint metric attribute {metric_attribute!r} is not a Metric."
                    )
                restored_metric: Metric | None = candidate
                metric_value: torch.Tensor | Metric = candidate
            else:
                if metric_state_value:
                    raise ValueError("A scalar epoch metric cannot contain Metric state.")
                metric_value: torch.Tensor | Metric = torch.tensor(
                    0.0,
                    device=module.device
                )
            metric: LoggedMetric = LoggedMetric._from_checkpoint_state(
                identity,
                metric_value
            )
            if isinstance(update_count_value, bool) or not isinstance(
                update_count_value,
                int
            ):
                raise TypeError("Checkpoint Metric update count must be an integer.")
            metric._load_metric_checkpoint_state(
                cast(StateDict, metric_state_value),
                update_count_value
            )
            if metric.storage_key != storage_key:
                raise ValueError("Checkpoint epoch metric storage key is inconsistent.")
            accumulator: MetricAccumulator = MetricAccumulator(
                reduction=metric.reduction,
                sync_dist=metric.sync_dist,
                rank_zero_only=metric.rank_zero_only,
                enable_graph=False
            )
            accumulator._load_checkpoint_state(
                cast(StateDict, accumulator_value),
                module.device,
                restored_metric
            )
            self._epoch_accum[storage_key]: _EpochMetricEntry = (
                metric,
                accumulator
            )
        self.epoch_completed: bool = not bool(state)

    def _reset_epoch_metrics(self) -> None:
        # Resets every live accumulator before dropping the epoch window mapping.
        metric_entry: _EpochMetricEntry
        for metric_entry in self._epoch_accum.values():
            metric_entry[1].reset()
        self._epoch_accum: dict[str, _EpochMetricEntry] = {}

    def _dispatch_step_metrics(
        self,
        module: Module,
        state: TrainerState,
        epoch_accum: dict[str, _EpochMetricEntry],
        batch: Batch,
        is_accumulating: bool = False
    ) -> None:
        # Computes step observations, validates resumed epoch metadata, updates
        # scalar or Metric accumulators, routes callback/logger/progress views,
        # emits on real optimizer cadence, and drains Module state.
        logged: list[LoggedMetric] = module.logged_metrics
        if not logged:
            module._reset_logged_metrics()
            return

        step_metrics: dict[str, torch.Tensor] = {}
        callback_step_metrics: dict[str, torch.Tensor] = {}
        progress_bar_step_metrics: dict[str, float] = {}
        for metric in logged:
            if metric.on_step:
                value: torch.Tensor | None = metric.compute_step(
                    self.trainer.strategy.root_device,
                    self.trainer.strategy.reduce
                )
                if value is not None:
                    step_name: str = metric.output_name(on_step=True)
                    callback_step_metrics[metric.base_name()] = value
                    callback_step_metrics[step_name] = value
                    if metric.logger:
                        step_metrics[step_name] = value
                    if metric.progress_bar:
                        progress_bar_step_metrics[step_name] = (
                            self.trainer._metric_to_float(value)
                        )
            if metric.on_epoch:
                if metric.storage_key not in epoch_accum:
                    epoch_accum[metric.storage_key] = (
                        metric,
                        MetricAccumulator(
                            reduction=metric.reduction,
                            sync_dist=metric.sync_dist,
                            sync_dist_group=metric.sync_dist_group,
                            rank_zero_only=metric.rank_zero_only,
                            enable_graph=metric.enable_graph
                        )
                    )
                metric_entry: _EpochMetricEntry = epoch_accum[metric.storage_key]
                if metric_entry[0].metadata != metric.metadata:
                    raise MisconfigurationError(
                        "A resumed epoch metric changed its logging configuration."
                    )
                accumulator: MetricAccumulator = metric_entry[1]
                if metric.is_metric:
                    accumulator.update_metric(metric.value)
                else:
                    accumulator.update(
                        metric.local_value(),
                        batch_size=metric.batch_size,
                        batch=batch
                    )

        self.trainer._logger_connector.callback_metrics.update(callback_step_metrics)
        self.trainer._update_logged_metrics(step_metrics)
        self.trainer._update_progress_bar_metrics("training", progress_bar_step_metrics)
        if (
            step_metrics
            and self.trainer.loggers
            and self.log_every_n_steps > 0
            and not is_accumulating
            and state.global_step % self.log_every_n_steps == 0
        ):
            if self.trainer.is_global_zero:
                self.trainer._log_metrics(step_metrics, step=state.global_step)

        module._reset_logged_metrics()

    def _dispatch_epoch_metrics(
        self,
        epoch_accum: dict[str, _EpochMetricEntry],
        state: TrainerState
    ) -> None:
        # Computes complete epoch windows after true loader exhaustion, routes all
        # consumer views, resets accumulators, and emits one rank-zero logger record.
        if not epoch_accum:
            return
        callback_reduced: dict[str, torch.Tensor] = {}
        logger_reduced: dict[str, torch.Tensor] = {}
        progress_bar_reduced: dict[str, float] = {}
        for metric, accumulator in epoch_accum.values():
            value: torch.Tensor = accumulator.compute(
                self.trainer.strategy.root_device,
                self.trainer.strategy.reduce
            )
            if not metric.enable_graph:
                value: torch.Tensor = value.detach()
            epoch_name: str = metric.output_name(on_step=False)
            callback_reduced[metric.base_name()] = value
            callback_reduced[epoch_name] = value
            if metric.logger:
                logger_reduced[epoch_name] = value
            if metric.progress_bar:
                progress_bar_reduced[epoch_name] = self.trainer._metric_to_float(value)
        for _, accumulator in epoch_accum.values():
            accumulator.reset()
        self.trainer._logger_connector.callback_metrics.update(callback_reduced)
        self.trainer._update_logged_metrics(logger_reduced)
        self.trainer._update_progress_bar_metrics("training", progress_bar_reduced)
        if not self.trainer.loggers:
            return
        if not self.trainer.is_global_zero:
            return
        if logger_reduced:
            self.trainer._log_metrics(logger_reduced, step=state.global_step)
