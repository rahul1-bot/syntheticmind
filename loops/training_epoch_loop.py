
import warnings
from collections.abc import Callable, Iterator
from functools import partial
from typing import Literal, cast, override

import syntheticmind
import torch
from torch import nn

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import LoggedMetric, Module
from syntheticmind.core.optimizer import SyntheticOptimizer
from syntheticmind.loops.fetchers import _DataFetcher
from syntheticmind.loops.loop import Loop
from syntheticmind.loops.optimization.automatic import _AutomaticOptimization
from syntheticmind.loops.optimization.manual import _ManualOptimization
from syntheticmind.trainer.states import TrainerState
from syntheticmind.strategies.strategy import Strategy
from syntheticmind.combined_loader import CombinedLoader
from syntheticmind.exceptions import MisconfigurationError
from syntheticmind.trainer.connectors.logger_connector.metric_accumulator import (
    MetricAccumulator,
    extract_batch_size
)
from syntheticmind.types import (
    Batch,
    BackwardInputs,
    CheckpointDict,
    ClipAlgorithm,
    DataLoaderState,
    OptimizerClosure,
    OptimizerStepOptions,
    OptimizerStepOutput,
    RunningStage,
    SchedulerInterval,
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


class TrainingEpochLoop(Loop):
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
        self.is_last_batch: bool = False

    def reset_on_run(self) -> None:
        self.reset_on_teardown()
        self.is_last_batch: bool = False

    def reset_on_teardown(self) -> None:
        self._completed_batches: int = 0
        self._total_batches_processed: int = 0
        self._has_entered_batch_loop: bool = False
        self._resume_rng_state: CheckpointDict | None = None
        self._resume_checkpoint_rng_state: CheckpointDict | None = None
        self._combined_loader_state_restored: bool = False
        self._completed_loader_states: list[DataLoaderState] = []
        self._use_completed_loader_states: bool = False

    def run(self) -> None:
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

        epoch_accum: dict[str, tuple[LoggedMetric, MetricAccumulator]] = {}
        batch_result: _BatchLoopResult = self._run_batches(
            trainer,
            module,
            combined_loader,
            epoch_accum
        )
        batch_idx: int = batch_result[0]
        tail_is_accumulating: bool = batch_result[1]
        epoch_completed: bool = batch_result[2]
        self._restore_checkpoint_random_state()
        self._finish_optimization(
            trainer,
            module,
            strategy,
            state,
            optimizer,
            batch_idx,
            tail_is_accumulating
        )
        self._store_training_continuation(
            trainer,
            combined_loader,
            epoch_completed
        )
        self._has_entered_batch_loop: bool = False
        self._completed_batches: int = 0
        self._dispatch_epoch_metrics(epoch_accum, state)

    def _start_epoch(self) -> None:
        self.validation_ran_this_epoch: bool = False
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
        epoch_accum: dict[str, tuple[LoggedMetric, MetricAccumulator]]
    ) -> _BatchLoopResult:
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
        epoch_accum: dict[str, tuple[LoggedMetric, MetricAccumulator]]
    ) -> bool | None:
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
        return is_accumulating

    def _should_stop_batch_loop(
        self,
        trainer: syntheticmind.Trainer,
        state: TrainerState
    ) -> bool:
        return self._max_steps_reached(state) or (
            trainer.should_stop and trainer.fit_loop._can_stop_early(state)
        )

    def _restore_checkpoint_random_state(self) -> None:
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
        tail_is_accumulating: bool
    ) -> None:
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
        if module.automatic_optimization:
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

    def _dispatch_step_metrics(
        self,
        module: Module,
        state: TrainerState,
        epoch_accum: dict[str, tuple[LoggedMetric, MetricAccumulator]],
        batch: Batch,
        is_accumulating: bool = False
    ) -> None:
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
                accumulator: MetricAccumulator = epoch_accum[metric.storage_key][1]
                if metric.is_metric:
                    accumulator.update_metric(metric.value)
                else:
                    resolved_batch_size: int | None = metric.batch_size
                    if resolved_batch_size is None and metric.reduction == "mean":
                        resolved_batch_size: int | None = extract_batch_size(batch)
                    accumulator.update(
                        metric.local_value(),
                        batch_size=resolved_batch_size
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
        epoch_accum: dict[str, tuple[LoggedMetric, MetricAccumulator]],
        state: TrainerState
    ) -> None:
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
