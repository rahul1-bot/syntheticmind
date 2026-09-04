import inspect
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager
from typing import Literal, cast

import syntheticmind
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from syntheticmind.callbacks.callback import Callback
from syntheticmind.callbacks.checkpointing import Checkpoint
from syntheticmind.callbacks.early_stopping import EarlyStopping
from syntheticmind.core.metric import LoggedMetric
from syntheticmind.core.module import Module
from syntheticmind.loops.fetchers import _DataFetcher
from syntheticmind.loops.loop import Loop
from syntheticmind.trainer.states import TrainerState
from syntheticmind.combined_loader import CombinedLoader
from syntheticmind.trainer.connectors.logger_connector.metric_accumulator import (
    MetricAccumulator
)
from syntheticmind.types import Batch, RunningStage, StepArgument, StepOutput

__all__: list[str] = ["EvaluationLoop"]

type _EvaluationStage = Literal["validate", "test"]
type _EvaluationSetup = tuple[
    CombinedLoader,
    RunningStage,
    int | None,
    bool,
    int,
    bool
]
type _EpochAccumulator = dict[str, tuple[LoggedMetric, MetricAccumulator]]
type _BatchHookArguments = tuple[StepArgument, ...]
type _EvaluationStepHook = Callable[..., StepOutput]
type _EvaluationHook = Callable[..., StepOutput | None]


class EvaluationLoop(Loop):
    def __init__(
        self,
        stage: _EvaluationStage = "validate",
        inference_mode: bool = True
    ) -> None:
        super().__init__()
        self.stage: _EvaluationStage = stage
        self.inference_mode: bool = inference_mode
        self.verbose: bool = True
        self.epoch_metrics: dict[str, float] = {}
        self._callback_epoch_metrics: dict[str, torch.Tensor] = {}
        self._logger_epoch_metrics: dict[str, torch.Tensor] = {}
        self._progress_bar_epoch_metrics: dict[str, float] = {}
        self.results: list[dict[str, float]] = []

    def run(self) -> list[dict[str, float]]:
        use_inference_mode: bool = (
            self.inference_mode and self.trainer.state.stage != "fit"
        )
        with self._evaluation_context(use_inference_mode):
            return self._run()

    def _run(self) -> list[dict[str, float]]:
        self._reset_results()

        trainer: syntheticmind.Trainer = self.trainer
        module: Module | None = trainer.module
        assert module is not None
        state: TrainerState = trainer.state
        callbacks: list[Callback] = trainer.callbacks
        monitoring_callbacks: list[Callback] = [
            callback
            for callback in callbacks
            if isinstance(callback, (EarlyStopping, Checkpoint))
        ]
        non_monitoring_callbacks: list[Callback] = [
            callback
            for callback in callbacks
            if not isinstance(callback, (EarlyStopping, Checkpoint))
        ]
        evaluation_setup: _EvaluationSetup = self._prepare_evaluation(
            trainer,
            module,
            state
        )
        combined_loader: CombinedLoader = evaluation_setup[0]
        running_stage: RunningStage = evaluation_setup[1]
        max_batches: int | None = evaluation_setup[2]
        sequential: bool = evaluation_setup[3]
        num_dataloaders: int = evaluation_setup[4]
        include_dataloader_idx: bool = evaluation_setup[5]
        if max_batches == 0:
            module._reset_logged_metrics()
            return self.results
        self._verify_dataloader_idx_requirements(module, include_dataloader_idx)

        training_modes: dict[str, bool] = module._capture_training_modes()
        self._set_evaluation_mode(trainer, module)
        self._run_stage_start(trainer, module, callbacks)
        result_count: int = num_dataloaders if sequential else 1
        epoch_accumulators: list[_EpochAccumulator] = [
            {} for _ in range(result_count)
        ]
        self._run_batches(
            trainer,
            module,
            callbacks,
            combined_loader,
            max_batches,
            sequential,
            include_dataloader_idx,
            running_stage,
            epoch_accumulators
        )

        self._publish_epoch_metrics(
            trainer,
            state,
            running_stage,
            epoch_accumulators
        )

        self._run_stage_end(
            trainer,
            module,
            state,
            running_stage,
            callbacks,
            monitoring_callbacks,
            non_monitoring_callbacks,
            training_modes
        )

        module._reset_logged_metrics()
        trainer._raise_if_sigterm()
        return self.results

    def _reset_results(self) -> None:
        self.epoch_metrics: dict[str, float] = {}
        self._callback_epoch_metrics: dict[str, torch.Tensor] = {}
        self._logger_epoch_metrics: dict[str, torch.Tensor] = {}
        self._progress_bar_epoch_metrics: dict[str, float] = {}
        self.results: list[dict[str, float]] = []

    def _publish_epoch_metrics(
        self,
        trainer: syntheticmind.Trainer,
        state: TrainerState,
        running_stage: RunningStage,
        epoch_accumulators: list[_EpochAccumulator]
    ) -> None:
        self.results: list[dict[str, float]] = [
            self._reduce_epoch_metrics(epoch_accumulator)
            for epoch_accumulator in epoch_accumulators
        ]
        self.epoch_metrics: dict[str, float] = {
            name: value
            for result in self.results
            for name, value in result.items()
        }
        if not trainer.sanity_checking:
            trainer._update_progress_bar_metrics(
                running_stage,
                self._progress_bar_epoch_metrics
            )
            trainer._update_logged_metrics(self._logger_epoch_metrics)
        if trainer.loggers and self._logger_epoch_metrics:
            trainer._log_metrics(
                self._logger_epoch_metrics,
                step=state.global_step
            )

    def _run_stage_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        state: TrainerState,
        running_stage: RunningStage,
        callbacks: list[Callback],
        monitoring_callbacks: list[Callback],
        non_monitoring_callbacks: list[Callback],
        training_modes: dict[str, bool]
    ) -> None:
        if self.stage == "validate":
            self._run_validation_end(
                trainer,
                module,
                state,
                running_stage,
                callbacks,
                monitoring_callbacks,
                non_monitoring_callbacks,
                training_modes
            )
            return
        self._run_test_end(
            trainer,
            module,
            state,
            running_stage,
            callbacks,
            training_modes
        )

    def _run_validation_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        state: TrainerState,
        running_stage: RunningStage,
        callbacks: list[Callback],
        monitoring_callbacks: list[Callback],
        non_monitoring_callbacks: list[Callback],
        training_modes: dict[str, bool]
    ) -> None:
        trainer._logger_connector.callback_metrics.update(
            self._callback_epoch_metrics
        )
        trainer._hook_dispatcher.call_callbacks(
            module,
            "on_validation_epoch_end",
            callback_group=non_monitoring_callbacks
        )
        trainer._hook_dispatcher.call_module(
            module,
            "on_validation_epoch_end",
            module.on_validation_epoch_end
        )
        self._merge_epoch_end_hook_metrics(module, state, running_stage)
        trainer._hook_dispatcher.call_callbacks(
            module,
            "on_validation_epoch_end",
            callback_group=monitoring_callbacks
        )
        trainer._hook_dispatcher.call_callbacks(
            module,
            "on_validation_end",
            callback_group=callbacks
        )
        trainer._hook_dispatcher.call_module(
            module,
            "on_validation_end",
            module.on_validation_end
        )
        if trainer._hook_dispatcher.is_overridden(
            "on_validation_model_train",
            module,
            Module
        ):
            trainer._hook_dispatcher.call_module(
                module,
                "on_validation_model_train",
                module.on_validation_model_train
            )
        else:
            module._restore_training_modes(training_modes)

    def _run_test_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        state: TrainerState,
        running_stage: RunningStage,
        callbacks: list[Callback],
        training_modes: dict[str, bool]
    ) -> None:
        trainer._logger_connector.callback_metrics.update(
            self._callback_epoch_metrics
        )
        trainer._hook_dispatcher.call_callbacks(
            module,
            "on_test_epoch_end",
            callback_group=callbacks
        )
        trainer._hook_dispatcher.call_module(
            module,
            "on_test_epoch_end",
            module.on_test_epoch_end
        )
        self._merge_epoch_end_hook_metrics(module, state, running_stage)
        trainer._hook_dispatcher.call_callbacks(
            module,
            "on_test_end",
            callback_group=callbacks
        )
        trainer._hook_dispatcher.call_module(
            module,
            "on_test_end",
            module.on_test_end
        )
        if trainer._hook_dispatcher.is_overridden(
            "on_test_model_train",
            module,
            Module
        ):
            trainer._hook_dispatcher.call_module(
                module,
                "on_test_model_train",
                module.on_test_model_train
            )
        else:
            module._restore_training_modes(training_modes)

    def _merge_epoch_end_hook_metrics(
        self,
        module: Module,
        state: TrainerState,
        running_stage: RunningStage
    ) -> None:
        batch_epoch_metrics: dict[str, float] = dict(self.epoch_metrics)
        self._consume_epoch_end_hook_metrics(module, state, running_stage)
        hook_epoch_metrics: dict[str, float] = {
            name: value
            for name, value in self.epoch_metrics.items()
            if name not in batch_epoch_metrics or batch_epoch_metrics[name] != value
        }
        result: dict[str, float]
        for result in self.results:
            result.update(hook_epoch_metrics)
        self.trainer._logger_connector.callback_metrics.update(
            self._callback_epoch_metrics
        )

    def _prepare_evaluation(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        state: TrainerState
    ) -> _EvaluationSetup:
        running_stage: RunningStage = (
            "validating" if self.stage == "validate" else "testing"
        )
        active_running_stage: RunningStage = (
            "sanity_checking" if trainer.sanity_checking else running_stage
        )
        state.set_running_stage(active_running_stage)
        module._running_stage: RunningStage | None = active_running_stage
        if not trainer.sanity_checking:
            trainer._reset_progress_bar_metrics(running_stage)
        combined_loader: CombinedLoader = self._get_combined_loader()
        dataloader: Iterable[Batch]
        for dataloader in combined_loader.flattened:
            if (
                isinstance(dataloader, DataLoader)
                and isinstance(dataloader.sampler, DistributedSampler)
            ):
                dataloader.sampler.set_epoch(state.current_epoch)
        num_batches: list[int | float] = (
            trainer.num_val_batches
            if self.stage == "validate"
            else trainer.num_test_batches
        )
        if trainer.sanity_checking:
            num_batches: list[int | float] = [
                min(trainer.num_sanity_val_steps, batches)
                for batches in num_batches
            ]
        combined_loader.limits: list[int | float | None] | None = num_batches
        max_batches: int | None = combined_loader._length()
        trainer._raise_if_sigterm()
        sequential: bool = combined_loader.mode == "sequential"
        num_dataloaders: int = len(combined_loader.flattened)
        include_dataloader_idx: bool = sequential and num_dataloaders > 1
        return (
            combined_loader,
            running_stage,
            max_batches,
            sequential,
            num_dataloaders,
            include_dataloader_idx
        )

    def _set_evaluation_mode(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        if self.stage == "validate":
            trainer._hook_dispatcher.call_module(
                module,
                "on_validation_model_eval",
                module.on_validation_model_eval
            )
            return
        trainer._hook_dispatcher.call_module(
            module,
            "on_test_model_eval",
            module.on_test_model_eval
        )

    def _run_stage_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        callbacks: list[Callback]
    ) -> None:
        if self.stage == "validate":
            trainer._hook_dispatcher.call_callbacks(
                module,
                "on_validation_start",
                callback_group=callbacks
            )
            trainer._hook_dispatcher.call_module(
                module,
                "on_validation_start",
                module.on_validation_start
            )
            trainer._hook_dispatcher.call_callbacks(
                module,
                "on_validation_epoch_start",
                callback_group=callbacks
            )
            trainer._hook_dispatcher.call_module(
                module,
                "on_validation_epoch_start",
                module.on_validation_epoch_start
            )
            return
        trainer._hook_dispatcher.call_callbacks(
            module,
            "on_test_start",
            callback_group=callbacks
        )
        trainer._hook_dispatcher.call_module(
            module,
            "on_test_start",
            module.on_test_start
        )
        trainer._hook_dispatcher.call_callbacks(
            module,
            "on_test_epoch_start",
            callback_group=callbacks
        )
        trainer._hook_dispatcher.call_module(
            module,
            "on_test_epoch_start",
            module.on_test_epoch_start
        )

    def _run_batches(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        callbacks: list[Callback],
        combined_loader: CombinedLoader,
        max_batches: int | None,
        sequential: bool,
        include_dataloader_idx: bool,
        running_stage: RunningStage,
        epoch_accumulators: list[_EpochAccumulator]
    ) -> None:
        with trainer.profiler.profile(f"setup_{self.stage}_dataloader"):
            data_fetcher: _DataFetcher[tuple[Batch, int, int]] = _DataFetcher(
                combined_loader,
                max_batches,
                sized=combined_loader._has_sized_iterables()
            )
            dataloader_iterator: Iterator[tuple[Batch, int, int]] = iter(
                data_fetcher
            )
        while not data_fetcher.done:
            trainer._raise_if_sigterm()
            try:
                dataloader_prefix: str = (
                    "val" if self.stage == "validate" else "test"
                )
                with trainer.profiler.profile(
                    f"[{type(self).__name__}].{dataloader_prefix}_next"
                ):
                    fetched: tuple[Batch, int, int] = next(dataloader_iterator)
            except StopIteration:
                break
            self._advance_batch(
                trainer,
                module,
                callbacks,
                fetched,
                sequential,
                include_dataloader_idx,
                running_stage,
                epoch_accumulators
            )

    def _advance_batch(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        callbacks: list[Callback],
        fetched: tuple[Batch, int, int],
        sequential: bool,
        include_dataloader_idx: bool,
        running_stage: RunningStage,
        epoch_accumulators: list[_EpochAccumulator]
    ) -> None:
        batch: Batch = trainer.precision_plugin.convert_input(fetched[0])
        batch_idx: int = fetched[1]
        dataloader_idx: int = fetched[2]
        hook_dataloader_idx: int | None = (
            dataloader_idx if include_dataloader_idx else None
        )
        module._current_dataloader_idx: int | None = hook_dataloader_idx
        batch: Batch = trainer._transfer_batch(
            module,
            batch,
            trainer.strategy.root_device,
            dataloader_idx
        )
        hook_arguments: _BatchHookArguments = self._batch_hook_arguments(
            batch,
            batch_idx,
            hook_dataloader_idx
        )
        step_hook: _EvaluationStepHook = (
            module.validation_step
            if self.stage == "validate"
            else module.test_step
        )
        step_arguments: _BatchHookArguments = self._step_arguments(
            module,
            step_hook,
            batch,
            batch_idx,
            hook_dataloader_idx
        )
        self._run_batch_start(trainer, module, callbacks, hook_arguments)
        outputs: StepOutput = self._run_evaluation_step(
            trainer,
            module,
            step_arguments
        )
        trainer._preview_logged_metrics(module, running_stage)
        self._run_batch_end(
            trainer,
            module,
            callbacks,
            outputs,
            hook_arguments
        )
        result_idx: int = dataloader_idx if sequential else 0
        self._accumulate_metrics(
            module,
            epoch_accumulators[result_idx],
            trainer.state,
            running_stage,
            batch
        )

    def _batch_hook_arguments(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int | None
    ) -> _BatchHookArguments:
        if dataloader_idx is None:
            return batch, batch_idx
        return batch, batch_idx, dataloader_idx

    def _step_arguments(
        self,
        module: Module,
        step_hook: _EvaluationStepHook,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int | None
    ) -> _BatchHookArguments:
        if module._hook_accepts_parameter(
            step_hook,
            "batch_idx",
            minimum_arguments=2
        ):
            return self._batch_hook_arguments(batch, batch_idx, dataloader_idx)
        if dataloader_idx is None:
            return tuple([batch])
        return batch, dataloader_idx

    def _run_batch_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        callbacks: list[Callback],
        hook_arguments: _BatchHookArguments
    ) -> None:
        if self.stage == "validate":
            trainer._hook_dispatcher.call_callbacks(
                module,
                "on_validation_batch_start",
                *hook_arguments,
                callback_group=callbacks
            )
            trainer._hook_dispatcher.call_module(
                module,
                "on_validation_batch_start",
                module.on_validation_batch_start,
                *hook_arguments
            )
            return
        trainer._hook_dispatcher.call_callbacks(
            module,
            "on_test_batch_start",
            *hook_arguments,
            callback_group=callbacks
        )
        trainer._hook_dispatcher.call_module(
            module,
            "on_test_batch_start",
            module.on_test_batch_start,
            *hook_arguments
        )

    def _run_evaluation_step(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        step_arguments: _BatchHookArguments
    ) -> StepOutput:
        model: nn.Module | None = trainer.strategy.model
        assert model is not None
        step_context: AbstractContextManager[None] = (
            trainer.precision_plugin.val_step_context()
            if self.stage == "validate"
            else trainer.precision_plugin.test_step_context()
        )
        with step_context:
            if self.stage == "validate":
                with module._hook_context("validation_step"):
                    return trainer._hook_dispatcher.call_strategy(
                        "validation_step",
                        trainer.strategy.validation_step,
                        model,
                        *step_arguments
                    )
            with module._hook_context("test_step"):
                return trainer._hook_dispatcher.call_strategy(
                    "test_step",
                    trainer.strategy.test_step,
                    model,
                    *step_arguments
                )

    def _run_batch_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        callbacks: list[Callback],
        outputs: StepOutput,
        hook_arguments: _BatchHookArguments
    ) -> None:
        if self.stage == "validate":
            trainer._hook_dispatcher.call_callbacks(
                module,
                "on_validation_batch_end",
                outputs,
                *hook_arguments,
                callback_group=callbacks
            )
            trainer._hook_dispatcher.call_module(
                module,
                "on_validation_batch_end",
                module.on_validation_batch_end,
                outputs,
                *hook_arguments
            )
            return
        trainer._hook_dispatcher.call_callbacks(
            module,
            "on_test_batch_end",
            outputs,
            *hook_arguments,
            callback_group=callbacks
        )
        trainer._hook_dispatcher.call_module(
            module,
            "on_test_batch_end",
            module.on_test_batch_end,
            outputs,
            *hook_arguments
        )

    def _verify_dataloader_idx_requirements(
        self,
        module: Module,
        include_dataloader_idx: bool
    ) -> None:
        prefix: str = "validation" if self.stage == "validate" else "test"
        for hook_name in (
            f"{prefix}_step",
            f"on_{prefix}_batch_start",
            f"on_{prefix}_batch_end"
        ):
            hook: _EvaluationHook | None = cast(
                _EvaluationHook | None,
                getattr(module, hook_name, None)
            )
            if hook is None or not callable(hook):
                continue
            parameters: dict[str, inspect.Parameter] = dict(
                inspect.signature(hook).parameters
            )
            has_varargs: bool = any(
                parameter.kind is inspect.Parameter.VAR_POSITIONAL
                for parameter in parameters.values()
            )
            dataloader_parameter: inspect.Parameter | None = parameters.get(
                "dataloader_idx"
            )
            if (
                include_dataloader_idx
                and dataloader_parameter is None
                and not has_varargs
            ):
                raise RuntimeError(
                    f"Multiple {self.stage} dataloaders require `dataloader_idx` in "
                    f"{type(module).__qualname__}.{hook_name}()."
                )
            if (
                not include_dataloader_idx
                and dataloader_parameter is not None
                and dataloader_parameter.default is inspect.Parameter.empty
            ):
                raise RuntimeError(
                    f"A single {self.stage} dataloader requires "
                    f"{type(module).__qualname__}.{hook_name}() to remove `dataloader_idx` "
                    "or give it the default value 0."
                )

    def _get_combined_loader(self) -> CombinedLoader:
        trainer: syntheticmind.Trainer = self.trainer
        if self.stage == "validate":
            if trainer.state.stage == "fit":
                trainer._ensure_val_dataloader()
            assert trainer._data_connector.val_combined_loader is not None
            return trainer._data_connector.val_combined_loader
        assert trainer._data_connector.test_combined_loader is not None
        return trainer._data_connector.test_combined_loader

    def _accumulate_metrics(
        self,
        module: Module,
        epoch_accum: dict[str, tuple[LoggedMetric, MetricAccumulator]],
        state: TrainerState,
        running_stage: RunningStage,
        batch: Batch
    ) -> None:
        logged: list[LoggedMetric] = module.logged_metrics
        if not logged:
            module._reset_logged_metrics()
            return

        step_metrics: dict[str, torch.Tensor] = {}
        progress_bar_step_metrics: dict[str, float] = {}
        for metric in logged:
            if metric.on_step:
                value: torch.Tensor | None = metric.compute_step(
                    self.trainer.strategy.root_device,
                    self.trainer.strategy.reduce
                )
                if value is not None:
                    step_name: str = metric.output_name(on_step=True)
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
                    accumulator.update(
                        metric.local_value(),
                        batch_size=metric.batch_size,
                        batch=batch
                    )

        if not self.trainer.sanity_checking:
            self.trainer._update_logged_metrics(step_metrics)
            if (
                step_metrics
                and self.trainer.loggers
                and self.trainer.is_global_zero
            ):
                self.trainer._log_metrics(step_metrics, step=state.global_step)
            self.trainer._update_progress_bar_metrics(
                running_stage,
                progress_bar_step_metrics
            )

        module._reset_logged_metrics()

    def _consume_epoch_end_hook_metrics(
        self,
        module: Module,
        state: TrainerState,
        running_stage: RunningStage
    ) -> None:
        logged: list[LoggedMetric] = module.logged_metrics
        if not logged:
            return
        hook_logger_metrics: dict[str, torch.Tensor] = {}
        hook_progress_bar_metrics: dict[str, float] = {}
        for metric in logged:
            value: torch.Tensor = metric.compute_epoch(
                self.trainer.strategy.root_device,
                self.trainer.strategy.reduce
            )
            epoch_name: str = metric.output_name(on_step=False)
            if metric.on_epoch:
                self._callback_epoch_metrics[metric.base_name()] = value
                self._callback_epoch_metrics[epoch_name] = value
            if metric.logger:
                if metric.on_epoch:
                    self.epoch_metrics[epoch_name] = self.trainer._metric_to_float(value)
                hook_logger_metrics[epoch_name] = value
            if metric.progress_bar:
                hook_progress_bar_metrics[epoch_name] = (
                    self.trainer._metric_to_float(value)
                )
        if hook_logger_metrics and self.trainer.loggers and self.trainer.is_global_zero:
            self.trainer._log_metrics(hook_logger_metrics, step=state.global_step)
        if not self.trainer.sanity_checking:
            self.trainer._update_logged_metrics(hook_logger_metrics)
            self.trainer._update_progress_bar_metrics(
                running_stage,
                hook_progress_bar_metrics
            )
        module._reset_logged_metrics()

    def _reduce_epoch_metrics(
        self,
        epoch_accum: dict[str, tuple[LoggedMetric, MetricAccumulator]]
    ) -> dict[str, float]:
        reduced: dict[str, float] = {}
        for metric, accumulator in epoch_accum.values():
            value: torch.Tensor = accumulator.compute(
                self.trainer.strategy.root_device,
                self.trainer.strategy.reduce
            )
            if not metric.enable_graph:
                value: torch.Tensor = value.detach()
            epoch_name: str = metric.output_name(on_step=False)
            self._callback_epoch_metrics[metric.base_name()] = value
            self._callback_epoch_metrics[epoch_name] = value
            if metric.logger:
                reduced[epoch_name] = self.trainer._metric_to_float(value)
                self._logger_epoch_metrics[epoch_name] = value
            if metric.progress_bar:
                self._progress_bar_epoch_metrics[epoch_name] = (
                    self.trainer._metric_to_float(value)
                )
        for _, accumulator in epoch_accum.values():
            accumulator.reset()
        return reduced
