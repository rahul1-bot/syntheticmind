from collections.abc import Callable, Iterable
from typing import cast, override

import syntheticmind
import torch
from torch.utils.data import DataLoader, DistributedSampler

from syntheticmind.callbacks.callback import Callback
from syntheticmind.callbacks.checkpointing import Checkpoint
from syntheticmind.callbacks.early_stopping import EarlyStopping
from syntheticmind.core.metric import LoggedMetric
from syntheticmind.core.module import Module
from syntheticmind.loops.loop import Loop
from syntheticmind.loops.training_epoch_loop import TrainingEpochLoop
from syntheticmind.trainer.states import TrainerState
from syntheticmind.combined_loader import CombinedLoader
from syntheticmind.types import Batch, RunningStage

__all__: list[str] = ["FitLoop"]


class FitLoop(Loop):
    @override
    def __init__(self, min_epochs: int, max_epochs: int) -> None:
        super().__init__()
        self.min_epochs: int = min_epochs
        self.max_epochs: int = max_epochs
        self.epoch_loop: TrainingEpochLoop | None = None

    def run(self) -> None:
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
        assert self.epoch_loop is not None
        max_train_batches: int | None = trainer._data_connector.total_train_batches
        trainer._raise_if_sigterm()
        if (
            self._max_epochs_reached(state)
            or self._max_steps_reached(state)
            or max_train_batches == 0
            or (trainer.should_stop and self._can_stop_early(state))
        ):
            return

        self._run_training_start(trainer, module, callbacks)

        while not self._max_epochs_reached(state) and not self._max_steps_reached(state):
            trainer._raise_if_sigterm()
            if self.should_stop or (trainer.should_stop and self._can_stop_early(state)):
                break
            self._run_epoch(
                trainer,
                module,
                state,
                callbacks,
                monitoring_callbacks,
                non_monitoring_callbacks
            )

        self._run_training_end(trainer, module, callbacks)

    def _run_training_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        callbacks: list[Callback]
    ) -> None:
        trainer._hook_dispatcher.call_callbacks(
            module,
            "on_train_start",
            callback_group=callbacks
        )
        trainer._hook_dispatcher.call_module(
            module,
            "on_train_start",
            module.on_train_start
        )

    def _run_epoch(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        state: TrainerState,
        callbacks: list[Callback],
        monitoring_callbacks: list[Callback],
        non_monitoring_callbacks: list[Callback]
    ) -> None:
        assert self.epoch_loop is not None
        epoch: int = state.current_epoch
        trainer._ensure_train_dataloader()
        self._set_sampler_epoch(trainer, epoch)
        state.set_running_stage("training")
        module._running_stage: RunningStage | None = "training"
        trainer._reset_progress_bar_metrics("training")
        trainer._reset_progress_bar_metrics("validating")
        module.train()
        self.epoch_loop.is_last_batch: bool = False
        self._run_epoch_start(trainer, module, callbacks)
        with trainer.profiler.profile("run_training_epoch"):
            self.epoch_loop.run()
        if not self.epoch_loop.epoch_completed:
            return
        self._run_epoch_end(
            trainer,
            module,
            state,
            monitoring_callbacks,
            non_monitoring_callbacks
        )
        state.increment_epoch()
        module._current_epoch: int = state.current_epoch
        trainer._raise_if_sigterm()
        if trainer.should_stop and self._can_stop_early(state):
            self.should_stop: bool = True

    def _set_sampler_epoch(
        self,
        trainer: syntheticmind.Trainer,
        epoch: int
    ) -> None:
        combined_loader: CombinedLoader | None = (
            trainer._data_connector.train_combined_loader
        )
        if combined_loader is None:
            return
        train_dataloader: DataLoader | Iterable[Batch]
        for train_dataloader in combined_loader.flattened:
            if isinstance(
                train_dataloader,
                DataLoader
            ) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

    def _run_epoch_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        callbacks: list[Callback]
    ) -> None:
        trainer._hook_dispatcher.call_callbacks(
            module,
            "on_train_epoch_start",
            callback_group=callbacks
        )
        trainer._hook_dispatcher.call_module(
            module,
            "on_train_epoch_start",
            module.on_train_epoch_start
        )

    def _run_epoch_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        state: TrainerState,
        monitoring_callbacks: list[Callback],
        non_monitoring_callbacks: list[Callback]
    ) -> None:
        trainer._hook_dispatcher.call_callbacks(
            module,
            "on_train_epoch_end",
            callback_group=non_monitoring_callbacks
        )
        trainer._hook_dispatcher.call_module(
            module,
            "on_train_epoch_end",
            module.on_train_epoch_end
        )
        self._flush_hook_metrics(module, trainer, state)
        if module.automatic_optimization:
            assert self.epoch_loop is not None
            self.epoch_loop.update_plateau_schedulers(
                trainer,
                module,
                "epoch",
                state.current_epoch + 1
            )
        trainer._hook_dispatcher.call_callbacks(
            module,
            "on_train_epoch_end",
            callback_group=monitoring_callbacks
        )
        module._reset_logged_metrics()

    def _run_training_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        callbacks: list[Callback]
    ) -> None:
        trainer._hook_dispatcher.call_callbacks(
            module,
            "on_train_end",
            callback_group=callbacks
        )
        trainer._hook_dispatcher.call_module(module, "on_train_end", module.on_train_end)
        trainer._raise_if_sigterm()

    def _max_epochs_reached(self, state: TrainerState) -> bool:
        return self.max_epochs != -1 and state.current_epoch >= self.max_epochs

    def _max_steps_reached(self, state: TrainerState) -> bool:
        assert self.epoch_loop is not None
        return self.epoch_loop.max_steps != -1 and state.global_step >= self.epoch_loop.max_steps

    def _can_stop_early(self, state: TrainerState) -> bool:
        assert self.epoch_loop is not None
        return (
            state.current_epoch >= self.min_epochs
            and (
                self.epoch_loop.min_steps is None
                or state.global_step >= self.epoch_loop.min_steps
            )
        )

    def _flush_hook_metrics(
        self,
        module: Module,
        trainer: syntheticmind.Trainer,
        state: TrainerState
    ) -> None:
        logged: list[LoggedMetric] = module.logged_metrics
        if not logged:
            return
        log_metrics: dict[str, torch.Tensor] = {}
        callback_metrics: dict[str, torch.Tensor] = {}
        progress_bar_metrics: dict[str, float] = {}
        for metric in logged:
            val: torch.Tensor = metric.compute_epoch(
                trainer.strategy.root_device,
                trainer.strategy.reduce
            )
            metric_name: str = metric.output_name(on_step=False)
            if metric.on_step or metric.on_epoch:
                callback_metrics[metric.base_name()] = val
                callback_metrics[metric_name] = val
            if metric.logger:
                log_metrics[metric_name] = val
            if metric.progress_bar:
                progress_bar_metrics[metric_name] = trainer._metric_to_float(val)
            if metric.is_metric:
                reset: Callable[[], None] = cast(
                    Callable[[], None],
                    getattr(metric.value, "reset")
                )
                reset()
        trainer._logger_connector.callback_metrics.update(callback_metrics)
        trainer._update_logged_metrics(log_metrics)
        trainer._update_progress_bar_metrics("training", progress_bar_metrics)
        if log_metrics and trainer.loggers:
            trainer._log_metrics(log_metrics, step=state.global_step)
        module._reset_logged_metrics()
