# This module:
# 1. Orchestrates the epoch sequence of a fit run: per-epoch dataloader
#    management, the train-epoch hook pairs, delegation to the training epoch
#    loop, epoch-end validation, epoch-interval scheduler stepping, and epoch
#    counter advancement
# 2. Propagates cooperative stop requests raised during an epoch into
#    termination of the epoch sequence
#
# Design decisions:
# - The epoch range starts at the trainer state's current epoch rather than
#   zero, so a run restored from a checkpoint resumes at the correct epoch
#   without additional bookkeeping here
# - When dataloader reloading is configured, the dataloaders are rebuilt from
#   the datamodule at the configured epoch interval and re-wrapped with a
#   DistributedSampler where distributed execution requires one, and the
#   trainer's cached batch totals are refreshed to match
# - DistributedSampler.set_epoch is called before each epoch because the
#   sampler derives its shuffling permutation from the epoch number; omitting
#   the call would repeat the first epoch's shuffle order indefinitely
# - The module's on_train_epoch_end hook runs before the callback epoch-end
#   hooks, and metrics logged inside it are flushed to the logger first, so
#   callbacks observe a fully updated metric state at their epoch boundary
# - Epoch-interval schedulers advance after epoch-end validation because the
#   plateau scheduler consumes a monitored validation metric; a missing
#   monitor key raises a configuration error that lists the available metric
#   names instead of silently skipping the scheduler step
# - Gradients are cleared before epoch-end validation so validation never
#   observes stale training gradients, and the module is returned to training
#   mode afterwards
#
# Author: Rahul Sawhney

from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, DistributedSampler

from syntheticmind.callbacks.callback import Callback
from syntheticmind.callbacks.checkpoint import Checkpoint
from syntheticmind.callbacks.early_stopping import EarlyStopping
from syntheticmind.core.module import LoggedMetric, Module
from syntheticmind.loops.loop import Loop
from syntheticmind.loops.training_epoch_loop import TrainingEpochLoop
from syntheticmind.state.trainer_state import TrainerState
from syntheticmind.utilities.combined_loader import CombinedLoader
from syntheticmind.utilities.distributed import DistributedUtils

if TYPE_CHECKING:
    from syntheticmind.core.trainer import Trainer

__all__: list[str] = ["FitLoop"]


class FitLoop(Loop):
    # Loop that drives the full fit run. The trainer constructs it with the
    # epoch budget and attaches the configured TrainingEpochLoop before
    # execution; each run() invocation then iterates epochs until the budget
    # is exhausted or a cooperative stop is raised.
    def __init__(self, min_epochs: int, max_epochs: int) -> None:
        # Binds the epoch budget. The epoch-loop slot is populated by the
        # trainer during its own construction, after the epoch loop has been
        # configured from the trainer arguments.
        super().__init__()
        self.min_epochs: int = min_epochs
        self.max_epochs: int = max_epochs
        self.epoch_loop: TrainingEpochLoop | None = None

    def run(self) -> None:
        # Executes the epoch sequence. Each iteration performs: the optional
        # dataloader reload, distributed-sampler epoch seeding, the
        # train-epoch-start hook pair, one full training epoch through the
        # epoch loop, the train-epoch-end hook pair with hook-metric flushing,
        # optional epoch-end validation, epoch-interval scheduler stepping,
        # and the epoch counter advancement with stop-flag propagation.
        trainer: Trainer = self.trainer
        module: Module | None = trainer.module
        assert module is not None
        state: TrainerState = trainer.state
        callbacks: list[Callback] = trainer.callbacks
        monitoring_callbacks: list[Callback] = [
            callback for callback in callbacks if isinstance(callback, (EarlyStopping, Checkpoint))
        ]
        non_monitoring_callbacks: list[Callback] = [
            callback for callback in callbacks if not isinstance(callback, (EarlyStopping, Checkpoint))
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

        trainer._call_callback_hooks(
            module,
            "on_train_start",
            callback_group=callbacks
        )
        trainer._call_module_hook(
            module,
            "on_train_start",
            module.on_train_start
        )

        while not self._max_epochs_reached(state) and not self._max_steps_reached(state):
            trainer._raise_if_sigterm()
            if self.should_stop or (trainer.should_stop and self._can_stop_early(state)):
                break
            epoch: int = state.current_epoch

            # Optional dataloader reload: rebuild from the datamodule at the
            # configured epoch interval, re-wrap for distributed execution,
            # and refresh the trainer's cached batch totals to match.
            trainer._ensure_train_dataloader()

            # DistributedSampler derives its shuffle permutation from the
            # epoch number; seeding it every epoch prevents repeating the
            # first epoch's shuffle order indefinitely.
            combined_loader: CombinedLoader | None = trainer._data_connector.train_combined_loader
            if combined_loader is not None:
                for train_dl in combined_loader.flattened:
                    if (
                        isinstance(train_dl, DataLoader)
                        and isinstance(train_dl.sampler, DistributedSampler)
                    ):
                        train_dl.sampler.set_epoch(epoch)

            state.set_running_stage("training")
            module._running_stage = "training"
            trainer._reset_progress_bar_metrics("training")
            trainer._reset_progress_bar_metrics("validating")
            module.train()
            self.epoch_loop.is_last_batch = False

            trainer._call_callback_hooks(
                module,
                "on_train_epoch_start",
                callback_group=callbacks
            )
            trainer._call_module_hook(
                module,
                "on_train_epoch_start",
                module.on_train_epoch_start
            )

            with trainer.profiler.profile("run_training_epoch"):
                self.epoch_loop.run()

            trainer._call_callback_hooks(
                module,
                "on_train_epoch_end",
                callback_group=non_monitoring_callbacks
            )
            trainer._call_module_hook(
                module,
                "on_train_epoch_end",
                module.on_train_epoch_end
            )
            self._flush_hook_metrics(module, trainer, state)
            trainer._call_callback_hooks(
                module,
                "on_train_epoch_end",
                callback_group=monitoring_callbacks
            )

            module._reset_logged_metrics()

            # Epoch-end validation runs with cleared gradients so it never
            # observes stale training gradients, and the module returns to
            # training mode afterwards.

            # Epoch-interval schedulers advance after epoch-end validation
            # because the plateau family consumes a monitored validation
            # metric; a missing monitor key is a configuration error, never a
            # silent skip.
            if module.automatic_optimization:
                self.epoch_loop.update_lr_schedulers(
                    trainer,
                    module,
                    "epoch",
                    state.current_epoch + 1,
                    update_plateau_schedulers=True
                )

            state.increment_epoch()
            module._current_epoch = state.current_epoch
            trainer._raise_if_sigterm()

            # A stop raised at trainer level (early stopping, strategies)
            # propagates to the loop-local flag, ending the epoch sequence
            # cooperatively at the epoch boundary.
            if trainer.should_stop and self._can_stop_early(state):
                self.should_stop: bool = True

        trainer._call_callback_hooks(
            module,
            "on_train_end",
            callback_group=callbacks
        )
        trainer._call_module_hook(module, "on_train_end", module.on_train_end)
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

    def _flush_hook_metrics(self, module: Module, trainer: Trainer, state: TrainerState) -> None:
        # Emits metrics logged inside the module's on_train_epoch_end hook.
        # These arrive after the epoch loop has already dispatched its epoch
        # metrics, so they are flushed here directly to the logger from rank
        # zero and the module's logging buffer is cleared afterwards.
        logged: list[LoggedMetric] = module.logged_metrics
        if not logged:
            return
        log_metrics: dict[str, torch.Tensor] = {}
        callback_metrics: dict[str, torch.Tensor] = {}
        progress_bar_metrics: dict[str, float] = {}
        for metric in logged:
            val: torch.Tensor = metric.compute_epoch(trainer.strategy.root_device)
            metric_name: str = metric.output_name(on_step=False)
            if metric.on_step or metric.on_epoch:
                callback_metrics[metric.base_name()] = val
                callback_metrics[metric_name] = val
            if metric.logger:
                log_metrics[metric_name] = val
            if metric.progress_bar:
                progress_bar_metrics[metric_name] = trainer._metric_to_float(val)
            if metric.is_metric:
                reset: object = getattr(metric.value, "reset")
                if callable(reset):
                    reset()
        trainer._callback_metrics.update(callback_metrics)
        trainer._update_logged_metrics(log_metrics)
        trainer._update_progress_bar_metrics("training", progress_bar_metrics)
        if log_metrics and trainer.loggers and DistributedUtils.is_rank_zero():
            trainer._log_metrics(log_metrics, step=state.global_step)
        module._reset_logged_metrics()
