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
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset

from syntheticmind.callbacks.callback import Callback
from syntheticmind.callbacks.early_stopping import EarlyStopping
from syntheticmind.callbacks.model_checkpoint import ModelCheckpoint
from syntheticmind.core.module import LoggedMetric, Module
from syntheticmind.loops.evaluation_loop import EvaluationLoop
from syntheticmind.loops.loop import Loop
from syntheticmind.loops.training_epoch_loop import TrainingEpochLoop
from syntheticmind.state.trainer_state import TrainerState
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
            callback for callback in callbacks if isinstance(callback, (EarlyStopping, ModelCheckpoint))
        ]
        non_monitoring_callbacks: list[Callback] = [
            callback for callback in callbacks if not isinstance(callback, (EarlyStopping, ModelCheckpoint))
        ]
        assert self.epoch_loop is not None
        total_train_batches: int | None = trainer._total_train_batches
        max_train_batches: int | None = self.epoch_loop._resolve_limit_batches(total_train_batches)
        if (
            self._max_epochs_reached(state)
            or self._max_steps_reached(state)
            or max_train_batches == 0
            or (trainer.should_stop and self._can_stop_early(state))
        ):
            return

        with module._hook_context("on_train_start"):
            for callback in callbacks:
                callback.on_train_start(trainer, module)
            module.on_train_start()

        while not self._max_epochs_reached(state) and not self._max_steps_reached(state):
            if self.should_stop or (trainer.should_stop and self._can_stop_early(state)):
                break
            epoch: int = state.current_epoch

            # Optional dataloader reload: rebuild from the datamodule at the
            # configured epoch interval, re-wrap for distributed execution,
            # and refresh the trainer's cached batch totals to match.
            if trainer.reload_dataloaders_every_n_epochs > 0 and epoch > 0:
                if epoch % trainer.reload_dataloaders_every_n_epochs == 0:
                    assert trainer.datamodule is not None
                    trainer.train_dataloader: DataLoader | None = trainer.datamodule.train_dataloader()

                    if trainer.use_distributed_sampler and trainer.strategy.is_distributed:
                        if not isinstance(trainer.train_dataloader.dataset, IterableDataset):
                            trainer.train_dataloader: DataLoader | None = trainer._wrap_with_distributed_sampler(
                                trainer.train_dataloader, shuffle=True
                            )
                    trainer._total_train_batches: int | None = trainer._safe_len(trainer.train_dataloader)

                    if trainer.val_dataloader is not None:
                        trainer.val_dataloader: DataLoader | None = trainer.datamodule.val_dataloader()
                        if trainer.use_distributed_sampler and trainer.strategy.is_distributed:
                            if not isinstance(trainer.val_dataloader.dataset, IterableDataset):
                                trainer.val_dataloader: DataLoader | None = trainer._wrap_with_distributed_sampler(
                                    trainer.val_dataloader, shuffle=False
                                )
                        trainer._total_val_batches: int | None = trainer._safe_len(trainer.val_dataloader)

            # DistributedSampler derives its shuffle permutation from the
            # epoch number; seeding it every epoch prevents repeating the
            # first epoch's shuffle order indefinitely.
            train_dl: DataLoader | None = trainer.train_dataloader  # type: ignore[type-arg]
            if train_dl is not None and isinstance(train_dl.sampler, DistributedSampler):
                train_dl.sampler.set_epoch(epoch)

            state.set_running_stage("training")
            module._running_stage: str | None = "training"
            trainer._reset_progress_bar_metrics("training")
            trainer._reset_progress_bar_metrics("validating")
            module.train()

            with module._hook_context("on_train_epoch_start"):
                for callback in callbacks:
                    callback.on_train_epoch_start(trainer, module)
                module.on_train_epoch_start()

            self.epoch_loop.run()

            val_dataloader: DataLoader | None = trainer.val_dataloader  # type: ignore[type-arg]
            if val_dataloader is not None and self.epoch_loop.validate_at_epoch_end:
                trainer._call_on_validation_model_zero_grad(module)
                validate_loop: EvaluationLoop = trainer.validate_loop
                validate_loop.run()
                state.set_running_stage("training")
                module._running_stage: str | None = "training"

            with module._hook_context("on_train_epoch_end"):
                for callback in non_monitoring_callbacks:
                    callback.on_train_epoch_end(trainer, module)
            with module._hook_context("on_train_epoch_end"):
                module.on_train_epoch_end()
            self._flush_hook_metrics(module, trainer, state)
            with module._hook_context("on_train_epoch_end"):
                for callback in monitoring_callbacks:
                    callback.on_train_epoch_end(trainer, module)

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
            module._current_epoch: int = state.current_epoch

            # A stop raised at trainer level (early stopping, strategies)
            # propagates to the loop-local flag, ending the epoch sequence
            # cooperatively at the epoch boundary.
            if trainer.should_stop and self._can_stop_early(state):
                self.should_stop: bool = True

        with module._hook_context("on_train_end"):
            for callback in callbacks:
                callback.on_train_end(trainer, module)
            module.on_train_end()

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
        logged: dict[str, LoggedMetric] = module.logged_metrics
        if not logged:
            return
        log_metrics: dict[str, float] = {}
        callback_metrics: dict[str, float] = {}
        progress_bar_metrics: dict[str, float] = {}
        for name, metric in logged.items():
            val: torch.Tensor | float = metric.value
            if isinstance(val, torch.Tensor):
                val: torch.Tensor | float = val.item()
            if metric.on_step or metric.on_epoch:
                callback_metrics[name] = val
            if metric.logger:
                log_metrics[name] = val
            if metric.progress_bar:
                progress_bar_metrics[name] = val
        trainer._callback_metrics.update(callback_metrics)
        trainer._update_progress_bar_metrics("training", progress_bar_metrics)
        if log_metrics and trainer.logger is not None and DistributedUtils.is_rank_zero():
            trainer.logger.log_metrics(log_metrics, step=state.global_step)
        module._reset_logged_metrics()
