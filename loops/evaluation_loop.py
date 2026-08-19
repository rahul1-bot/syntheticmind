# This module:
# 1. Executes one complete validation or test pass under torch.inference_mode:
#    stage-specific lifecycle hooks, the batch-transfer hook chain, step
#    execution, and metric accumulation with epoch-level reduction
# 2. Serves both the standalone validate and test entry points and the
#    validation passes scheduled by the fit loop
#
# Design decisions:
# - One loop class serves both evaluation stages. The stage field selects the
#    dataloader, the running-stage marker, and the hook family, keeping the
#    execution skeleton identical for validation and test
# - The entire batch loop runs inside torch.inference_mode, which disables
#   gradient tracking and version-counter bookkeeping entirely, since no
#   evaluation path may mutate training state
# - The module's epoch-end hook executes before the callback epoch-end hooks,
#   and metrics logged inside that hook are folded into the epoch-metric
#   mapping first, so monitoring callbacks such as early stopping and model
#   checkpointing observe every epoch-level metric, including those produced
#   at epoch end
# - The batch-transfer hook chain applies the same datamodule-over-module
#   precedence rule as the training epoch loop, decided per hook by comparing
#   bound implementations against the DataModule base attributes
# - Step-scoped logger metrics are emitted immediately per batch from rank
#   zero; epoch-scoped metrics accumulate under each metric's declared
#   reduction and reach the logger only from rank zero, while the trainer's
#   callback metrics are updated on every rank
# - The reduced epoch metrics are returned to the caller and also stored on
#   the loop, because the fit loop reads them for monitored-metric resolution
#   after each validation pass
#
# Author: Rahul Sawhney

import time
from collections.abc import Iterator
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from syntheticmind.callbacks.callback import Callback
from syntheticmind.callbacks.checkpoint import Checkpoint
from syntheticmind.callbacks.early_stopping import EarlyStopping
from syntheticmind.core.datamodule import DataModule
from syntheticmind.core.module import LoggedMetric, Module
from syntheticmind.loops.batch_limits import resolve_batch_limit
from syntheticmind.loops.loop import Loop
from syntheticmind.state.trainer_state import TrainerState
from syntheticmind.strategies.strategy import Strategy
from syntheticmind.utilities.distributed import DistributedUtils
from syntheticmind.utilities.evaluation import evaluation_context
from syntheticmind.utilities.metric_accumulator import MetricAccumulator
from syntheticmind.utilities.model_helpers import ModuleMode, is_overridden
from syntheticmind.utilities.types import Batch, RunningStage, StepOutput

if TYPE_CHECKING:
    from syntheticmind.core.trainer import Trainer

__all__: list[str] = ["EvaluationLoop"]


class EvaluationLoop(Loop):
    # Loop that executes one evaluation pass over the stage's dataloader each
    # time run() is invoked. The trainer owns two instances, one per stage:
    # the validate instance additionally serves mid-epoch and epoch-end
    # validation during fit, and its reduced epoch metrics feed the monitored
    # values consumed by checkpointing and early stopping.
    def __init__(
        self,
        stage: Literal["validate", "test"] = "validate",
        inference_mode: bool = True
    ) -> None:
        # Binds the evaluation stage and the optional batch ceiling, and
        # initializes the epoch-metric mapping that run() repopulates on every
        # invocation.
        super().__init__()
        self.stage: Literal["validate", "test"] = stage
        self.inference_mode: bool = inference_mode
        self.epoch_metrics: dict[str, float] = {}

    def run(self) -> dict[str, float]:
        use_inference_mode: bool = (
            self.inference_mode and self.trainer.state.stage != "fit"
        )
        with evaluation_context(use_inference_mode):
            return self._run()

    def _run(self) -> dict[str, float]:
        # Executes one full evaluation pass. The sequence is: switch the module
        # to evaluation mode through its stage hook, fire the stage-start and
        # epoch-start hook pairs, iterate batches under inference mode with the
        # batch-start hooks, the batch-transfer chain, the strategy step, and
        # the batch-end hooks, then reduce epoch accumulators, fire the
        # epoch-end and stage-end hook pairs, and publish the reduced metrics
        # to the trainer's callback metrics.
        self.epoch_metrics: dict[str, float] = {}

        trainer: Trainer = self.trainer
        module: Module | None = trainer.module
        assert module is not None
        strategy: Strategy = trainer.strategy
        state: TrainerState = trainer.state
        callbacks: list[Callback] = trainer.callbacks
        monitoring_callbacks: list[Callback] = [
            callback for callback in callbacks if isinstance(callback, (EarlyStopping, Checkpoint))
        ]
        non_monitoring_callbacks: list[Callback] = [
            callback for callback in callbacks if not isinstance(callback, (EarlyStopping, Checkpoint))
        ]
        device: torch.device = strategy.root_device
        model: nn.Module | None = strategy.model
        assert model is not None

        running_stage: RunningStage = "validating" if self.stage == "validate" else "testing"
        state.set_running_stage(running_stage)
        module._running_stage = running_stage
        if not trainer._sanity_checking:
            trainer._reset_progress_bar_metrics(running_stage)

        dataloader: DataLoader = self._get_dataloader()  # type: ignore[type-arg]
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(state.current_epoch)
        total_batches: int | None = self._safe_len(dataloader)
        max_batches: int | None = self._resolve_limit_batches(total_batches)
        if trainer._sanity_checking and trainer.num_sanity_val_steps != float("inf"):
            sanity_steps: int = int(trainer.num_sanity_val_steps)
            max_batches = (
                sanity_steps
                if max_batches is None
                else min(max_batches, sanity_steps)
            )
        if max_batches == 0:
            module._reset_logged_metrics()
            return self.epoch_metrics

        module_mode: ModuleMode = ModuleMode()
        module_mode.capture(module)

        if self.stage == "validate":
            trainer._call_module_hook(
                module,
                "on_validation_model_eval",
                module.on_validation_model_eval
            )
        else:
            trainer._call_module_hook(
                module,
                "on_test_model_eval",
                module.on_test_model_eval
            )

        if self.stage == "validate":
            trainer._call_callback_hooks(
                module,
                "on_validation_start",
                callback_group=callbacks
            )
            trainer._call_module_hook(
                module,
                "on_validation_start",
                module.on_validation_start
            )
            trainer._call_callback_hooks(
                module,
                "on_validation_epoch_start",
                callback_group=callbacks
            )
            trainer._call_module_hook(
                module,
                "on_validation_epoch_start",
                module.on_validation_epoch_start
            )
        else:
            trainer._call_callback_hooks(
                module,
                "on_test_start",
                callback_group=callbacks
            )
            trainer._call_module_hook(
                module,
                "on_test_start",
                module.on_test_start
            )
            trainer._call_callback_hooks(
                module,
                "on_test_epoch_start",
                callback_group=callbacks
            )
            trainer._call_module_hook(
                module,
                "on_test_epoch_start",
                module.on_test_epoch_start
            )
        epoch_accum: dict[str, MetricAccumulator] = {}
        logger_metric_names: set[str] = set()
        progress_bar_metric_names: set[str] = set()

        dataloader_iterator_start_time: float = time.perf_counter()
        with trainer.profiler.profile(f"setup_{self.stage}_dataloader"):
            dataloader_iterator: Iterator[Batch] = iter(dataloader)
        trainer._record_dataloader_iterator_creation_time(
            running_stage,
            time.perf_counter() - dataloader_iterator_start_time
        )
        batch_idx: int = -1
        while True:
            dataloader_fetch_start_time: float = time.perf_counter()
            try:
                dataloader_prefix: str = "val" if self.stage == "validate" else "test"
                with trainer.profiler.profile(
                    f"[{type(self).__name__}].{dataloader_prefix}_next"
                ):
                    batch: Batch = next(dataloader_iterator)
            except StopIteration:
                break
            trainer._record_dataloader_fetch_time(
                running_stage,
                time.perf_counter() - dataloader_fetch_start_time
            )
            batch_idx += 1
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch: Batch = trainer.precision_plugin.convert_input(batch)

            if self.stage == "validate":
                trainer._call_callback_hooks(
                    module,
                    "on_validation_batch_start",
                    batch,
                    batch_idx,
                    0,
                    callback_group=callbacks
                )
                trainer._call_module_hook(
                    module,
                    "on_validation_batch_start",
                    module.on_validation_batch_start,
                    batch,
                    batch_idx,
                    0
                )
            else:
                trainer._call_callback_hooks(
                    module,
                    "on_test_batch_start",
                    batch,
                    batch_idx,
                    0,
                    callback_group=callbacks
                )
                trainer._call_module_hook(
                    module,
                    "on_test_batch_start",
                    module.on_test_batch_start,
                    batch,
                    batch_idx,
                    0
                )

            dm: DataModule | None = trainer.datamodule
            if dm is not None and is_overridden("on_before_batch_transfer", dm, DataModule):
                batch: Batch = trainer._call_datamodule_hook(
                    dm,
                    "on_before_batch_transfer",
                    dm.on_before_batch_transfer,
                    batch,
                    dataloader_idx=0
                )
            else:
                batch = trainer._call_module_hook(
                    module,
                    "on_before_batch_transfer",
                    module.on_before_batch_transfer,
                    batch,
                    dataloader_idx=0
                )
            if dm is not None and is_overridden("transfer_batch_to_device", dm, DataModule):
                batch = trainer._call_datamodule_hook(
                    dm,
                    "transfer_batch_to_device",
                    dm.transfer_batch_to_device,
                    batch,
                    device,
                    dataloader_idx=0
                )
            else:
                batch = trainer._call_module_hook(
                    module,
                    "transfer_batch_to_device",
                    module.transfer_batch_to_device,
                    batch,
                    device,
                    dataloader_idx=0
                )
            if dm is not None and is_overridden("on_after_batch_transfer", dm, DataModule):
                batch = trainer._call_datamodule_hook(
                    dm,
                    "on_after_batch_transfer",
                    dm.on_after_batch_transfer,
                    batch,
                    dataloader_idx=0
                )
            else:
                batch = trainer._call_module_hook(
                    module,
                    "on_after_batch_transfer",
                    module.on_after_batch_transfer,
                    batch,
                    dataloader_idx=0
                )

            step_context: AbstractContextManager[None] = (
                trainer.precision_plugin.val_step_context()
                if self.stage == "validate"
                else trainer.precision_plugin.test_step_context()
            )
            with step_context:
                if self.stage == "validate":
                    with module._hook_context("validation_step"):
                        outputs: StepOutput = trainer._call_strategy_hook(
                            "validation_step",
                            strategy.validation_step,
                            model,
                            batch,
                            batch_idx
                        )
                else:
                    with module._hook_context("test_step"):
                        outputs = trainer._call_strategy_hook(
                            "test_step",
                            strategy.test_step,
                            model,
                            batch,
                            batch_idx
                        )

            if self.stage == "validate":
                trainer._call_callback_hooks(
                    module,
                    "on_validation_batch_end",
                    outputs,
                    batch,
                    batch_idx,
                    0,
                    callback_group=callbacks
                )
                trainer._call_module_hook(
                    module,
                    "on_validation_batch_end",
                    module.on_validation_batch_end,
                    outputs,
                    batch,
                    batch_idx,
                    0
                )
            else:
                trainer._call_callback_hooks(
                    module,
                    "on_test_batch_end",
                    outputs,
                    batch,
                    batch_idx,
                    0,
                    callback_group=callbacks
                )
                trainer._call_module_hook(
                    module,
                    "on_test_batch_end",
                    module.on_test_batch_end,
                    outputs,
                    batch,
                    batch_idx,
                    0
                )

            self._accumulate_metrics(
                module,
                epoch_accum,
                logger_metric_names,
                progress_bar_metric_names,
                state,
                running_stage
            )

        self.epoch_metrics: dict[str, float] = self._reduce_epoch_metrics(epoch_accum)
        if not trainer._sanity_checking:
            progress_bar_epoch_metrics: dict[str, float] = {
                name: value
                for name, value in self.epoch_metrics.items()
                if name in progress_bar_metric_names
            }
            trainer._update_progress_bar_metrics(running_stage, progress_bar_epoch_metrics)

        if trainer.loggers and self.epoch_metrics and DistributedUtils.is_rank_zero():
            logger_epoch_metrics: dict[str, float] = {
                k: v for k, v in self.epoch_metrics.items() if k in logger_metric_names
            }
            if logger_epoch_metrics:
                trainer._log_metrics(logger_epoch_metrics, step=state.global_step)

        if self.stage == "validate":
            self.trainer._callback_metrics.update(self.epoch_metrics)
            trainer._call_callback_hooks(
                module,
                "on_validation_epoch_end",
                callback_group=non_monitoring_callbacks
            )
            trainer._call_module_hook(
                module,
                "on_validation_epoch_end",
                module.on_validation_epoch_end
            )
            self._consume_epoch_end_hook_metrics(module, state, running_stage)
            self.trainer._callback_metrics.update(self.epoch_metrics)
            trainer._call_callback_hooks(
                module,
                "on_validation_epoch_end",
                callback_group=monitoring_callbacks
            )
            trainer._call_callback_hooks(
                module,
                "on_validation_end",
                callback_group=callbacks
            )
            trainer._call_module_hook(
                module,
                "on_validation_end",
                module.on_validation_end
            )
            if is_overridden("on_validation_model_train", module, Module):
                trainer._call_module_hook(
                    module,
                    "on_validation_model_train",
                    module.on_validation_model_train
                )
            else:
                module_mode.restore(module)
        else:
            self.trainer._callback_metrics.update(self.epoch_metrics)
            trainer._call_callback_hooks(
                module,
                "on_test_epoch_end",
                callback_group=callbacks
            )
            trainer._call_module_hook(
                module,
                "on_test_epoch_end",
                module.on_test_epoch_end
            )
            self._consume_epoch_end_hook_metrics(module, state, running_stage)
            self.trainer._callback_metrics.update(self.epoch_metrics)
            trainer._call_callback_hooks(
                module,
                "on_test_end",
                callback_group=callbacks
            )
            trainer._call_module_hook(
                module,
                "on_test_end",
                module.on_test_end
            )
            if is_overridden("on_test_model_train", module, Module):
                trainer._call_module_hook(
                    module,
                    "on_test_model_train",
                    module.on_test_model_train
                )
            else:
                module_mode.restore(module)

        module._reset_logged_metrics()
        return self.epoch_metrics

    def _get_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        # Selects the dataloader belonging to the configured stage. The
        # assertions document that the trainer entry points guarantee the
        # corresponding dataloader exists before this loop runs.
        trainer: Trainer = self.trainer
        if self.stage == "validate":
            if trainer.state.stage == "fit":
                trainer._ensure_val_dataloader()
            assert trainer.val_dataloader is not None
            return trainer.val_dataloader
        assert trainer.test_dataloader is not None
        return trainer.test_dataloader

    @staticmethod
    def _safe_len(dataloader: DataLoader) -> int | None:  # type: ignore[type-arg]
        # Returns the dataloader length, or None for iterable datasets that do
        # not define one, so limit resolution can distinguish the two cases.
        try:
            return len(dataloader)
        except TypeError:
            return None

    def _resolve_limit_batches(self, total: int | None) -> int | None:
        # Resolves the effective batch ceiling for this pass. With a known
        # total, a float limit selects that fraction of the pass (at least one
        # batch) and an integer limit is capped at the total. Without a known
        # total, an integer limit is used directly, a fractional float limit is
        # rejected as a configuration error, and no limit means unbounded.
        trainer: Trainer = self.trainer
        if self.stage == "validate":
            return resolve_batch_limit(total, trainer.limit_val_batches, "val")
        return resolve_batch_limit(total, trainer.limit_test_batches, "test")

    def _accumulate_metrics(
        self,
        module: Module,
        epoch_accum: dict[str, MetricAccumulator],
        logger_metric_names: set[str],
        progress_bar_metric_names: set[str],
        state: TrainerState,
        running_stage: RunningStage
    ) -> None:
        # Drains the module's logged-metric buffer after each batch. Metrics
        # marked for distributed synchronization are mean-reduced across ranks
        # first. Step-scoped logger metrics are emitted immediately from rank
        # zero; epoch-scoped metrics feed per-name accumulators under their
        # declared reduction and batch-size weighting, with logger visibility
        # tracked separately from callback visibility.
        logged: dict[str, LoggedMetric] = module.logged_metrics
        if not logged:
            return

        step_metrics: dict[str, float] = {}
        progress_bar_step_metrics: dict[str, float] = {}
        for name, metric in logged.items():
            value: torch.Tensor | float = metric.value
            if metric.sync_distributed and DistributedUtils.is_distributed():
                if not isinstance(value, torch.Tensor):
                    value: torch.Tensor | float = torch.tensor(value, device=self.trainer.strategy.root_device)
                value: torch.Tensor | float = DistributedUtils.all_reduce_mean(value)
            if isinstance(value, torch.Tensor):
                value: torch.Tensor | float = value.item()
            if metric.on_step and metric.logger:
                step_metrics[name] = value
            if metric.on_step and metric.progress_bar:
                progress_bar_step_metrics[name] = value
            if metric.on_epoch:
                if name not in epoch_accum:
                    epoch_accum[name] = MetricAccumulator(reduction=metric.reduction)
                epoch_accum[name].update(value, batch_size=metric.batch_size)
                if metric.logger:
                    logger_metric_names.add(name)
                if metric.progress_bar:
                    progress_bar_metric_names.add(name)

        if step_metrics and self.trainer.loggers and DistributedUtils.is_rank_zero():
            self.trainer._log_metrics(step_metrics, step=state.global_step)
        if not self.trainer._sanity_checking:
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
        # Drains metrics logged inside the module's epoch-end hook, which runs
        # after the batch accumulators have already been reduced. Epoch-scoped
        # values are folded directly into the epoch-metric mapping so the
        # callback epoch-end hooks observe them, and logger-visible values are
        # emitted from rank zero.
        logged: dict[str, LoggedMetric] = module.logged_metrics
        if not logged:
            return
        hook_logger_metrics: dict[str, float] = {}
        hook_progress_bar_metrics: dict[str, float] = {}
        for name, metric in logged.items():
            value: torch.Tensor | float = metric.value
            if metric.sync_distributed and DistributedUtils.is_distributed():
                if not isinstance(value, torch.Tensor):
                    value: torch.Tensor | float = torch.tensor(value, device=self.trainer.strategy.root_device)
                value: torch.Tensor | float = DistributedUtils.all_reduce_mean(value)
            if isinstance(value, torch.Tensor):
                value: torch.Tensor | float = value.item()
            if metric.on_epoch:
                self.epoch_metrics[name] = value
            if metric.logger:
                hook_logger_metrics[name] = value
            if metric.progress_bar:
                hook_progress_bar_metrics[name] = value
        if hook_logger_metrics and self.trainer.loggers and DistributedUtils.is_rank_zero():
            self.trainer._log_metrics(hook_logger_metrics, step=state.global_step)
        if not self.trainer._sanity_checking:
            self.trainer._update_progress_bar_metrics(
                running_stage,
                hook_progress_bar_metrics
            )
        module._reset_logged_metrics()

    @staticmethod
    def _reduce_epoch_metrics(
        epoch_accum: dict[str, MetricAccumulator]
    ) -> dict[str, float]:
        # Reduces every per-name accumulator to its final epoch value under the
        # reduction mode each metric declared when it was logged.
        reduced: dict[str, float] = {}
        for name in epoch_accum:
            reduced[name] = epoch_accum[name].compute()
        return reduced
