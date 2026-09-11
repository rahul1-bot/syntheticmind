# This module:
# 1. Defines DeviceStatsMonitor, the callback that samples accelerator-native
#    utilization statistics during training, validation, testing, and
#    prediction
# 2. Routes sampled values into both callback_metrics and configured experiment
#    loggers under stage-qualified device metric names
#
# Design decisions:
# - Statistics come exclusively from Trainer.accelerator, so CPU, CUDA, and MPS
#   expose their own supported vocabulary without backend checks leaking into
#   client callbacks
# - filter_keys is validated against the selected accelerator at stage entry,
#   making a misspelled or backend-incompatible metric fail before a long run
# - Training cadence follows global_step and suppresses duplicate observations
#   across accumulation batches that share one optimizer step
# - Evaluation cadence follows the local batch index because evaluation does
#   not advance global_step; sanity validation is excluded from user telemetry
# - The complete stage-qualified mapping is written atomically, and empty
#   accelerator reports remain a valid no-op for backends without statistics
#
# Author: Rahul Sawhney

from typing import Literal, override

import syntheticmind
import torch

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.primitives.exceptions import MisconfigurationError
from syntheticmind.primitives.types import Batch, ModelOutput, StepOutput, TrainerStage

__all__: list[str] = ["DeviceStatsMonitor"]

type _DeviceStage = Literal["train", "validation", "test", "prediction"]


class DeviceStatsMonitor(Callback):
    # Accelerator-statistics observer for clients that already configured one
    # or more Trainer loggers. It never owns device discovery or sampling APIs;
    # those remain part of the active Accelerator contract.
    def __init__(
        self,
        log_every_n_steps: int = 50,
        filter_keys: set[str] | None = None
    ) -> None:
        # Configures sampling cadence and an optional exact backend-key filter.
        #
        # Args:
        #     log_every_n_steps: Positive cadence for emitted observations.
        #         Training counts optimizer steps; evaluation counts batches.
        #         Default: ``50``.
        #     filter_keys: Exact accelerator statistic keys to retain, or None
        #         to publish every available statistic. Default: ``None``.
        if type(log_every_n_steps) is not int or log_every_n_steps < 1:
            raise ValueError("`log_every_n_steps` must be a positive integer.")
        if filter_keys is not None and (
            not isinstance(filter_keys, set)
            or not all(isinstance(key, str) for key in filter_keys)
        ):
            raise TypeError("`filter_keys` must be a set of strings or None.")
        self.log_every_n_steps: int = log_every_n_steps
        self.filter_keys: frozenset[str] | None = (
            None if filter_keys is None else frozenset(filter_keys)
        )
        self._last_train_step: int = -1

    @override
    def setup(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        stage: TrainerStage
    ) -> None:
        # Rejects use without a logger before stage execution begins. Device
        # support itself is validated after the strategy has selected a root
        # device and the relevant stage starts.
        del module, stage
        if not trainer.loggers:
            raise MisconfigurationError(
                "DeviceStatsMonitor requires at least one configured logger."
            )

    @override
    def on_train_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        # Validates requested statistic keys against the training device.
        del module
        self._validate_filters(trainer)

    @override
    def on_validation_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        # Validates keys for real validation while leaving sanity checks free
        # from user-facing telemetry side effects.
        del module
        if not trainer.sanity_checking:
            self._validate_filters(trainer)

    @override
    def on_test_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        # Validates requested keys before standalone testing.
        del module
        self._validate_filters(trainer)

    @override
    def on_predict_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        # Validates requested keys before prediction.
        del module
        self._validate_filters(trainer)

    @override
    def on_train_batch_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int
    ) -> None:
        # Emits once per configured optimizer-step cadence. Accumulation may
        # dispatch several batch-end hooks with one global_step, so the last
        # emitted step guard prevents duplicates.
        del module, outputs, batch, batch_idx
        if trainer.global_step <= 0:
            return
        if trainer.global_step == self._last_train_step:
            return
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        self._emit(trainer, "train", trainer.global_step)
        self._last_train_step: int = trainer.global_step

    @override
    def on_validation_batch_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Samples validation by batch cadence outside sanity checking.
        del module, outputs, batch, dataloader_idx
        if trainer.sanity_checking:
            return
        self._emit_evaluation_batch(trainer, "validation", batch_idx)

    @override
    def on_test_batch_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Samples standalone test execution by batch cadence.
        del module, outputs, batch, dataloader_idx
        self._emit_evaluation_batch(trainer, "test", batch_idx)

    @override
    def on_predict_batch_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        outputs: ModelOutput,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        # Samples prediction execution by batch cadence.
        del module, outputs, batch, dataloader_idx
        self._emit_evaluation_batch(trainer, "prediction", batch_idx)

    def _emit_evaluation_batch(
        self,
        trainer: syntheticmind.Trainer,
        stage: Literal["validation", "test", "prediction"],
        batch_idx: int
    ) -> None:
        # Converts zero-based evaluation batch indices into a one-based cadence
        # without advancing or depending on the training step counter.
        if (batch_idx + 1) % self.log_every_n_steps != 0:
            return
        self._emit(trainer, stage, trainer.global_step)

    def _validate_filters(self, trainer: syntheticmind.Trainer) -> None:
        # Resolves filters against a live backend report so client configuration
        # cannot silently discard every requested statistic.
        if self.filter_keys is None:
            return
        statistics: dict[str, float] = trainer.accelerator.get_device_stats(
            trainer.strategy.root_device
        )
        missing_keys: set[str] = set(self.filter_keys) - set(statistics)
        if missing_keys:
            raise MisconfigurationError(
                f"DeviceStatsMonitor filter keys are unavailable: {sorted(missing_keys)}. "
                f"Available keys: {sorted(statistics)}."
            )

    def _emit(
        self,
        trainer: syntheticmind.Trainer,
        stage: _DeviceStage,
        step: int
    ) -> None:
        # Publishes a stage-qualified snapshot to callback consumers on every
        # rank and to experiment loggers through the trainer's rank policy.
        statistics: dict[str, float] = trainer.accelerator.get_device_stats(
            trainer.strategy.root_device
        )
        if self.filter_keys is not None:
            statistics: dict[str, float] = {
                name: value
                for name, value in statistics.items()
                if name in self.filter_keys
            }
        metrics: dict[str, float] = {
            f"device/{stage}/{name}": value
            for name, value in statistics.items()
        }
        if not metrics:
            return
        trainer._logger_connector.callback_metrics.update(
            {
                name: torch.tensor(value, device=trainer.strategy.root_device)
                for name, value in metrics.items()
            }
        )
        trainer._log_metrics(metrics, step=step)

    @override
    def __repr__(self) -> str:
        # Exposes the complete monitoring policy in trainer diagnostics.
        return (
            f"DeviceStatsMonitor(log_every_n_steps={self.log_every_n_steps}, "
            f"filter_keys={self.filter_keys!r})"
        )
