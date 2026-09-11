# This module:
# 1. Defines ModelCheckpoint, the public callback that translates fit lifecycle
#    events into periodic, monitored, last, and exception checkpoint requests
# 2. Exposes the retained-artifact ranking and naming surface clients use for
#    evaluation, export selection, and experiment bookkeeping
#
# Design decisions:
# - ModelCheckpoint composes strict configuration, schedule, ranking, path, and
#   transaction concepts instead of merging unrelated policy state into one
#   callback-shaped landfill
# - Batch hooks record save intent; actual persistence occurs only after the
#   training loop declares optimizer, hook, metric, scheduler, and accumulation
#   state durable
# - A monitored trigger that fires before its metric exists defers until
#   validation rather than writing an incorrectly unranked artifact
# - Candidate metrics are collected from the trainer's reduced callback view and
#   live on-step records, then broadcast from global zero so all ranks name and
#   rank the same artifact
# - Full checkpoint requests synchronize every participating rank. Exception
#   saves run only when the format is non-collective and the current trajectory
#   is durable; otherwise the original failure remains untouched
# - Callback state encodes the original path identity separately from a lazily
#   resolved run directory, allowing reliable duplicate-state keys before setup
#
# Author: Rahul Sawhney

import json
import time
from datetime import timedelta
from pathlib import Path
from typing import Literal, override

import syntheticmind
import torch
from loguru import logger as log

from syntheticmind.callbacks.checkpointing.callback import Checkpoint
from syntheticmind.callbacks.checkpointing.configuration import _ModelCheckpointConfiguration
from syntheticmind.callbacks.checkpointing.paths import _CheckpointPaths
from syntheticmind.callbacks.checkpointing.ranking import _CheckpointRanking
from syntheticmind.callbacks.checkpointing.schedule import _CheckpointSchedule
from syntheticmind.callbacks.checkpointing.transaction import _CheckpointTransaction
from syntheticmind.core.module import Module
from syntheticmind.storage.filesystem import StoragePath
from syntheticmind.primitives.types import Batch, StateDict, StepOutput, TrainerStage

__all__: list[str] = ["ModelCheckpoint"]


class ModelCheckpoint(Checkpoint):
    # Public checkpoint retention policy. Clients configure this callback on
    # Trainer and use best_model_path or last_model_path as the authoritative
    # artifact selection after fitting.
    def __init__(
        self,
        dirpath: str | Path | None = None,
        filename: str | None = None,
        monitor: str | None = None,
        verbose: bool = False,
        save_last: bool | Literal["link"] | None = None,
        save_top_k: int = 1,
        save_on_exception: bool = False,
        save_weights_only: bool = False,
        mode: Literal["min", "max"] = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: int | None = None,
        train_time_interval: timedelta | None = None,
        every_n_epochs: int | None = None,
        save_on_train_epoch_end: bool | None = None,
        enable_version_counter: bool = True
    ) -> None:
        # Constructs and validates the complete policy before exposing callback
        # state.
        #
        # Args:
        #     dirpath: Local or supported remote checkpoint directory. None
        #         resolves below the active run directory. Default: ``None``.
        #     filename: Format template populated from metric candidates plus
        #         epoch and step. Default: ``None`` for ``"{epoch}-{step}"``.
        #     monitor: Scalar callback metric used for top-k ranking, or None for
        #         unmonitored retention. Default: ``None``.
        #     verbose: Whether global zero logs save and skip decisions.
        #         Default: ``False``.
        #     save_last: Whether to maintain a latest artifact; ``"link"`` uses
        #         a local symbolic link or copy. Default: ``None``.
        #     save_top_k: Number of ranked artifacts, ``-1`` for all, or ``0``
        #         to disable ranked saves. Default: ``1``.
        #     save_on_exception: Whether eligible fit failures attempt a
        #         diagnostic checkpoint. Default: ``False``.
        #     save_weights_only: Whether artifacts omit optimizer, scheduler,
        #         scaler, callback, data, and RNG continuation state.
        #         Default: ``False``.
        #     mode: ``"min"`` or ``"max"`` ranking direction. Default:
        #         ``"min"``.
        #     auto_insert_metric_name: Whether filename fields include their
        #         metric names. Default: ``True``.
        #     every_n_train_steps: Optional positive optimizer-step cadence.
        #         Zero disables it. Default: ``None``.
        #     train_time_interval: Optional positive wall-time cadence.
        #         Default: ``None``.
        #     every_n_epochs: Optional positive epoch cadence. With no periodic
        #         trigger, it resolves to one. Default: ``None``.
        #     save_on_train_epoch_end: Explicit epoch-save placement, or None to
        #         derive placement from validation cadence. Default: ``None``.
        #     enable_version_counter: Whether existing names receive numeric
        #         version suffixes instead of replacement. Default: ``True``.
        configuration: _ModelCheckpointConfiguration = _ModelCheckpointConfiguration(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_on_exception=save_on_exception,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
            enable_version_counter=enable_version_counter
        )
        self._paths: _CheckpointPaths = _CheckpointPaths(
            configuration.dirpath,
            configuration.filename,
            configuration.auto_insert_metric_name,
            configuration.enable_version_counter
        )
        self._ranking: _CheckpointRanking = _CheckpointRanking(
            configuration.monitor,
            configuration.mode,
            configuration.save_top_k
        )
        self._schedule: _CheckpointSchedule = _CheckpointSchedule(
            configuration.resolved_every_n_train_steps,
            configuration.train_time_interval,
            configuration.resolved_every_n_epochs,
            configuration.save_on_train_epoch_end
        )
        self._transaction: _CheckpointTransaction = _CheckpointTransaction(
            self._paths,
            self._ranking,
            self._schedule,
            configuration.verbose,
            configuration.save_last,
            configuration.save_weights_only
        )
        self._pending_batch_trigger_time: float | None = None
        self._pending_epoch_trigger_time: float | None = None
        self._identity_dirpath: str | None = self._paths.dirpath
        self._identity_filename: str | None = configuration.filename
        self.verbose: bool = configuration.verbose
        self.save_last: bool | Literal["link"] | None = configuration.save_last
        self.save_on_exception: bool = configuration.save_on_exception
        self.save_weights_only: bool = configuration.save_weights_only

    @property
    def dirpath(self) -> str | None:
        # Returns the configured or setup-resolved checkpoint directory.
        return self._paths.dirpath

    @dirpath.setter
    def dirpath(self, value: str | Path | None) -> None:
        # Rebinds the destination through the same normalization policy used at
        # construction; clients changing it invalidate restored ranking paths.
        self._paths.dirpath: str | None = self._paths._normalize_directory(value)

    @property
    def filename(self) -> str | None:
        # Returns the configured filename template.
        return self._paths.filename

    @property
    def monitor(self) -> str | None:
        # Returns the scalar metric key used for ranking.
        return self._ranking.monitor

    @property
    def mode(self) -> Literal["min", "max"]:
        # Returns the ranking direction.
        return self._ranking.mode

    @property
    def save_top_k(self) -> int:
        # Returns the retained ranked-artifact count policy.
        return self._ranking.save_top_k

    @property
    def auto_insert_metric_name(self) -> bool:
        # Reports whether format fields receive explicit metric labels.
        return self._paths.auto_insert_metric_name

    @property
    def current_score(self) -> torch.Tensor | None:
        # Returns the score associated with the most recently accepted artifact.
        return self._ranking.current_score

    @property
    def best_k_models(self) -> dict[str, torch.Tensor]:
        # Returns the live retained path-to-score mapping.
        return self._ranking.best_k_models

    @property
    def kth_best_model_path(self) -> str:
        # Returns the current retention boundary path.
        return self._ranking.kth_best_model_path

    @property
    def kth_value(self) -> torch.Tensor:
        # Returns the scalar score at the current retention boundary.
        return self._ranking.kth_value

    @property
    def best_model_score(self) -> torch.Tensor | None:
        # Returns the best retained scalar score.
        return self._ranking.best_model_score

    @property
    def best_model_path(self) -> str:
        # Returns the best retained checkpoint path, or an empty string before
        # any artifact qualifies.
        return self._ranking.best_model_path

    @property
    def last_model_path(self) -> str:
        # Returns the mutable latest-artifact path maintained by save_last.
        return self._transaction.last_model_path

    @property
    def every_n_epochs(self) -> int | None:
        # Returns the resolved epoch cadence retained for client inspection.
        return self._schedule.every_n_epochs

    def _resolve_dirpath(self, trainer: syntheticmind.Trainer) -> str:
        # Internal connector surface for forcing run-aware directory resolution.
        return self._paths.resolve_directory(trainer)

    @override
    def setup(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        stage: TrainerStage
    ) -> None:
        # Resolves the destination, validates link support, and creates the fit
        # directory on global zero before any trigger can fire.
        del module
        directory: str = self._paths.resolve_directory(trainer)
        if self.save_last == "link" and not StoragePath(directory).is_local:
            raise ValueError("`save_last='link'` requires a local checkpoint directory.")
        if stage == "fit" and not trainer.fast_dev_run and trainer.is_global_zero:
            StoragePath(directory).make_directory()

    @override
    def on_train_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        # Clears transient trigger intent and reconstructs the schedule clock at
        # the start of each fit transaction.
        del trainer, module
        self._pending_batch_trigger_time: float | None = None
        self._pending_epoch_trigger_time: float | None = None
        self._schedule.start()

    @override
    def on_train_batch_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int
    ) -> None:
        # Records due step or time intent after a completed batch; the loop later
        # calls _save_after_batch at its declared durability boundary.
        del outputs, batch, batch_idx
        if self._schedule.should_skip(trainer):
            return
        trigger: tuple[bool, float] = self._schedule.batch_trigger(trainer)
        due: bool = trigger[0]
        trigger_time: float = trigger[1]
        if not due:
            return
        self._pending_batch_trigger_time: float | None = trigger_time

    def _save_after_batch(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        defer_to_epoch: bool
    ) -> None:
        # Consumes pending batch intent only when checkpoint state is durable,
        # optionally carrying terminal-batch intent into the epoch boundary.
        trigger_time: float | None = self._pending_batch_trigger_time
        if trigger_time is None or not trainer._checkpoint_is_durable:
            return
        self._pending_batch_trigger_time: float | None = None
        if defer_to_epoch:
            pending_epoch_time: float | None = self._pending_epoch_trigger_time
            self._pending_epoch_trigger_time: float | None = (
                trigger_time
                if pending_epoch_time is None
                else max(trigger_time, pending_epoch_time)
            )
            return
        if self._schedule.should_skip(trainer):
            return
        self._save_trigger(trainer, module, trigger_time)

    def _save_trigger(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        trigger_time: float
    ) -> None:
        # Synchronizes ranking inputs, resolves monitored deferral, and executes
        # one all-rank transaction with deferral rollback on failure.
        candidates: dict[str, torch.Tensor] = self._synchronize_monitor_candidates(
            trainer,
            module
        )
        if self.monitor is not None and self.monitor not in candidates:
            self._schedule.defer()
            return
        was_deferred: bool = self._schedule.deferred_until_validation
        self._schedule.clear_deferred()
        try:
            self._transaction.save_trigger(trainer, candidates, trigger_time, True)
        except BaseException:
            if was_deferred:
                self._schedule.defer()
            raise

    @override
    def on_validation_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        # Places a due validation-aware trigger at the current epoch or mid-epoch
        # batch durability boundary without persisting inside the hook itself.
        del module
        if self._schedule.should_skip(trainer):
            return
        if not self._schedule.validation_trigger(trainer):
            return
        trigger_time: float = time.monotonic()
        if trainer._is_mid_epoch_validation:
            self._pending_batch_trigger_time: float | None = trigger_time
        else:
            self._pending_epoch_trigger_time: float | None = trigger_time

    def _save_after_epoch(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        # Consumes deferred epoch intent after epoch metrics and schedulers are
        # stable.
        trigger_time: float | None = self._pending_epoch_trigger_time
        if trigger_time is None:
            return
        self._pending_epoch_trigger_time: float | None = None
        if self._schedule.should_skip(trainer):
            return
        self._save_trigger(
            trainer,
            module,
            trigger_time
        )

    @override
    def on_train_epoch_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        # Resolves carried intent first, then evaluates direct train-epoch-end
        # cadence for policies that do not require post-validation placement.
        self._save_after_epoch(trainer, module)
        if self._schedule.should_skip(trainer):
            return
        if not self._schedule.epoch_trigger(trainer):
            return
        candidates: dict[str, torch.Tensor] = self._synchronize_monitor_candidates(
            trainer,
            module
        )
        self._transaction.save_trigger(
            trainer,
            candidates,
            time.monotonic(),
            True
        )

    @override
    def on_train_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        # Flushes pending epoch intent and guarantees save_last reflects the final
        # global step even when no periodic artifact was due there.
        self._save_after_epoch(trainer, module)
        if (
            not self.save_last
            or self._schedule.last_global_step_saved == trainer.global_step
        ):
            return
        candidates: dict[str, torch.Tensor] = self._synchronize_monitor_candidates(
            trainer,
            module
        )
        self._transaction.save_last_if_stale(
            trainer,
            candidates,
            time.monotonic()
        )

    def check_monitor_top_k(
        self,
        trainer: syntheticmind.Trainer,
        current: torch.Tensor | None = None
    ) -> bool:
        # Public, non-mutating top-k qualification query for custom client
        # checkpoint workflows using this callback's current ranking.
        return self._ranking.qualifies(trainer, current)

    def _monitor_candidates(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> dict[str, torch.Tensor]:
        # Builds filename and ranking candidates from reduced callback metrics,
        # live on-step records, and canonical epoch and step coordinates.
        candidates: dict[str, torch.Tensor] = {
            name: value.detach().cpu()
            for name, value in trainer.callback_metrics.items()
        }
        for metric in module.logged_metrics:
            if not metric.on_step:
                continue
            value: torch.Tensor | None = metric.compute_step(
                module.device,
                trainer.strategy.reduce
            )
            if value is None:
                continue
            candidates[metric.base_name()] = value.detach().cpu()
            candidates[metric.output_name(on_step=True)] = value.detach().cpu()
        candidates["epoch"] = torch.tensor(trainer.current_epoch)
        candidates["step"] = torch.tensor(trainer.global_step)
        return candidates

    def _synchronize_monitor_candidates(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> dict[str, torch.Tensor]:
        # Broadcasts the global-zero candidate mapping so every rank participates
        # in one identical path and ranking decision.
        local_candidates: dict[str, torch.Tensor] = self._monitor_candidates(
            trainer,
            module
        )
        return trainer.strategy.broadcast(
            local_candidates if trainer.is_global_zero else {}
        )

    def format_checkpoint_name(
        self,
        metrics: dict[str, torch.Tensor],
        filename: str | None = None,
        ver: int | None = None,
        prefix: str | None = None
    ) -> str:
        # Public naming helper for clients that need to preview or reproduce the
        # callback's artifact path policy.
        return self._paths.format_checkpoint_name(
            metrics,
            filename,
            ver,
            prefix
        )

    def file_exists(
        self,
        filepath: str | Path,
        trainer: syntheticmind.Trainer
    ) -> bool:
        # Performs one rank-zero existence query and broadcasts its result.
        path_exists: bool = (
            StoragePath(filepath).exists() if trainer.is_global_zero else False
        )
        return trainer.strategy.broadcast(path_exists)

    def to_yaml(self, filepath: str | Path | None = None) -> None:
        # Writes the retained ranking as deterministic JSON-compatible YAML data;
        # an explicit destination may be local or supported remote storage.
        destination: str | Path
        if filepath is None:
            if self.dirpath is None:
                raise RuntimeError("ModelCheckpoint directory is unresolved before setup.")
            destination: str | Path = StoragePath(self.dirpath).join(
                "best_k_models.yaml"
            )
        else:
            destination: str | Path = filepath
        values: dict[str, float] = {
            path: float(score) for path, score in self.best_k_models.items()
        }
        StoragePath(destination).write_text(
            json.dumps(values, indent=2, sort_keys=True)
        )

    @override
    def on_exception(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        exception: BaseException
    ) -> None:
        # Attempts a best-effort diagnostic save only when one rank can represent
        # a durable, non-collective checkpoint without deadlocking peers.
        del module
        self._pending_batch_trigger_time: float | None = None
        self._pending_epoch_trigger_time: float | None = None
        if trainer.fast_dev_run or not self.save_on_exception:
            return
        if trainer.state.stage != "fit" or not trainer.is_global_zero:
            return
        if not self.save_weights_only and not trainer._checkpoint_is_durable:
            log.warning(
                "Skipping exception checkpoint because automatic optimization "
                "has not reached a durable optimizer-step boundary."
            )
            return
        if trainer.strategy.checkpoint_requires_all_ranks:
            log.warning(
                "Skipping exception checkpoint because the selected checkpoint "
                "format requires collective participation."
            )
            return
        candidates: dict[str, torch.Tensor] = {
            name: value.detach().cpu()
            for name, value in trainer.callback_metrics.items()
        }
        candidates["epoch"] = torch.tensor(trainer.current_epoch)
        candidates["step"] = torch.tensor(trainer.global_step)
        self._transaction.save_exception(trainer, candidates, exception)

    @property
    @override
    def state_key(self) -> str:
        # Qualifies callback state by artifact identity, ranking, retention, and
        # periodic trigger semantics.
        return self._generate_state_key(
            dirpath=self._identity_dirpath,
            filename=self._identity_filename,
            monitor=self.monitor,
            mode=self.mode,
            save_top_k=self.save_top_k,
            save_last=self.save_last,
            save_weights_only=self.save_weights_only,
            every_n_epochs=self._schedule.every_n_epochs,
            every_n_train_steps=self._schedule.every_n_train_steps,
            train_time_interval=(
                str(self._schedule.train_time_interval)
                if self._schedule.train_time_interval is not None
                else None
            )
        )

    @override
    def state_dict(self) -> StateDict:
        # Delegates the full resumable policy trajectory to its transaction.
        return self._transaction.state_dict()

    @override
    def load_state_dict(self, state_dict: StateDict) -> None:
        # Restores schedule, ranking, and artifact coordinates as one policy.
        self._transaction.load_state_dict(state_dict)

    @override
    def __repr__(self) -> str:
        # Presents the principal selection, cadence, and destination policy.
        return (
            f"ModelCheckpoint(monitor={self.monitor!r}, mode={self.mode!r}, "
            f"save_top_k={self.save_top_k}, save_last={self.save_last!r}, "
            f"every_n_train_steps={self._schedule.every_n_train_steps}, "
            f"every_n_epochs={self._schedule.every_n_epochs}, "
            f"dirpath={self.dirpath!r})"
        )
