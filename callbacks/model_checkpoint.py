# This module:
# 1. Saves training checkpoints on the configured cadence: a rolling last.ckpt,
#    monitored top-k checkpoints named from a filename template, and an
#    optional exception checkpoint when a run fails
# 2. Builds full-resume payloads through CheckpointState, including mid-epoch
#    resume markers, and records save duration and file size as runtime
#    metrics
# 3. Persists its own selection state (best score, best path, retained top-k
#    set) so checkpoint retention continues correctly across resumption
#
# Design decisions:
# - All filesystem activity is restricted to rank zero, so distributed runs
#   produce exactly one checkpoint directory without write races
# - The saved epoch number is resume-oriented: a mid-epoch save records the
#   current epoch together with the completed-batch count and the epoch-start
#   random state, so resumption replays the same epoch from the interruption
#   point, while an epoch-boundary save records the next epoch to begin
# - Checkpoints are written to a same-directory temporary file and promoted by
#   atomic rename, so a crash during writing can never leave a truncated file
#   under a checkpoint name; the temporary file is removed in a finally block
# - The filename template resolves against epoch, step, and the metric values;
#   a collision with a retained checkpoint appends a version suffix instead of
#   overwriting the earlier file
# - Top-k retention sorts by score in the monitored direction and evicts beyond
#   k, but never deletes the file recorded as the best path
# - save_top_k accepts -1 to retain every eligible checkpoint and 0 to disable
#   monitored saves while save_last may continue independently
# - The exception path saves a full payload when the optimizer already exists
#   and falls back to a weights-only partial payload marked in its metadata,
#   so an early-construction failure still preserves the model weights
# - Restored top-k paths that resolve outside the current checkpoint directory
#   clear the retention state instead of adopting it, so a changed dirpath can
#   never cause deletions in a previous run's directory
#
# Author: Rahul Sawhney

from __future__ import annotations

import os
import re
import shutil
import time
import warnings
from contextlib import AbstractContextManager, nullcontext
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, override

import torch
from loguru import logger as log

from syntheticmind.callbacks.checkpoint import Checkpoint
from syntheticmind.core.module import Module
from syntheticmind.utilities.distributed import is_rank_zero
from syntheticmind.utilities.filesystem import (
    exists,
    file_size,
    is_file,
    is_local_path,
    is_path_within,
    join_path,
    make_directory,
    normalize_path,
    same_path,
    write_text,
)
from syntheticmind.utilities.types import (
    Batch,
    CheckpointDict,
    StateDict,
    StateValue,
    StepOutput,
    TrainerStage,
)

if TYPE_CHECKING:
    from syntheticmind.core.trainer import Trainer

__all__: list[str] = ["ModelCheckpoint"]


class ModelCheckpoint(Checkpoint):
    CHECKPOINT_JOIN_CHAR: str = "-"
    CHECKPOINT_EQUALS_CHAR: str = "="
    CHECKPOINT_NAME_LAST: str = "last"
    FILE_EXTENSION: str = ".ckpt"
    STARTING_VERSION: int = 1

    # Checkpointing callback driven by validation passes during fit. On each
    # eligible validation end it refreshes last.ckpt, evaluates the monitored
    # metric against the best score, saves and retains monitored checkpoints
    # under the top-k policy, and reports save telemetry through the
    # experiment logger.
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
        # Validates the cadence options and binds the checkpoint directory,
        # monitored metric and direction, retention count, filename template,
        # and exception-save policy. The best score starts at the identity
        # element for the chosen direction so the first monitored value always
        # qualifies.
        #
        # Args:
        #     dirpath: Directory receiving every checkpoint file written by
        #         this callback. Default: ``"checkpoints"``.
        #     monitor: Metric key compared for ranked retention, read from
        #         the trainer's callback metrics at each eligible
        #         validation boundary. Default: ``"val_loss"``.
        #     mode: Improvement direction for the monitored metric,
        #         ``"min"`` or ``"max"``. Default: ``"min"``.
        #     save_top_k: Number of best-ranked checkpoints retained;
        #         displaced entries are deleted from disk. Default: ``1``.
        #     filename: Filename template formatted with the current epoch
        #         and global step. Default:
        #         ``"checkpoint-epoch{epoch:02d}-step{step:05d}"``.
        #     save_last: Whether a last.ckpt copy is refreshed at every
        #         eligible boundary independently of the monitored ranking.
        #         Default: ``True``.
        #     every_n_epochs: Epoch cadence gating the validation-boundary
        #         saves; zero disables monitored epoch-cadence saving.
        #         Default: ``1``.
        #     every_n_train_steps: Optional optimizer-step cadence for
        #         mid-epoch saves; ``None`` disables step-cadence saving.
        #         Must be at least one when provided. Default: ``None``.
        #     save_on_exception: Whether a rescue checkpoint is written
        #         when training aborts on an exception. Default: ``False``.
        #
        # Raises:
        #     MisconfigurationError: If every_n_epochs is negative or
        #         every_n_train_steps is provided below one.
        super().__init__()
        from syntheticmind.utilities.exceptions import MisconfigurationError

        if dirpath is not None and not isinstance(dirpath, str | Path):
            raise TypeError("`dirpath` must be a string, Path, or None.")
        if filename is not None and not isinstance(filename, str):
            raise TypeError("`filename` must be a string or None.")
        if monitor is not None and not isinstance(monitor, str):
            raise TypeError("`monitor` must be a string or None.")
        if type(verbose) is not bool:
            raise TypeError("`verbose` must be a bool.")
        if not (
            save_last is None
            or type(save_last) is bool
            or save_last == "link"
        ):
            raise MisconfigurationError("`save_last` must be a bool, 'link', or None.")
        if type(save_top_k) is not int or save_top_k < -1:
            raise MisconfigurationError("`save_top_k` must be an integer greater than or equal to -1.")
        if type(save_on_exception) is not bool:
            raise TypeError("`save_on_exception` must be a bool.")
        if type(save_weights_only) is not bool:
            raise TypeError("`save_weights_only` must be a bool.")
        if mode not in ("min", "max"):
            raise MisconfigurationError("`mode` must be 'min' or 'max'.")
        if type(auto_insert_metric_name) is not bool:
            raise TypeError("`auto_insert_metric_name` must be a bool.")
        if every_n_train_steps is not None and (
            type(every_n_train_steps) is not int or every_n_train_steps < 0
        ):
            raise MisconfigurationError("`every_n_train_steps` must be a non-negative integer or None.")
        if train_time_interval is not None and not isinstance(train_time_interval, timedelta):
            raise TypeError("`train_time_interval` must be a timedelta or None.")
        if train_time_interval is not None and train_time_interval.total_seconds() < 0:
            raise MisconfigurationError("`train_time_interval` must be non-negative.")
        if every_n_epochs is not None and (
            type(every_n_epochs) is not int or every_n_epochs < 0
        ):
            raise MisconfigurationError("`every_n_epochs` must be a non-negative integer or None.")
        if save_on_train_epoch_end is not None and type(save_on_train_epoch_end) is not bool:
            raise TypeError("`save_on_train_epoch_end` must be a bool or None.")
        if type(enable_version_counter) is not bool:
            raise TypeError("`enable_version_counter` must be a bool.")
        trigger_count: int = sum(
            (
                every_n_train_steps is not None and every_n_train_steps > 0,
                every_n_epochs is not None and every_n_epochs > 0,
                train_time_interval is not None
            )
        )
        if trigger_count > 1:
            raise MisconfigurationError(
                "`every_n_train_steps`, `every_n_epochs`, and `train_time_interval` "
                "must be mutually exclusive."
            )
        if monitor is None and save_top_k not in (-1, 0, 1):
            raise MisconfigurationError(
                "`save_top_k` must be -1, 0, or 1 when `monitor` is None."
            )
        if every_n_train_steps is None and every_n_epochs is None and train_time_interval is None:
            resolved_every_n_train_steps: int = 0
            resolved_every_n_epochs: int = 1
        else:
            resolved_every_n_train_steps = every_n_train_steps or 0
            resolved_every_n_epochs = every_n_epochs or 0

        self.dirpath: str | None = None
        if dirpath is not None:
            configured_dirpath: str = os.fspath(dirpath)
            self.dirpath: str | None = (
                os.path.realpath(normalize_path(configured_dirpath))
                if is_local_path(configured_dirpath)
                else configured_dirpath
            )
        self.filename: str | None = filename
        self.monitor: str | None = monitor
        self.verbose: bool = verbose
        self.save_last: bool | Literal["link"] | None = save_last
        self.save_top_k: int = save_top_k
        self.save_on_exception: bool = save_on_exception
        self.save_weights_only: bool = save_weights_only
        self.mode: Literal["min", "max"] = mode
        self.auto_insert_metric_name: bool = auto_insert_metric_name
        self._every_n_train_steps: int = resolved_every_n_train_steps
        self._train_time_interval: timedelta | None = train_time_interval
        self._every_n_epochs: int = resolved_every_n_epochs
        self._save_on_train_epoch_end: bool | None = save_on_train_epoch_end
        self._enable_version_counter: bool = enable_version_counter
        self._last_global_step_saved: int = 0
        self._last_time_checked: float | None = None
        self.current_score: torch.Tensor | None = None
        self.best_k_models: dict[str, torch.Tensor] = {}
        self.kth_best_model_path: str = ""
        self.kth_value: torch.Tensor = torch.tensor(
            torch.inf if mode == "min" else -torch.inf
        )
        self.best_model_score: torch.Tensor | None = None
        self.best_model_path: str = ""
        self.last_model_path: str = ""
        self._last_checkpoint_saved: str = ""
        self._defer_save_until_validation: bool = False

    @override
    def setup(self, trainer: Trainer, module: Module, stage: TrainerStage) -> None:
        # Creates the checkpoint directory on rank zero before the stage
        # begins, so later saves never race on directory creation.
        self._resolve_dirpath(trainer)
        assert self.dirpath is not None
        if self.save_last == "link" and not is_local_path(self.dirpath):
            raise ValueError(
                "`ModelCheckpoint(save_last='link')` is only supported for "
                f"local file paths, got `dirpath={self.dirpath}`."
            )
        if stage == "fit" and not trainer.fast_dev_run and is_rank_zero():
            make_directory(self.dirpath)

    def _resolve_dirpath(self, trainer: Trainer) -> str:
        if self.dirpath is None:
            root: str = (
                trainer.log_dir
                if trainer.loggers and not trainer.fast_dev_run
                else trainer.default_root_dir
            )
            self.dirpath: str | None = join_path(root, "checkpoints")
        return self.dirpath

    @override
    def on_train_start(self, trainer: Trainer, module: Module) -> None:
        self._last_time_checked = time.monotonic()

    @override
    def on_train_batch_end(
        self,
        trainer: Trainer,
        module: Module,
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int
    ) -> None:
        del outputs, batch, batch_idx
        if self._should_skip_saving_checkpoint(trainer):
            return
        step_due: bool = (
            self._every_n_train_steps > 0
            and trainer.global_step % self._every_n_train_steps == 0
        )
        time_due: bool = False
        now: float = time.monotonic()
        if (
            trainer.is_global_zero
            and self._train_time_interval is not None
            and self._last_time_checked is not None
        ):
            time_due = (
                now - self._last_time_checked
                >= self._train_time_interval.total_seconds()
            )
        time_due = trainer.strategy.broadcast(time_due)
        if not step_due and not time_due:
            return
        monitor_candidates: dict[str, torch.Tensor] = self._synchronize_monitor_candidates(
            trainer,
            module
        )
        if self.monitor is not None and self.monitor not in monitor_candidates:
            self._defer_save_until_validation = True
            return
        self._save_topk_checkpoint(trainer, module, monitor_candidates)
        self._save_last_checkpoint(trainer, module, monitor_candidates)
        self._last_time_checked = now

    @override
    def on_validation_end(self, trainer: Trainer, module: Module) -> None:
        # Checkpoint decision point, executed on rank zero after each real
        # validation pass inside fit. The rolling last.ckpt is refreshed
        # first when enabled; the monitored path then requires the monitor
        # key to be present, saves on improvement (or unconditionally with
        # save_top_k=-1), and updates the retention set.
        if self._should_skip_saving_checkpoint(trainer):
            return
        if self._should_save_on_train_epoch_end(trainer) and not self._defer_save_until_validation:
            return
        if (
            not self._defer_save_until_validation
            and (
                self._every_n_epochs < 1
                or (trainer.current_epoch + 1) % self._every_n_epochs != 0
            )
        ):
            return
        monitor_candidates: dict[str, torch.Tensor] = self._synchronize_monitor_candidates(
            trainer,
            module
        )
        self._save_topk_checkpoint(trainer, module, monitor_candidates)
        self._save_last_checkpoint(trainer, module, monitor_candidates)
        self._defer_save_until_validation = False

    @override
    def on_train_epoch_end(self, trainer: Trainer, module: Module) -> None:
        if self._should_skip_saving_checkpoint(trainer):
            return
        if not self._should_save_on_train_epoch_end(trainer):
            return
        if (
            self._every_n_epochs < 1
            or (trainer.current_epoch + 1) % self._every_n_epochs != 0
        ):
            return
        monitor_candidates: dict[str, torch.Tensor] = self._synchronize_monitor_candidates(
            trainer,
            module
        )
        self._save_topk_checkpoint(trainer, module, monitor_candidates)
        self._save_last_checkpoint(trainer, module, monitor_candidates)

    @override
    def on_train_end(self, trainer: Trainer, module: Module) -> None:
        if self.save_last and not self._last_checkpoint_saved:
            monitor_candidates: dict[str, torch.Tensor] = self._synchronize_monitor_candidates(
                trainer,
                module
            )
            self._save_last_checkpoint(trainer, module, monitor_candidates)

    def _should_save_on_validation_end(self, trainer: Trainer, module: Module) -> bool:
        # Applies the configured cadence: a step interval, when set, gates on
        # the global step and takes precedence; otherwise the epoch interval
        # gates on the completed epoch count, and zero disables saving.
        del module
        if self._defer_save_until_validation:
            return True
        if self._should_save_on_train_epoch_end(trainer):
            return False
        return (
            self._every_n_epochs > 0
            and (trainer.current_epoch + 1) % self._every_n_epochs == 0
        )

    def _is_improvement(self, current: torch.Tensor) -> bool:
        # Applies the strict improvement test in the monitored direction.
        if self.best_model_score is None:
            return True
        if self.mode == "min":
            return bool(current < self.best_model_score)
        return bool(current > self.best_model_score)

    def _format_filepath(
        self,
        epoch: int,
        step: int,
        metrics: dict[str, float]
    ) -> str:
        # Resolves the filename template against epoch, step, and the metric
        # values, appends the checkpoint suffix when absent, and disambiguates
        # collisions with retained checkpoints through a version suffix so an
        # earlier retained file is never overwritten.
        candidates: dict[str, torch.Tensor] = {
            name: torch.tensor(value) for name, value in metrics.items()
        }
        candidates["epoch"] = torch.tensor(epoch)
        candidates["step"] = torch.tensor(step)
        return self._get_metric_interpolated_filepath_name(candidates)

    def _save_checkpoint(
        self,
        trainer: Trainer,
        filepath: str | Path
    ) -> None:
        self._save_checkpoint_internal(trainer, filepath, synchronize=True)

    def _save_checkpoint_internal(
        self,
        trainer: Trainer,
        filepath: str | Path,
        synchronize: bool
    ) -> None:
        # Assembles and writes one full-resume checkpoint. Mid-epoch state is
        # read from the training epoch loop: when the batch loop is active,
        # the completed-batch count and the epoch-start random state enter the
        # payload and the current epoch is recorded for replay; at an epoch
        # boundary the next epoch is recorded instead. The module and callback
        # save hooks run on the assembled payload, the write is atomic through
        # a same-directory temporary file, and save duration and size are
        # logged as runtime metrics.
        if trainer.module is None:
            raise RuntimeError("ModelCheckpoint requires an attached Trainer module.")
        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = str(filepath)
        checkpoint_start_time: float = time.perf_counter()
        trainer._save_checkpoint(
            filepath,
            self.save_weights_only,
            None,
            synchronize
        )
        if (
            trainer.is_global_zero
            and is_file(filepath)
            and trainer.loggers
        ):
            checkpoint_duration_ms: float = (
                time.perf_counter() - checkpoint_start_time
            ) * 1000.0
            checkpoint_size_mb: float = file_size(filepath) / 1048576.0
            trainer._log_metrics(
                {
                    "runtime/checkpoint_save_time_ms": checkpoint_duration_ms,
                    "runtime/checkpoint_size_mb": checkpoint_size_mb
                },
                step=trainer.state.global_step
            )

    def _save_partial_checkpoint(
        self,
        module: Module,
        filepath: str | Path
    ) -> None:
        # Writes a weights-only payload marked partial in its metadata, used
        # by the exception path when the run failed before the optimizer
        # existed. The write follows the same atomic temporary-file protocol
        # as full checkpoints.
        trainer: Trainer = module.trainer
        profile_context: AbstractContextManager[str] | AbstractContextManager[None] = (
            trainer.profiler.profile("save_checkpoint")
            if trainer._profiler_is_setup
            else nullcontext()
        )
        with profile_context:
            model_state_dict: CheckpointDict = cast(
                CheckpointDict,
                module.state_dict()
            )
            checkpoint: CheckpointDict = {
                "state_dict": model_state_dict,
                "model_state_dict": model_state_dict,
                "metadata": {
                    "torch_version": str(torch.__version__),
                    "partial": True
                }
            }
            trainer._call_module_hook(
                module,
                "on_save_checkpoint",
                module.on_save_checkpoint,
                checkpoint
            )

            trainer.strategy.save_checkpoint(checkpoint, filepath)

    def _update_top_k(
        self,
        score: torch.Tensor,
        filepath: str | Path
    ) -> None:
        # Inserts the new checkpoint into the retention set, sorts by score in
        # the monitored direction, and evicts entries beyond the retention
        # count, deleting their files unless a file is the recorded best path.
        # Unbounded retention (save_top_k=-1) appends without eviction.
        self.best_k_models[str(filepath)] = score
        if not self.best_k_models:
            return
        best_operation = min if self.mode == "min" else max
        worst_operation = max if self.mode == "min" else min
        self.best_model_path = best_operation(
            self.best_k_models,
            key=self.best_k_models.__getitem__
        )
        self.best_model_score = self.best_k_models[self.best_model_path]
        self.kth_best_model_path = worst_operation(
            self.best_k_models,
            key=self.best_k_models.__getitem__
        )
        self.kth_value = self.best_k_models[self.kth_best_model_path]

    def _should_skip_saving_checkpoint(self, trainer: Trainer) -> bool:
        return (
            bool(trainer.fast_dev_run)
            or trainer.state.stage != "fit"
            or trainer._sanity_checking
            or self._last_global_step_saved == trainer.global_step
        )

    def _should_save_on_train_epoch_end(self, trainer: Trainer) -> bool:
        if self._save_on_train_epoch_end is not None:
            return self._save_on_train_epoch_end
        if trainer.check_val_every_n_epoch != 1:
            return False
        if not trainer.enable_validation or trainer.num_val_batches == 0:
            return True
        return trainer.val_check_interval == 1.0

    def _monitor_candidates(
        self,
        trainer: Trainer,
        module: Module
    ) -> dict[str, torch.Tensor]:
        candidates: dict[str, torch.Tensor] = {
            name: torch.tensor(value)
            for name, value in trainer.callback_metrics.items()
        }
        for name, metric in module.logged_metrics.items():
            value: torch.Tensor | float = metric.value
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    candidates[name] = value.detach().cpu()
            else:
                candidates[name] = torch.tensor(value)
        candidates["epoch"] = torch.tensor(trainer.current_epoch)
        candidates["step"] = torch.tensor(trainer.global_step)
        return candidates

    def _synchronize_monitor_candidates(
        self,
        trainer: Trainer,
        module: Module
    ) -> dict[str, torch.Tensor]:
        candidates: dict[str, torch.Tensor] = (
            self._monitor_candidates(trainer, module)
            if trainer.is_global_zero
            else {}
        )
        return trainer.strategy.broadcast(candidates)

    def _format_checkpoint_name(
        self,
        filename: str | None,
        metrics: dict[str, torch.Tensor],
        prefix: str | None = None,
        auto_insert_metric_name: bool = True
    ) -> str:
        resolved_filename: str = filename or "{epoch}-{step}"
        groups: list[str] = sorted(
            re.findall(r"(\{.*?)[:\}]", resolved_filename),
            key=len,
            reverse=True
        )
        for group in groups:
            name: str = group[1:]
            if auto_insert_metric_name:
                resolved_filename = resolved_filename.replace(
                    group,
                    name + self.CHECKPOINT_EQUALS_CHAR + "{" + name
                )
            resolved_filename = resolved_filename.replace(group, f"{{0[{name}]")
            if name not in metrics:
                metrics[name] = torch.tensor(0)
        resolved_filename = resolved_filename.format(metrics)
        if prefix is not None:
            resolved_filename = self.CHECKPOINT_JOIN_CHAR.join(
                (prefix, resolved_filename)
            )
        return resolved_filename

    def format_checkpoint_name(
        self,
        metrics: dict[str, torch.Tensor],
        filename: str | None = None,
        ver: int | None = None,
        prefix: str | None = None
    ) -> str:
        resolved_filename: str = self._format_checkpoint_name(
            filename or self.filename,
            dict(metrics),
            prefix=prefix,
            auto_insert_metric_name=self.auto_insert_metric_name
        )
        if ver is not None:
            resolved_filename = self.CHECKPOINT_JOIN_CHAR.join(
                (resolved_filename, f"v{ver}")
            )
        name: str = f"{resolved_filename}{self.FILE_EXTENSION}"
        return join_path(self.dirpath, name) if self.dirpath is not None else name

    def _get_metric_interpolated_filepath_name(
        self,
        monitor_candidates: dict[str, torch.Tensor],
        trainer: Trainer | None = None,
        del_filepath: str | None = None
    ) -> str:
        filepath: str = self.format_checkpoint_name(monitor_candidates)
        if self._enable_version_counter:
            version: int = self.STARTING_VERSION
            while (
                self.file_exists(filepath, trainer)
                if trainer is not None
                else exists(filepath)
            ) and filepath != del_filepath:
                filepath = self.format_checkpoint_name(
                    monitor_candidates,
                    ver=version
                )
                version += 1
        return filepath

    def check_monitor_top_k(
        self,
        trainer: Trainer,
        current: torch.Tensor | None = None
    ) -> bool:
        if current is None or self.save_top_k == 0:
            return False
        if self.save_top_k == -1:
            return True
        if len(self.best_k_models) < self.save_top_k:
            return True
        if not self.kth_best_model_path:
            return True
        kth_value: torch.Tensor = self.best_k_models[self.kth_best_model_path]
        should_update: bool = (
            bool(current < kth_value)
            if self.mode == "min"
            else bool(current > kth_value)
        )
        return trainer.strategy.reduce_boolean_decision(should_update)

    def _save_topk_checkpoint(
        self,
        trainer: Trainer,
        module: Module,
        monitor_candidates: dict[str, torch.Tensor]
    ) -> None:
        if self.save_top_k == 0:
            return
        if self.monitor is None:
            self._save_none_monitor_checkpoint(
                trainer,
                module,
                monitor_candidates
            )
            return
        current: torch.Tensor | None = monitor_candidates.get(self.monitor)
        if current is None:
            from syntheticmind.utilities.exceptions import MisconfigurationError

            raise MisconfigurationError(
                f"`ModelCheckpoint(monitor={self.monitor!r})` could not find the "
                f"monitored key in metrics: {list(monitor_candidates)}."
            )
        if current.numel() != 1:
            from syntheticmind.utilities.exceptions import MisconfigurationError

            raise MisconfigurationError(
                f"`ModelCheckpoint(monitor={self.monitor!r})` requires a scalar metric."
            )
        if self.check_monitor_top_k(trainer, current):
            self._update_best_and_save(
                current,
                trainer,
                module,
                monitor_candidates
            )
        elif self.verbose and trainer.is_global_zero:
            log.info(
                "Checkpoint skipped | epoch={} | step={} | {}={} | outside top {}",
                int(monitor_candidates["epoch"].item()),
                int(monitor_candidates["step"].item()),
                self.monitor,
                float(current.item()),
                self.save_top_k
            )

    def _save_none_monitor_checkpoint(
        self,
        trainer: Trainer,
        module: Module,
        monitor_candidates: dict[str, torch.Tensor]
    ) -> None:
        filepath: str = self._get_metric_interpolated_filepath_name(
            monitor_candidates,
            trainer,
            self.best_model_path or None
        )
        previous: str = self.best_model_path
        self.best_model_path = filepath
        self._save_checkpoint(trainer, filepath)
        if self.verbose and trainer.is_global_zero:
            log.info(
                "Checkpoint saved | epoch={} | step={} | {}",
                int(monitor_candidates["epoch"].item()),
                int(monitor_candidates["step"].item()),
                filepath
            )
        if (
            self.save_top_k == 1
            and previous
            and self._should_remove_checkpoint(trainer, previous, filepath)
        ):
            self._remove_checkpoint(trainer, previous)

    def _update_best_and_save(
        self,
        current: torch.Tensor,
        trainer: Trainer,
        module: Module,
        monitor_candidates: dict[str, torch.Tensor]
    ) -> None:
        resolved_current: torch.Tensor = current.detach().cpu()
        if bool(torch.isnan(resolved_current)):
            replacement: float = torch.inf if self.mode == "min" else -torch.inf
            resolved_current = torch.tensor(
                replacement,
                device=resolved_current.device
            )
        delete_filepath: str | None = None
        if (
            self.save_top_k > 0
            and len(self.best_k_models) >= self.save_top_k
            and self.kth_best_model_path
        ):
            delete_filepath = self.kth_best_model_path
            self.best_k_models.pop(delete_filepath, None)
        filepath: str = self._get_metric_interpolated_filepath_name(
            monitor_candidates,
            trainer,
            delete_filepath
        )
        self.current_score = resolved_current
        self._update_top_k(resolved_current, filepath)
        self._save_checkpoint(trainer, filepath)
        if self.verbose and trainer.is_global_zero:
            log.info(
                "Checkpoint saved | epoch={} | step={} | {}={} | {}",
                int(monitor_candidates["epoch"].item()),
                int(monitor_candidates["step"].item()),
                self.monitor,
                float(resolved_current.item()),
                filepath
            )
        if (
            delete_filepath
            and self._should_remove_checkpoint(
                trainer,
                delete_filepath,
                filepath
            )
        ):
            self._remove_checkpoint(trainer, delete_filepath)

    def _save_last_checkpoint(
        self,
        trainer: Trainer,
        module: Module,
        monitor_candidates: dict[str, torch.Tensor],
        synchronize: bool = True
    ) -> None:
        if not self.save_last:
            return
        filepath: str = self.format_checkpoint_name(
            monitor_candidates,
            self.CHECKPOINT_NAME_LAST
        )
        if self._enable_version_counter:
            version: int = self.STARTING_VERSION
            while (
                self.file_exists(filepath, trainer)
                if synchronize
                else exists(filepath)
            ) and filepath != self.last_model_path:
                filepath = self.format_checkpoint_name(
                    monitor_candidates,
                    self.CHECKPOINT_NAME_LAST,
                    ver=version
                )
                version += 1
        previous: str = self.last_model_path
        self.last_model_path = filepath
        if (
            self.save_last == "link"
            and self._last_checkpoint_saved
            and self.save_top_k != 0
        ):
            self._link_checkpoint_internal(
                trainer,
                self._last_checkpoint_saved,
                filepath,
                synchronize
            )
        else:
            self._save_checkpoint_internal(
                trainer,
                filepath,
                synchronize
            )
        if (
            previous
            and self._should_remove_checkpoint(trainer, previous, filepath)
        ):
            self._remove_checkpoint(trainer, previous)

    @staticmethod
    def _link_checkpoint(
        trainer: Trainer,
        filepath: str | Path,
        linkpath: str | Path
    ) -> None:
        ModelCheckpoint._link_checkpoint_internal(
            trainer,
            filepath,
            linkpath,
            True
        )

    @staticmethod
    def _link_checkpoint_internal(
        trainer: Trainer,
        filepath: str | Path,
        linkpath: str | Path,
        synchronize: bool
    ) -> None:
        if not is_local_path(filepath) or not is_local_path(linkpath):
            raise ValueError("Checkpoint links require local filesystem paths.")
        local_filepath: Path = Path(normalize_path(filepath))
        local_linkpath: Path = Path(normalize_path(linkpath))
        if not trainer.is_global_zero:
            if synchronize:
                trainer._call_strategy_hook("barrier", trainer.strategy.barrier)
            return
        if local_filepath.resolve() == local_linkpath.resolve():
            if synchronize:
                trainer._call_strategy_hook("barrier", trainer.strategy.barrier)
            return
        try:
            local_linkpath.parent.mkdir(parents=True, exist_ok=True)
            if local_linkpath.is_symlink() or local_linkpath.is_file():
                local_linkpath.unlink()
            try:
                local_linkpath.symlink_to(
                    os.path.relpath(local_filepath, local_linkpath.parent)
                )
            except OSError:
                shutil.copy2(local_filepath, local_linkpath)
        finally:
            if synchronize:
                trainer._call_strategy_hook("barrier", trainer.strategy.barrier)

    def _should_remove_checkpoint(
        self,
        trainer: Trainer,
        previous: str,
        current: str
    ) -> bool:
        if previous == current:
            return False
        if not is_local_path(previous):
            return True
        resume_value: object = getattr(
            trainer,
            "ckpt_path",
            trainer.resume_from_checkpoint
        )
        if resume_value is not None and same_path(previous, str(resume_value)):
            return False
        if self.dirpath is None:
            return False
        return is_path_within(previous, self.dirpath)

    def _remove_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        del self
        trainer.strategy.remove_checkpoint(filepath)

    @property
    def every_n_epochs(self) -> int | None:
        return self._every_n_epochs

    def file_exists(self, filepath: str | Path, trainer: Trainer) -> bool:
        path_exists: bool = exists(filepath) if trainer.is_global_zero else False
        return trainer.strategy.reduce_boolean_decision(path_exists)

    def to_yaml(self, filepath: str | Path | None = None) -> None:
        import yaml

        if filepath is None:
            if self.dirpath is None:
                raise RuntimeError("ModelCheckpoint directory is unresolved before setup.")
            filepath = join_path(self.dirpath, "best_k_models.yaml")
        values: dict[str, float] = {
            path: float(score.item()) for path, score in self.best_k_models.items()
        }
        write_text(filepath, yaml.safe_dump(values))

    @override
    def on_exception(self, trainer: Trainer, module: Module, exception: BaseException) -> None:
        # Optional last-resort save on run failure, restricted to rank zero
        # during fit. A full checkpoint is attempted when the optimizer
        # exists; otherwise the partial weights-only form is written. A
        # failure while saving is logged and suppressed so the original
        # exception continues to propagate.
        if trainer.fast_dev_run or not self.save_on_exception:
            return
        if not is_rank_zero():
            return
        if trainer.state.stage != "fit":
            return
        del exception
        dirpath: str = self._resolve_dirpath(trainer)
        monitor_candidates: dict[str, torch.Tensor] = self._monitor_candidates(
            trainer,
            module
        )
        exception_path: str = self._get_metric_interpolated_filepath_name(
            monitor_candidates
        )
        try:
            make_directory(dirpath)
            self._save_checkpoint_internal(
                trainer,
                exception_path,
                False
            )
            self._save_last_checkpoint(
                trainer,
                module,
                monitor_candidates,
                synchronize=False
            )
        except Exception:
            log.warning("Failed to save exception checkpoint")

    @property
    def state_key(self) -> str:
        # Qualifies the checkpoint identity with monitor, direction, and both
        # cadence options so differently configured instances persist their
        # retention state independently.
        return self._generate_state_key(
            monitor=self.monitor,
            mode=self.mode,
            every_n_epochs=self._every_n_epochs,
            every_n_train_steps=self._every_n_train_steps,
            train_time_interval=(
                str(self._train_time_interval)
                if self._train_time_interval is not None
                else None
            )
        )

    @override
    def state_dict(self) -> StateDict:
        # Persists the selection state: best score and path, the retained
        # top-k set with stringified paths, the collision counter, and the
        # step cadence for inspection.
        state_dict: StateDict = {
            "monitor": self.monitor,
            "best_model_score": self.best_model_score,
            "best_model_path": self.best_model_path,
            "current_score": self.current_score,
            "dirpath": str(self.dirpath) if self.dirpath is not None else None,
            "best_k_models": cast(StateValue, self.best_k_models),
            "kth_best_model_path": self.kth_best_model_path,
            "kth_value": self.kth_value,
            "last_model_path": self.last_model_path
        }
        return state_dict

    @override
    def load_state_dict(self, state_dict: StateDict) -> None:
        # Restores the selection state, then applies the cross-directory
        # protection: when any restored top-k path resolves outside the
        # current checkpoint directory, the retention state is cleared so
        # eviction can never delete files that belong to another directory.
        best_model_path_value: StateValue = state_dict.get("best_model_path", "")
        if isinstance(best_model_path_value, str):
            self.best_model_path = best_model_path_value
        dirpath_value: StateValue = state_dict.get("dirpath")
        current_dirpath: str | None = (
            str(self.dirpath) if self.dirpath is not None else None
        )
        restored_dirpath: str | None = (
            dirpath_value if isinstance(dirpath_value, str) else None
        )
        if self.dirpath is None and restored_dirpath is not None:
            self.dirpath = restored_dirpath
            current_dirpath = restored_dirpath
        if current_dirpath != restored_dirpath:
            warnings.warn(
                f"The checkpoint directory changed from {restored_dirpath!r} to "
                f"{current_dirpath!r}; only `best_model_path` was restored.",
                stacklevel=2
            )
            return
        best_model_score_value: StateValue = state_dict.get("best_model_score")
        self.best_model_score = (
            best_model_score_value
            if isinstance(best_model_score_value, torch.Tensor)
            else None
        )
        current_score_value: StateValue = state_dict.get("current_score")
        self.current_score = (
            current_score_value
            if isinstance(current_score_value, torch.Tensor)
            else None
        )
        best_k_models_value: StateValue = state_dict.get("best_k_models", {})
        self.best_k_models = {}
        if isinstance(best_k_models_value, dict):
            for path, score in best_k_models_value.items():
                if isinstance(path, str) and isinstance(score, torch.Tensor):
                    self.best_k_models[path] = score
        kth_best_model_path_value: StateValue = state_dict.get(
            "kth_best_model_path",
            ""
        )
        if isinstance(kth_best_model_path_value, str):
            self.kth_best_model_path = kth_best_model_path_value
        kth_value_value: StateValue = state_dict.get("kth_value")
        if isinstance(kth_value_value, torch.Tensor):
            self.kth_value = kth_value_value
        last_model_path_value: StateValue = state_dict.get("last_model_path", "")
        if isinstance(last_model_path_value, str):
            self.last_model_path = last_model_path_value

    @override
    def __repr__(self) -> str:
        # Compact configuration line for logs and interactive debugging.
        return (
            f"ModelCheckpoint("
            f"monitor={self.monitor!r}, "
            f"mode={self.mode!r}, "
            f"save_top_k={self.save_top_k}, "
            f"every_n_train_steps={self._every_n_train_steps}, "
            f"dirpath={str(self.dirpath)!r})"
        )
