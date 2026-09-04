import time
import warnings
from pathlib import Path
from typing import Literal

import syntheticmind
import torch
from loguru import logger as log

from syntheticmind.callbacks.checkpointing.paths import _CheckpointPaths
from syntheticmind.callbacks.checkpointing.ranking import (
    _CheckpointRanking,
    _RankingSnapshot
)
from syntheticmind.callbacks.checkpointing.schedule import (
    _CheckpointSchedule,
    _ScheduleSnapshot
)
from syntheticmind.exceptions import MisconfigurationError
from syntheticmind.storage.filesystem import StoragePath
from syntheticmind.types import StateDict, StateValue

__all__: list[str] = []

type _TransactionSnapshot = tuple[_ScheduleSnapshot, _RankingSnapshot, str, str]


class _CheckpointTransaction:
    def __init__(
        self,
        paths: _CheckpointPaths,
        ranking: _CheckpointRanking,
        schedule: _CheckpointSchedule,
        verbose: bool,
        save_last: bool | Literal["link"] | None,
        save_weights_only: bool
    ) -> None:
        self._paths: _CheckpointPaths = paths
        self._ranking: _CheckpointRanking = ranking
        self._schedule: _CheckpointSchedule = schedule
        self._verbose: bool = verbose
        self._save_last: bool | Literal["link"] | None = save_last
        self._save_weights_only: bool = save_weights_only
        self.last_model_path: str = ""
        self.last_checkpoint_saved: str = ""

    def save_trigger(
        self,
        trainer: syntheticmind.Trainer,
        candidates: dict[str, torch.Tensor],
        trigger_time: float,
        synchronize: bool
    ) -> None:
        previous_last_path: str = self.last_model_path
        planned_link_path: str | None = None
        if self._save_last == "link":
            planned_link_path: str | None = self._paths.last_filepath(
                candidates,
                self.last_model_path,
                trainer if synchronize else None
            )
            self.last_model_path: str = planned_link_path
        try:
            saved_top_k_path: str | None = self._save_top_k_checkpoint(
                trainer,
                candidates,
                trigger_time,
                synchronize
            )
        except BaseException:
            self.last_model_path: str = previous_last_path
            raise
        if self._save_last == "link" and saved_top_k_path is not None:
            assert planned_link_path is not None
            self._paths.link_checkpoint(
                trainer,
                saved_top_k_path,
                planned_link_path,
                synchronize
            )
            self._remove_after_save(
                trainer,
                previous_last_path or None,
                planned_link_path
            )
            return
        if self._save_last == "link":
            self.last_model_path: str = previous_last_path
        self._save_last_checkpoint(
            trainer,
            candidates,
            saved_top_k_path,
            trigger_time,
            synchronize
        )
        if saved_top_k_path is None and not self._save_last:
            self._schedule.mark_time_checked(trigger_time)

    def save_last_if_stale(
        self,
        trainer: syntheticmind.Trainer,
        candidates: dict[str, torch.Tensor],
        trigger_time: float
    ) -> None:
        if (
            not self._save_last
            or self._schedule.last_global_step_saved == trainer.global_step
        ):
            return
        self._save_last_checkpoint(
            trainer,
            candidates,
            None,
            trigger_time,
            True
        )

    def save_exception(
        self,
        trainer: syntheticmind.Trainer,
        candidates: dict[str, torch.Tensor],
        exception: BaseException
    ) -> None:
        filepath: str = self._paths.ranked_filepath(candidates)
        snapshot: _TransactionSnapshot = self.snapshot()
        self._schedule.mark_saved(trainer.global_step, time.monotonic())
        self.last_checkpoint_saved: str = filepath
        try:
            StoragePath(self._paths.resolve_directory(trainer)).make_directory()
            self._persist_checkpoint(trainer, filepath, False)
            self._save_last_checkpoint(
                trainer,
                candidates,
                filepath,
                time.monotonic(),
                False
            )
        except BaseException as save_exception:
            self.restore_snapshot(snapshot)
            log.warning(
                f"Exception checkpoint failed with {type(save_exception).__name__}: "
                f"{save_exception}. Original failure: {type(exception).__name__}."
            )

    def state_dict(self) -> StateDict:
        state: StateDict = self._ranking.state_dict()
        state.update(self._schedule.state_dict())
        state["dirpath"] = self._paths.dirpath
        state["last_model_path"] = self.last_model_path
        state["last_checkpoint_saved"] = self.last_checkpoint_saved
        return state

    def load_state_dict(self, state_dict: StateDict) -> None:
        self._schedule.load_state_dict(state_dict)
        restored_dirpath_value: StateValue | None = state_dict.get("dirpath")
        if restored_dirpath_value is not None and not isinstance(restored_dirpath_value, str):
            raise TypeError("ModelCheckpoint dirpath state must be a string or None.")
        restored_dirpath: str | None = (
            restored_dirpath_value
            if isinstance(restored_dirpath_value, str)
            else None
        )
        current_dirpath: str | None = self._paths.dirpath
        if current_dirpath is None and restored_dirpath is not None:
            self._paths.dirpath: str | None = restored_dirpath
            current_dirpath: str | None = restored_dirpath
        last_path: str = self._required_string(state_dict, "last_model_path")
        last_saved: str = self._required_string(
            state_dict,
            "last_checkpoint_saved"
        )
        if current_dirpath != restored_dirpath:
            warnings.warn(
                f"Checkpoint directory changed from {restored_dirpath!r} to "
                f"{current_dirpath!r}; ranking paths were reset.",
                UserWarning,
                stacklevel=3
            )
            self._ranking.reset()
            self.last_model_path: str = ""
            self.last_checkpoint_saved: str = ""
            return
        self._ranking.load_state_dict(state_dict)
        self.last_model_path: str = last_path
        self.last_checkpoint_saved: str = last_saved

    def snapshot(self) -> _TransactionSnapshot:
        return (
            self._schedule.snapshot(),
            self._ranking.snapshot(),
            self.last_model_path,
            self.last_checkpoint_saved
        )

    def restore_snapshot(self, snapshot: _TransactionSnapshot) -> None:
        self._schedule.restore_snapshot(snapshot[0])
        self._ranking.restore_snapshot(snapshot[1])
        self.last_model_path: str = snapshot[2]
        self.last_checkpoint_saved: str = snapshot[3]

    def _save_top_k_checkpoint(
        self,
        trainer: syntheticmind.Trainer,
        candidates: dict[str, torch.Tensor],
        trigger_time: float,
        synchronize: bool
    ) -> str | None:
        if self._ranking.save_top_k == 0:
            return None
        if self._ranking.monitor is None:
            return self._save_unmonitored_checkpoint(
                trainer,
                candidates,
                trigger_time,
                synchronize
            )
        current: torch.Tensor | None = candidates.get(self._ranking.monitor)
        if current is None:
            raise MisconfigurationError(
                f"ModelCheckpoint monitor {self._ranking.monitor!r} is unavailable. "
                f"Available metrics: {sorted(candidates)}."
            )
        if current.numel() != 1:
            raise MisconfigurationError(
                f"ModelCheckpoint monitor {self._ranking.monitor!r} must be scalar."
            )
        if not self._ranking.qualifies(trainer, current):
            if self._verbose and trainer.is_global_zero:
                log.info(
                    f"Checkpoint skipped at epoch={trainer.current_epoch}, "
                    f"step={trainer.global_step}; {self._ranking.monitor}={float(current)} "
                    f"is outside top {self._ranking.save_top_k}."
                )
            return None
        return self._save_ranked_checkpoint(
            trainer,
            candidates,
            current,
            trigger_time,
            synchronize
        )

    def _save_unmonitored_checkpoint(
        self,
        trainer: syntheticmind.Trainer,
        candidates: dict[str, torch.Tensor],
        trigger_time: float,
        synchronize: bool
    ) -> str:
        filepath: str = self._paths.ranked_filepath(
            candidates,
            trainer if synchronize else None
        )
        snapshot: _TransactionSnapshot = self.snapshot()
        delete_path: str | None = self._ranking.accept_unmonitored(filepath)
        self._schedule.mark_saved(trainer.global_step, trigger_time)
        self.last_checkpoint_saved: str = filepath
        try:
            self._persist_checkpoint(trainer, filepath, synchronize)
        except BaseException:
            self.restore_snapshot(snapshot)
            raise
        if self._verbose and trainer.is_global_zero:
            log.info(
                f"Checkpoint saved at epoch={trainer.current_epoch}, "
                f"step={trainer.global_step}: {filepath}"
            )
        self._remove_after_save(trainer, delete_path, filepath)
        return filepath

    def _save_ranked_checkpoint(
        self,
        trainer: syntheticmind.Trainer,
        candidates: dict[str, torch.Tensor],
        current: torch.Tensor,
        trigger_time: float,
        synchronize: bool
    ) -> str:
        filepath: str = self._paths.ranked_filepath(
            candidates,
            trainer if synchronize else None
        )
        snapshot: _TransactionSnapshot = self.snapshot()
        delete_path: str | None = self._ranking.accept_ranked(current, filepath)
        self._schedule.mark_saved(trainer.global_step, trigger_time)
        self.last_checkpoint_saved: str = filepath
        try:
            self._persist_checkpoint(trainer, filepath, synchronize)
        except BaseException:
            self.restore_snapshot(snapshot)
            raise
        if self._verbose and trainer.is_global_zero:
            assert self._ranking.current_score is not None
            log.info(
                f"Checkpoint saved at epoch={trainer.current_epoch}, "
                f"step={trainer.global_step}, {self._ranking.monitor}="
                f"{float(self._ranking.current_score)}: {filepath}"
            )
        self._remove_after_save(trainer, delete_path, filepath)
        return filepath

    def _save_last_checkpoint(
        self,
        trainer: syntheticmind.Trainer,
        candidates: dict[str, torch.Tensor],
        source_filepath: str | None,
        trigger_time: float,
        synchronize: bool
    ) -> None:
        if not self._save_last:
            return
        filepath: str = self._paths.last_filepath(
            candidates,
            self.last_model_path,
            trainer if synchronize else None
        )
        previous_path: str = self.last_model_path
        snapshot: _TransactionSnapshot = self.snapshot()
        self.last_model_path: str = filepath
        self._schedule.mark_saved(trainer.global_step, trigger_time)
        try:
            if self._save_last == "link" and source_filepath is not None:
                self.last_checkpoint_saved: str = source_filepath
                self._paths.link_checkpoint(
                    trainer,
                    source_filepath,
                    filepath,
                    synchronize
                )
            else:
                self.last_checkpoint_saved: str = filepath
                self._persist_checkpoint(trainer, filepath, synchronize)
        except BaseException:
            self.restore_snapshot(snapshot)
            raise
        self._remove_after_save(trainer, previous_path or None, filepath)

    def _persist_checkpoint(
        self,
        trainer: syntheticmind.Trainer,
        filepath: str | Path,
        synchronize: bool
    ) -> None:
        if trainer.module is None:
            raise RuntimeError("ModelCheckpoint requires an attached Trainer module.")
        started_at: float = time.perf_counter()
        trainer._save_checkpoint(
            filepath,
            self._save_weights_only,
            None,
            synchronize
        )
        checkpoint_path: StoragePath = StoragePath(filepath)
        if trainer.is_global_zero and checkpoint_path.is_file() and trainer.loggers:
            trainer._log_metrics(
                {
                    "runtime/checkpoint_save_time_ms": (
                        time.perf_counter() - started_at
                    ) * 1000.0,
                    "runtime/checkpoint_size_mb": checkpoint_path.size / 1048576.0
                },
                step=trainer.global_step
            )

    def _remove_after_save(
        self,
        trainer: syntheticmind.Trainer,
        delete_path: str | None,
        current_path: str
    ) -> None:
        if (
            delete_path is not None
            and self._paths.should_remove(trainer, delete_path, current_path)
        ):
            trainer._checkpoint_connector.remove(delete_path)

    def _required_string(self, state_dict: StateDict, name: str) -> str:
        value: StateValue | None = state_dict.get(name)
        if not isinstance(value, str):
            raise TypeError(f"ModelCheckpoint {name} state must be a string.")
        return value
