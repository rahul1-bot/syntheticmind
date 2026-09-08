# This module:
# 1. Captures bounded CUDA allocator history for an explicitly selected fit stage
#
# Design decisions:
# - The callback records only on a CUDA Strategy and owns recorder shutdown
# - Failure snapshots are rank-qualified and unique rather than overwriting runs
# - CUDA allocator tracing is process-global; use one recorder per process
# - Diagnostic failure is reported through Trainer without hiding training errors
#
# Author: Rahul Sawhney

from pathlib import Path
from typing import Literal, override
from uuid import uuid7

import syntheticmind
import torch

from syntheticmind.callbacks.callback import Callback
from syntheticmind.primitives.types import TrainerStage

__all__: list[str] = ["CUDAMemorySnapshot"]

type MemoryHistoryContext = Literal["state", "alloc", "all"]
type MemoryHistoryStacks = Literal["python", "all"]
type MemorySnapshotFailurePolicy = Literal["out_of_memory", "any_failure"]


class CUDAMemorySnapshot(Callback):
    # Opt-in diagnostic callback. Snapshots remain in the explicit local directory;
    # they can include source paths and stacks and are never uploaded automatically.
    def __init__(
        self,
        dirpath: Path,
        max_entries: int = 100000,
        context: MemoryHistoryContext = "all",
        stacks: MemoryHistoryStacks = "python",
        failure_policy: MemorySnapshotFailurePolicy = "out_of_memory"
    ) -> None:
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError(
                f"CUDA memory history max_entries must be positive, got {max_entries}."
            )
        if context not in {"state", "alloc", "all"}:
            raise ValueError(f"Unsupported CUDA memory context: {context!r}.")
        if stacks not in {"python", "all"}:
            raise ValueError(f"Unsupported CUDA memory stack policy: {stacks!r}.")
        if failure_policy not in {"out_of_memory", "any_failure"}:
            raise ValueError(
                f"Unsupported CUDA memory failure policy: {failure_policy!r}."
            )
        self._dirpath: Path = Path(dirpath)
        self._max_entries: int = max_entries
        self._context: MemoryHistoryContext = context
        self._stacks: MemoryHistoryStacks = stacks
        self._failure_policy: MemorySnapshotFailurePolicy = failure_policy
        self._is_recording: bool = False
        self._device: torch.device | None = None

    @override
    def setup(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module,
        stage: TrainerStage
    ) -> None:
        del module
        if stage != "fit":
            return
        callback: Callback
        if sum(isinstance(callback, CUDAMemorySnapshot) for callback in trainer.callbacks) != 1:
            raise ValueError("Configure exactly one CUDAMemorySnapshot per Trainer.")
        if self._is_recording:
            raise RuntimeError("CUDA memory recording is already active on this callback.")
        if trainer.strategy.root_device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "CUDAMemorySnapshot requires a CUDA runtime. Remove the callback "
                "for CPU or MPS execution."
            )
        self._dirpath.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._record_memory_history(
            enabled="all",
            context=self._context,
            stacks=self._stacks,
            max_entries=self._max_entries,
            device=trainer.strategy.root_device
        )
        self._device: torch.device | None = trainer.strategy.root_device
        self._is_recording: bool = True

    @override
    def teardown(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module,
        stage: TrainerStage
    ) -> None:
        del trainer, module
        if stage != "fit" or not self._is_recording:
            return
        try:
            torch.cuda.memory._record_memory_history(enabled=None, device=self._device)
        finally:
            self._is_recording: bool = False
            self._device: torch.device | None = None

    @override
    def on_exception(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module,
        exception: BaseException
    ) -> None:
        del module
        if not self._is_recording:
            return
        if (
            self._failure_policy == "out_of_memory"
            and not isinstance(exception, torch.OutOfMemoryError)
        ):
            return
        snapshot_path: Path = self._dirpath / (
            f"cuda_memory_rank_{trainer.global_rank}_step_{trainer.global_step}_{uuid7()}.pickle"
        )
        torch.cuda.memory._dump_snapshot(str(snapshot_path))
