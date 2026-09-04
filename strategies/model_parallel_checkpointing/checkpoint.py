from collections.abc import Callable, Mapping
from typing import override

import torch
from torch import nn

from syntheticmind.plugins.checkpoint_io import (
    CheckpointIO,
    CheckpointPath,
    CheckpointStorageOptions
)
from syntheticmind.strategies.checkpoint import StrategyCheckpoint
from syntheticmind.strategies.model_parallel_checkpointing.checkpoint_io import (
    ModelParallelCheckpointFormat,
    _ModelParallelCheckpointIO
)
from syntheticmind.strategies.model_parallel_checkpointing.generation import CheckpointBroadcast
from syntheticmind.exceptions import MisconfigurationError
from syntheticmind.storage.filesystem import is_file, normalize_path
from syntheticmind.types import (
    CheckpointDict,
    CheckpointValue,
    ModelState,
    OptimizerState
)

__all__: list[str] = []


class _ModelParallelCheckpoint(StrategyCheckpoint):
    def __init__(
        self,
        checkpoint_format: ModelParallelCheckpointFormat,
        barrier: Callable[[], None],
        broadcast: CheckpointBroadcast
    ) -> None:
        self._checkpoint_io: _ModelParallelCheckpointIO = _ModelParallelCheckpointIO(
            checkpoint_format
        )
        self._barrier: Callable[[], None] = barrier
        self._broadcast: CheckpointBroadcast = broadcast
        self._model: nn.Module | None = None
        self._optimizers: list[torch.optim.Optimizer] = []
        self._preloaded_path: str | None = None
        self._preloaded_checkpoint: CheckpointDict | None = None
        self._pending_path: str | None = None
        self._pending_checkpoint: CheckpointDict | None = None
        self._pending_weights_only: bool | None = None

    @override
    def connect_model(self, model: nn.Module) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("Checkpoint model must be a torch.nn.Module.")
        self._model: nn.Module | None = model

    @override
    def connect_optimizers(self, optimizers: list[torch.optim.Optimizer]) -> None:
        if not isinstance(optimizers, list):
            raise TypeError("Checkpoint optimizers must be a list.")
        optimizer: torch.optim.Optimizer
        for optimizer in optimizers:
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise TypeError("Checkpoint optimizers must contain Optimizer instances.")
        self._optimizers: list[torch.optim.Optimizer] = list(optimizers)

    @override
    def save(
        self,
        checkpoint: CheckpointDict,
        filepath: CheckpointPath,
        storage_options: CheckpointStorageOptions,
        global_rank: int
    ) -> None:
        if storage_options is not None:
            raise TypeError("Model-parallel checkpoints do not accept storage_options.")
        path: str = self._synchronize_path(filepath)
        self._checkpoint_io.save(
            checkpoint,
            path,
            global_rank,
            self._barrier,
            self._broadcast
        )

    @override
    def load(
        self,
        checkpoint_path: CheckpointPath,
        weights_only: bool | None
    ) -> CheckpointDict:
        path: str = self._synchronize_path(checkpoint_path)
        checkpoint: CheckpointDict | None = (
            self._preloaded_checkpoint
            if self._preloaded_path == path
            else None
        )
        self._preloaded_path: str | None = None
        self._preloaded_checkpoint: CheckpointDict | None = None
        if checkpoint is None:
            checkpoint: CheckpointDict | None = self._checkpoint_io.load_metadata(
                path,
                weights_only
            )
        self._pending_path: str | None = path
        self._pending_checkpoint: CheckpointDict | None = (
            checkpoint if is_file(path) else None
        )
        self._pending_weights_only: bool | None = weights_only
        return checkpoint

    @override
    def load_metadata(
        self,
        checkpoint_path: CheckpointPath,
        weights_only: bool | None
    ) -> CheckpointDict:
        path: str = self._synchronize_path(checkpoint_path)
        checkpoint: CheckpointDict = self._checkpoint_io.load_metadata(
            path,
            weights_only
        )
        self._preloaded_path: str | None = path
        self._preloaded_checkpoint: CheckpointDict | None = checkpoint
        return checkpoint

    @override
    def is_checkpoint_path(self, checkpoint_path: CheckpointPath) -> bool:
        return self._checkpoint_io.is_checkpoint(checkpoint_path)

    @override
    def model_state_dict(self) -> ModelState:
        return self._checkpoint_io.model_state(self._require_model())

    @override
    def optimizer_state(
        self,
        optimizer: torch.optim.Optimizer
    ) -> OptimizerState:
        return self._checkpoint_io.optimizer_state(
            self._require_model(),
            optimizer
        )

    @override
    def restore(
        self,
        checkpoint: Mapping[str, CheckpointValue],
        strict: bool,
        restore_optimizers: bool
    ) -> None:
        del checkpoint
        path: str | None = self._pending_path
        if path is None:
            raise RuntimeError("Model-parallel checkpoint restore has no loaded path.")
        try:
            self._checkpoint_io.load(
                path,
                self._require_model(),
                self._optimizers if restore_optimizers else [],
                strict,
                self._pending_weights_only,
                self._pending_checkpoint
            )
        finally:
            self._pending_path: str | None = None
            self._pending_checkpoint: CheckpointDict | None = None
            self._pending_weights_only: bool | None = None

    @override
    def remove(self, filepath: CheckpointPath, global_rank: int) -> None:
        path: str = self._synchronize_path(filepath)
        if global_rank == 0:
            self._checkpoint_io.remove(path)

    @override
    def teardown(self) -> None:
        self._preloaded_path: str | None = None
        self._preloaded_checkpoint: CheckpointDict | None = None
        self._pending_path: str | None = None
        self._pending_checkpoint: CheckpointDict | None = None
        self._pending_weights_only: bool | None = None

    @override
    def attach_checkpoint_io(self, checkpoint_io: CheckpointIO) -> None:
        del checkpoint_io
        raise MisconfigurationError(
            "ModelParallelStrategy owns its checkpoint implementation."
        )

    @property
    @override
    def configured_checkpoint_io(self) -> CheckpointIO | None:
        return None

    def _synchronize_path(self, checkpoint_path: CheckpointPath) -> str:
        path: str | None = self._broadcast(normalize_path(checkpoint_path), 0)
        if not isinstance(path, str):
            raise TypeError("Checkpoint path broadcast must return a string.")
        return path

    def _require_model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("Checkpoint state requires a connected model.")
        return self._model
