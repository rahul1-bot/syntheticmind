# This module:
# 1. Implements StrategyCheckpoint for ModelParallelStrategy and coordinates
#    all-rank path agreement, metadata preload, deferred live-state restoration,
#    and full or sharded format ownership
#
# Design decisions:
# - Every rank broadcasts and uses one normalized path before distributed state
#   operations, preventing divergent filesystem targets across workers
# - Metadata may load before model and optimizers exist, but distributed state is
#   deferred until restore; pending path, payload, and weights-only policy bridge
#   those lifecycle phases without duplicate reads
# - Full-file metadata remains cached because it already contains distributed
#   state, while sharded metadata is separate from rank-partitioned storage
# - ModelParallelStrategy owns its checkpoint representation and rejects an
#   external CheckpointIO plugin that could not preserve all-rank semantics
# - Pending and preload state clear after restore or teardown so sequential stages
#   cannot consume a prior artifact accidentally
#
# Author: Rahul Sawhney

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
from syntheticmind.primitives.exceptions import MisconfigurationError
from syntheticmind.storage.filesystem import StoragePath
from syntheticmind.primitives.types import (
    CheckpointDict,
    CheckpointValue,
    ModelState,
    OptimizerState
)

__all__: list[str] = []


class _ModelParallelCheckpoint(StrategyCheckpoint):
    # Model-parallel state coordinator connecting Trainer checkpoint lifecycle to
    # distributed full or sharded representation.
    def __init__(
        self,
        checkpoint_format: ModelParallelCheckpointFormat,
        barrier: Callable[[], None],
        broadcast: CheckpointBroadcast
    ) -> None:
        # Binds format, collective operations, participant slots, and empty
        # preload and pending-restore state.
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
        # Validates and binds the distributed model used by later state-dict APIs.
        if not isinstance(model, nn.Module):
            raise TypeError("Checkpoint model must be a torch.nn.Module.")
        self._model: nn.Module | None = model

    @override
    def connect_optimizers(self, optimizers: list[torch.optim.Optimizer]) -> None:
        # Validates and copies the ordered distributed optimizer participants.
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
        # Rejects unsupported transport options, synchronizes the path, and
        # delegates all-rank state persistence to the format owner.
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
        # Consumes preloaded metadata or reads it, then records the path and
        # restore policy required when live model and optimizers are ready.
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
            checkpoint if StoragePath(path).is_file() else None
        )
        self._pending_weights_only: bool | None = weights_only
        return checkpoint

    @override
    def load_metadata(
        self,
        checkpoint_path: CheckpointPath,
        weights_only: bool | None
    ) -> CheckpointDict:
        # Reads schema metadata once and caches it for the subsequent load phase.
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
        # Recognizes complete full files or published sharded checkpoint roots.
        return self._checkpoint_io.is_checkpoint(checkpoint_path)

    @override
    def model_state_dict(self) -> ModelState:
        # Extracts model state using the configured full or sharded representation.
        return self._checkpoint_io.model_state(self._require_model())

    @override
    def optimizer_state(
        self,
        optimizer: torch.optim.Optimizer
    ) -> OptimizerState:
        # Extracts model-aware distributed optimizer state for one participant.
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
        # Applies distributed model and optional optimizer state from the pending
        # artifact and clears every pending reference in finally.
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
        # Synchronizes artifact identity across ranks and applies deletion on
        # global zero after caller-owned retention policy selects it.
        path: str = self._synchronize_path(filepath)
        if global_rank == 0:
            self._checkpoint_io.remove(path)

    @override
    def teardown(self) -> None:
        # Releases every preloaded and deferred restore reference while leaving
        # immutable published artifacts untouched.
        self._preloaded_path: str | None = None
        self._preloaded_checkpoint: CheckpointDict | None = None
        self._pending_path: str | None = None
        self._pending_checkpoint: CheckpointDict | None = None
        self._pending_weights_only: bool | None = None

    @override
    def attach_checkpoint_io(self, checkpoint_io: CheckpointIO) -> None:
        # Rejects external transport because model-parallel publication and
        # representation are Strategy-owned collective algorithms.
        del checkpoint_io
        raise MisconfigurationError(
            "ModelParallelStrategy owns its checkpoint implementation."
        )

    @property
    @override
    def configured_checkpoint_io(self) -> CheckpointIO | None:
        # Reports no external transport because this Strategy owns collective
        # representation and publication end to end.
        return None

    def _synchronize_path(self, checkpoint_path: CheckpointPath) -> str:
        # Broadcasts global-rank-zero normalized path and validates the collective
        # result before any rank touches storage.
        path: str | None = self._broadcast(
            StoragePath(checkpoint_path).normalized,
            0
        )
        if not isinstance(path, str):
            raise TypeError("Checkpoint path broadcast must return a string.")
        return path

    def _require_model(self) -> nn.Module:
        # Prevents distributed state extraction before Strategy model connection.
        if self._model is None:
            raise RuntimeError("Checkpoint state requires a connected model.")
        return self._model
