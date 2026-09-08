# This module:
# 1. Namespaces partition-local model/optimizer state and preserves rank metadata
# 2. Validates parallel policy before restoring a partitioned training trajectory
#
# Design decisions:
# - Equal local parameter names on different pipeline stages never share a key
# - RNG, callbacks, data, and loop metadata belong to each producing rank
# - Existing immutable generation storage owns synchronous or asynchronous commit
# - Exact continuation requires unchanged world size and parallel configuration
#
# Author: Rahul Sawhney

from collections.abc import Callable
from pathlib import Path
from typing import cast, override

import torch
import torch.distributed as dist
from pydantic import TypeAdapter
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
    set_model_state_dict, set_optimizer_state_dict
)

from syntheticmind.plugins.checkpoint_io import CheckpointPath, CheckpointStorageOptions
from syntheticmind.primitives.types import (
    CheckpointDict, CheckpointValue, DistributedCheckpointState, ModelState, OptimizerState
)
from syntheticmind.strategies.parallel_configuration import _ParallelConfiguration
from syntheticmind.strategies.model_parallel_checkpointing.checkpoint import _ModelParallelCheckpoint
from syntheticmind.strategies.model_parallel_checkpointing.checkpoint_io import (
    ModelParallelCheckpointWrite, _ModelParallelCheckpointIO
)
from syntheticmind.strategies.model_parallel_checkpointing.generation import CheckpointBroadcast

__all__: list[str] = []


class _RankLocalCheckpoint(_ModelParallelCheckpoint):
    # Strategy's normal checkpoint lifecycle with a partition-aware representation
    # and a durability fence supplied by the parallel execution owner.
    @override
    def __init__(
        self,
        configuration: _ParallelConfiguration,
        checkpoint_write: ModelParallelCheckpointWrite,
        barrier: Callable[[], None],
        broadcast: CheckpointBroadcast,
        validate_boundary: Callable[[], None]
    ) -> None:
        super().__init__("sharded", checkpoint_write, barrier, broadcast)
        self._checkpoint_io: _ModelParallelCheckpointIO = _RankLocalCheckpointIO(
            configuration, checkpoint_write
        )
        self._validate_boundary: Callable[[], None] = validate_boundary

    @override
    def model_state_dict(self) -> ModelState:
        self._validate_boundary()
        return super().model_state_dict()

    @override
    def save(
        self,
        checkpoint: CheckpointDict,
        filepath: CheckpointPath,
        storage_options: CheckpointStorageOptions,
        global_rank: int
    ) -> None:
        self._validate_boundary()
        super().save(checkpoint, filepath, storage_options, global_rank)


class _RankLocalCheckpointIO(_ModelParallelCheckpointIO):
    # Complete checkpoint representation for non-interchangeable rank state.
    # All ranks contribute metadata; ordinary state-dict APIs still own tensors.
    @override
    def __init__(
        self,
        configuration: _ParallelConfiguration,
        checkpoint_write: ModelParallelCheckpointWrite
    ) -> None:
        super().__init__("sharded", checkpoint_write)
        self._configuration: _ParallelConfiguration = configuration
        self._configuration_adapter: TypeAdapter[_ParallelConfiguration] = TypeAdapter(
            _ParallelConfiguration
        )

    @override
    def is_checkpoint(self, path: str | Path) -> bool:
        return self._sharded_store.is_checkpoint(path)

    @override
    def load_metadata(self, path: str | Path, weights_only: bool | None) -> CheckpointDict:
        envelope: CheckpointDict = self._sharded_store.load_metadata(path, weights_only)
        if set(envelope) != {"rank_local_schema_version", "parallel_configuration", "rank_metadata"}:
            raise ValueError("Rank-local checkpoint envelope fields are invalid.")
        if type(envelope["rank_local_schema_version"]) is not int or envelope["rank_local_schema_version"] != 1:
            raise ValueError("Unsupported rank-local checkpoint schema.")
        saved: _ParallelConfiguration = self._configuration_adapter.validate_python(
            envelope["parallel_configuration"]
        )
        if saved != self._configuration or dist.get_world_size() != saved.world_size:
            raise ValueError("Parallel checkpoint configuration differs from this trajectory.")
        ranks: CheckpointValue = envelope["rank_metadata"]
        if not isinstance(ranks, list) or len(ranks) != saved.world_size:
            raise ValueError("Checkpoint must contain one metadata record per rank.")
        selected: CheckpointValue = ranks[dist.get_rank()]
        if not isinstance(selected, dict):
            raise TypeError("Rank checkpoint metadata must be a mapping.")
        return selected

    @override
    def load(
        self,
        path: str | Path,
        model: nn.Module,
        optimizers: list[torch.optim.Optimizer],
        strict: bool,
        weights_only: bool | None,
        preloaded_checkpoint: CheckpointDict | None = None
    ) -> CheckpointDict:
        del preloaded_checkpoint
        return self._load_sharded(path, model, optimizers, strict, weights_only)

    @override
    def _save_sharded(
        self,
        checkpoint: CheckpointDict,
        path: str | Path,
        global_rank: int,
        barrier: Callable[[], None],
        broadcast: CheckpointBroadcast
    ) -> None:
        state: DistributedCheckpointState
        metadata: CheckpointDict
        state, metadata = self._split_checkpoint(checkpoint)
        metadata["rank_optimizer_count"]: CheckpointValue = len(state) - 1
        ranks: list[CheckpointDict | None] = [None] * self._configuration.world_size
        dist.all_gather_object(ranks, metadata)
        envelope: CheckpointDict = {
            "rank_local_schema_version": 1,
            "parallel_configuration": cast(
                CheckpointValue, self._configuration.model_dump(mode="python")
            ),
            "rank_metadata": cast(CheckpointValue, ranks)
        }
        self._sharded_store.save(
            state, envelope, path, global_rank, barrier, broadcast
        )

    @override
    def _split_checkpoint(
        self, checkpoint: CheckpointDict
    ) -> tuple[DistributedCheckpointState, CheckpointDict]:
        state: DistributedCheckpointState
        metadata: CheckpointDict
        state, metadata = super()._split_checkpoint(checkpoint)
        key: str
        value: ModelState | OptimizerState
        return {
            f"rank_{dist.get_rank()}_{key}": value for key, value in state.items()
        }, metadata

    @override
    def _load_sharded(
        self,
        path: str | Path,
        model: nn.Module,
        optimizers: list[torch.optim.Optimizer],
        strict: bool,
        weights_only: bool | None
    ) -> CheckpointDict:
        # Validate policy and optimizer cardinality before applying live state.
        metadata: CheckpointDict = self.load_metadata(path, weights_only)
        count: CheckpointValue | None = metadata.get("rank_optimizer_count")
        if type(count) is not int or count < 0:
            raise ValueError("Rank checkpoint optimizer cardinality is invalid.")
        if optimizers and count != len(optimizers):
            raise ValueError("Rank checkpoint optimizer count differs from the configured count.")
        prefix: str = f"rank_{dist.get_rank()}_"
        state: DistributedCheckpointState = {
            f"{prefix}state_dict": cast(ModelState, get_model_state_dict(model))
        }
        index: int
        optimizer: torch.optim.Optimizer
        for index, optimizer in enumerate(optimizers):
            state[f"{prefix}optimizer_{index}"]: ModelState | OptimizerState = cast(
                OptimizerState, get_optimizer_state_dict(model, optimizer)
            )
        self._sharded_store.load_state(state, path)
        options: StateDictOptions = StateDictOptions(strict=strict)
        set_model_state_dict(model, cast(ModelState, state[f"{prefix}state_dict"]), options=options)
        for index, optimizer in enumerate(optimizers):
            set_optimizer_state_dict(
                model, optimizer,
                cast(OptimizerState, state[f"{prefix}optimizer_{index}"]),
                options=options
            )
        return metadata
