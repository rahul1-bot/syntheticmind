# This module:
# 1. Converts complete Trainer checkpoints into model-parallel distributed model
#    and optimizer state plus ordinary trajectory metadata
# 2. Persists and restores full single-file or sharded distributed formats
# 3. Reconciles PyTorch optimizer key formats for composable FSDP2 restoration
#
# Design decisions:
# - Full and sharded formats share one semantic owner because both split the same
#   trajectory and use the same PyTorch distributed state-dict APIs
# - Full state dictionaries use CPU offload and rank-zero file publication;
#   sharded state initializes against live model and optimizer structures on every
#   rank before torch.distributed.checkpoint loads partitions
# - Trainer metadata remains an ordinary CheckpointDict independent from model
#   shards so callbacks, progress, RNG, and configuration restore uniformly
# - A path cannot silently switch between full-file and sharded-directory formats;
#   collisions fail before deleting or replacing the existing artifact
# - Resumable restore validates one distributed optimizer state per configured
#   optimizer before applying model state
# - Integer-keyed optimizer state from a full checkpoint is rekeyed by parameter
#   name when FSDP2 requires name-based mapping
#
# Author: Rahul Sawhney

from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict
)
from torch.distributed.fsdp import FullyShardedDataParallel, OptimStateKeyType

from syntheticmind.plugins.checkpoint_io import TorchCheckpointIO
from syntheticmind.strategies.model_parallel_checkpointing.generation import CheckpointBroadcast
from syntheticmind.strategies.model_parallel_checkpointing.sharded_store import (
    _ShardedCheckpointStore
)
from syntheticmind.storage.filesystem import StoragePath
from syntheticmind.primitives.types import (
    CheckpointDict,
    CheckpointValue,
    DistributedCheckpointState,
    ModelState,
    OptimizerState,
    OptimizerStateKey,
    OptimizerStateValue
)

__all__: list[str] = []

type ModelParallelCheckpointFormat = Literal["full", "sharded"]


class _ModelParallelCheckpointIO:
    # Internal format owner over PyTorch distributed state-dict APIs, ordinary
    # TorchCheckpointIO, and atomic sharded generations.
    def __init__(self, checkpoint_format: ModelParallelCheckpointFormat) -> None:
        # Binds one output format while retaining format detection for loading
        # existing full or sharded artifacts.
        self._checkpoint_format: ModelParallelCheckpointFormat = checkpoint_format
        self._checkpoint_io: TorchCheckpointIO = TorchCheckpointIO()
        self._sharded_store: _ShardedCheckpointStore = _ShardedCheckpointStore()

    def is_checkpoint(self, path: str | Path) -> bool:
        # Recognizes ordinary full files or complete pointer-backed and legacy
        # sharded layouts.
        return StoragePath(path).is_file() or self._sharded_store.is_checkpoint(path)

    def load_metadata(
        self,
        path: str | Path,
        weights_only: bool | None
    ) -> CheckpointDict:
        # Detects sharded or full format from storage and loads trajectory metadata
        # without applying live model state.
        if self._sharded_store.is_checkpoint(path):
            return self._sharded_store.load_metadata(path, weights_only)
        if StoragePath(path).is_file():
            return self._checkpoint_io.load_checkpoint(path, weights_only=weights_only)
        raise ValueError(
            f"The path {str(path)!r} does not point to a full or sharded checkpoint."
        )

    def save(
        self,
        checkpoint: CheckpointDict,
        path: str | Path,
        global_rank: int,
        barrier: Callable[[], None],
        broadcast: CheckpointBroadcast
    ) -> None:
        # Selects sharded all-rank persistence or rank-zero full publication from
        # the configured format.
        if self._checkpoint_format == "sharded":
            self._save_sharded(
                checkpoint,
                path,
                global_rank,
                barrier,
                broadcast
            )
            return
        self._save_full(checkpoint, path, global_rank)

    def load(
        self,
        path: str | Path,
        model: nn.Module,
        optimizers: list[torch.optim.Optimizer],
        strict: bool,
        weights_only: bool | None,
        preloaded_checkpoint: CheckpointDict | None = None
    ) -> CheckpointDict:
        # Restores detected distributed state into live participants and returns
        # the ordinary metadata consumed by Trainer.
        if self._sharded_store.is_checkpoint(path):
            return self._load_sharded(
                path,
                model,
                optimizers,
                strict,
                weights_only
            )
        if not StoragePath(path).is_file():
            raise ValueError(
                f"The path {str(path)!r} does not point to a full or sharded checkpoint."
            )
        checkpoint: CheckpointDict = (
            self._checkpoint_io.load_checkpoint(path, weights_only=weights_only)
            if preloaded_checkpoint is None
            else preloaded_checkpoint
        )
        split_state: tuple[DistributedCheckpointState, CheckpointDict] = (
            self._split_checkpoint(checkpoint)
        )
        distributed_state: DistributedCheckpointState = split_state[0]
        metadata: CheckpointDict = split_state[1]
        self._restore_full_state(distributed_state, model, optimizers, strict)
        return metadata

    def model_state(self, model: nn.Module) -> ModelState:
        # Extracts model state under full or sharded options matching the selected
        # save format.
        options: StateDictOptions = self._state_dict_options()
        return cast(ModelState, get_model_state_dict(model, options=options))

    def optimizer_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> OptimizerState:
        # Extracts optimizer state through the model-aware distributed API.
        options: StateDictOptions = self._state_dict_options()
        return cast(
            OptimizerState,
            get_optimizer_state_dict(model, optimizer, options=options)
        )

    def remove(self, path: str | Path) -> None:
        # Removes the complete full file or sharded root selected by retention.
        StoragePath(path).remove(recursive=True)

    def _save_sharded(
        self,
        checkpoint: CheckpointDict,
        path: str | Path,
        global_rank: int,
        barrier: Callable[[], None],
        broadcast: CheckpointBroadcast
    ) -> None:
        # Splits distributed participant state from metadata and commits both
        # through the sharded generation store.
        split_state: tuple[DistributedCheckpointState, CheckpointDict] = (
            self._split_checkpoint(checkpoint)
        )
        distributed_state: DistributedCheckpointState = split_state[0]
        metadata: CheckpointDict = split_state[1]
        self._sharded_store.save(
            distributed_state,
            metadata,
            path,
            global_rank,
            barrier,
            broadcast
        )

    def _save_full(
        self,
        checkpoint: CheckpointDict,
        path: str | Path,
        global_rank: int
    ) -> None:
        # Rejects directory or sharded collisions before rank-zero single-file
        # publication.
        if StoragePath(path).is_directory():
            if self._sharded_store.is_checkpoint(path):
                raise FileExistsError(
                    "A sharded checkpoint already owns this path. Save the full "
                    f"checkpoint to a different path: {path}"
                )
            raise IsADirectoryError(f"The checkpoint path exists as a directory: {path}")
        if global_rank == 0:
            self._checkpoint_io.save_checkpoint(checkpoint, path)

    def _load_sharded(
        self,
        path: str | Path,
        model: nn.Module,
        optimizers: list[torch.optim.Optimizer],
        strict: bool,
        weights_only: bool | None
    ) -> CheckpointDict:
        # Creates state placeholders from live participants, loads rank shards,
        # applies strict state, then returns separately stored Trainer metadata.
        state: DistributedCheckpointState = {
            "state_dict": cast(ModelState, get_model_state_dict(model))
        }
        index: int
        optimizer: torch.optim.Optimizer
        for index, optimizer in enumerate(optimizers):
            state[f"optimizer_{index}"]: ModelState | OptimizerState = cast(
                OptimizerState,
                get_optimizer_state_dict(model, optimizer)
            )
        self._sharded_store.load_state(state, path)
        options: StateDictOptions = StateDictOptions(strict=strict)
        set_model_state_dict(model, cast(ModelState, state["state_dict"]), options=options)
        index: int
        optimizer: torch.optim.Optimizer
        for index, optimizer in enumerate(optimizers):
            set_optimizer_state_dict(
                model,
                optimizer,
                cast(OptimizerState, state[f"optimizer_{index}"]),
                options=options
            )
        return self.load_metadata(path, weights_only)

    def _restore_full_state(
        self,
        state: DistributedCheckpointState,
        model: nn.Module,
        optimizers: list[torch.optim.Optimizer],
        strict: bool
    ) -> None:
        # Validates complete optimizer cardinality, applies full model state, and
        # restores each optimizer through PyTorch's model-aware API.
        options: StateDictOptions = StateDictOptions(
            full_state_dict=True,
            strict=strict
        )
        available_optimizer_states: list[OptimizerState] = []
        state_name: str
        state_value: ModelState | OptimizerState
        for state_name, state_value in sorted(state.items()):
            if state_name.startswith("optimizer_"):
                available_optimizer_states.append(cast(OptimizerState, state_value))
        if optimizers and len(available_optimizer_states) != len(optimizers):
            raise KeyError(
                "A resumable model-parallel checkpoint must contain exactly one "
                "state entry for every configured optimizer."
            )
        set_model_state_dict(model, cast(ModelState, state["state_dict"]), options=options)
        optimizer: torch.optim.Optimizer
        optimizer_state: OptimizerState
        for optimizer, optimizer_state in zip(
            optimizers,
            available_optimizer_states,
            strict=True
        ):
            set_optimizer_state_dict(
                model,
                optimizer,
                self._rekey_optimizer_state(optimizer_state, model),
                options=options
            )

    def _split_checkpoint(
        self,
        checkpoint: CheckpointDict
    ) -> tuple[DistributedCheckpointState, CheckpointDict]:
        # Removes model and optimizer entries from an owned metadata copy and
        # assigns stable distributed state names without mutating the caller.
        metadata: CheckpointDict = dict(checkpoint)
        model_state: CheckpointValue | None = metadata.pop("state_dict", None)
        if not isinstance(model_state, dict):
            raise KeyError("Checkpoint is missing a model `state_dict`.")
        optimizer_states: CheckpointValue | None = metadata.pop(
            "optimizer_states",
            None
        )
        if not isinstance(optimizer_states, list):
            optimizer_states: CheckpointValue | None = []
        distributed_state: DistributedCheckpointState = {
            "state_dict": cast(ModelState, model_state)
        }
        index: int
        serialized_optimizer_state: CheckpointValue
        for index, serialized_optimizer_state in enumerate(optimizer_states):
            if not isinstance(serialized_optimizer_state, dict):
                raise TypeError("Optimizer checkpoint state must be a dictionary.")
            distributed_state[f"optimizer_{index}"]: ModelState | OptimizerState = cast(
                OptimizerState,
                serialized_optimizer_state
            )
        return distributed_state, metadata

    def _rekey_optimizer_state(
        self,
        optimizer_state: OptimizerState,
        model: nn.Module
    ) -> OptimizerState:
        # Converts integer parameter IDs to parameter names only when the incoming
        # state still uses the ordinary optimizer representation.
        state: OptimizerStateValue | None = optimizer_state.get("state")
        if not isinstance(state, dict) or not state:
            return optimizer_state
        state_key: OptimizerStateKey = next(iter(state))
        if not isinstance(state_key, int):
            return optimizer_state
        return cast(
            OptimizerState,
            FullyShardedDataParallel.rekey_optim_state_dict(
                optimizer_state,
                OptimStateKeyType.PARAM_NAME,
                model
            )
        )

    def _state_dict_options(self) -> StateDictOptions:
        # Selects full aggregation or sharded extraction with CPU offload.
        return StateDictOptions(
            full_state_dict=self._checkpoint_format == "full",
            cpu_offload=True
        )
