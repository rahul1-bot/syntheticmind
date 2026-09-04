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

from syntheticmind.strategies.model_parallel_checkpointing.generation import CheckpointBroadcast
from syntheticmind.strategies.model_parallel_checkpointing.sharded_store import _ShardedCheckpointStore
from syntheticmind.utilities.checkpoint import load_checkpoint, save_checkpoint
from syntheticmind.utilities.filesystem import (
    is_directory,
    is_file,
    remove_path
)
from syntheticmind.utilities.types import CheckpointDict

__all__: list[str] = []

type ModelParallelCheckpointFormat = Literal["full", "sharded"]


class _ModelParallelCheckpointIO:
    def __init__(self, checkpoint_format: ModelParallelCheckpointFormat) -> None:
        self._checkpoint_format: ModelParallelCheckpointFormat = checkpoint_format
        self._sharded_store: _ShardedCheckpointStore = _ShardedCheckpointStore()

    def is_checkpoint(self, path: str | Path) -> bool:
        return is_file(path) or self._sharded_store.is_checkpoint(path)

    def load_metadata(
        self,
        path: str | Path,
        weights_only: bool | None
    ) -> CheckpointDict:
        if self._sharded_store.is_checkpoint(path):
            return self._sharded_store.load_metadata(path, weights_only)
        if is_file(path):
            return load_checkpoint(path, weights_only=weights_only)
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
        if self._sharded_store.is_checkpoint(path):
            return self._load_sharded(
                path,
                model,
                optimizers,
                strict,
                weights_only
            )
        if not is_file(path):
            raise ValueError(
                f"The path {str(path)!r} does not point to a full or sharded checkpoint."
            )
        checkpoint: CheckpointDict = (
            load_checkpoint(path, weights_only=weights_only)
            if preloaded_checkpoint is None
            else preloaded_checkpoint
        )
        distributed_state: dict[str, object]
        metadata: dict[str, object]
        distributed_state, metadata = self._split_checkpoint(checkpoint)
        self._restore_full_state(
            distributed_state,
            model,
            optimizers,
            strict
        )
        return cast(CheckpointDict, metadata)

    def model_state(self, model: nn.Module) -> dict[str, object]:
        options: StateDictOptions = self._state_dict_options()
        return cast(dict[str, object], get_model_state_dict(model, options=options))

    def optimizer_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> dict[str, object]:
        options: StateDictOptions = self._state_dict_options()
        return cast(
            dict[str, object],
            get_optimizer_state_dict(model, optimizer, options=options)
        )

    def remove(self, path: str | Path) -> None:
        remove_path(path, recursive=True)

    def _save_sharded(
        self,
        checkpoint: CheckpointDict,
        path: str | Path,
        global_rank: int,
        barrier: Callable[[], None],
        broadcast: CheckpointBroadcast
    ) -> None:
        distributed_state: dict[str, object]
        metadata: dict[str, object]
        distributed_state, metadata = self._split_checkpoint(checkpoint)
        self._sharded_store.save(
            distributed_state,
            cast(CheckpointDict, metadata),
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
        if is_directory(path):
            if self._sharded_store.is_checkpoint(path):
                raise FileExistsError(
                    "A sharded checkpoint already owns this path. Save the full "
                    f"checkpoint to a different path: {path}"
                )
            raise IsADirectoryError(f"The checkpoint path exists and is a directory: {path}")
        if global_rank == 0:
            save_checkpoint(checkpoint, path)

    def _load_sharded(
        self,
        path: str | Path,
        model: nn.Module,
        optimizers: list[torch.optim.Optimizer],
        strict: bool,
        weights_only: bool | None
    ) -> CheckpointDict:
        state: dict[str, object] = {"state_dict": get_model_state_dict(model)}
        for index, optimizer in enumerate(optimizers):
            state[f"optimizer_{index}"] = get_optimizer_state_dict(model, optimizer)
        self._sharded_store.load_state(state, path)
        options: StateDictOptions = StateDictOptions(strict=strict)
        set_model_state_dict(
            model,
            cast(dict[str, object], state["state_dict"]),
            options=options
        )
        for index, optimizer in enumerate(optimizers):
            set_optimizer_state_dict(
                model,
                optimizer,
                cast(dict[str, object], state[f"optimizer_{index}"]),
                options=options
            )
        return self.load_metadata(path, weights_only)

    def _restore_full_state(
        self,
        state: dict[str, object],
        model: nn.Module,
        optimizers: list[torch.optim.Optimizer],
        strict: bool
    ) -> None:
        options: StateDictOptions = StateDictOptions(
            full_state_dict=True,
            strict=strict
        )
        available_optimizer_states: list[dict[str, object]] = [
            cast(dict[str, object], value)
            for key, value in sorted(state.items())
            if key.startswith("optimizer_") and isinstance(value, dict)
        ]
        if optimizers and len(available_optimizer_states) != len(optimizers):
            raise KeyError(
                "A resumable model-parallel checkpoint must contain exactly one "
                "state entry for every configured optimizer."
            )
        set_model_state_dict(
            model,
            cast(dict[str, object], state["state_dict"]),
            options=options
        )
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
    ) -> tuple[dict[str, object], dict[str, object]]:
        metadata: dict[str, object] = dict(checkpoint)
        model_state: object | None = metadata.pop("state_dict", None)
        model_state_alias: object | None = metadata.pop("model_state_dict", None)
        if not isinstance(model_state, dict):
            model_state: object | None = model_state_alias
        if not isinstance(model_state, dict):
            raise KeyError("Checkpoint is missing a model `state_dict`.")
        optimizer_states: object | None = metadata.pop("optimizer_states", None)
        optimizer_state_aliases: object | None = metadata.pop(
            "optimizer_state_dicts",
            None
        )
        optimizer_state: object | None = metadata.pop("optimizer_state_dict", None)
        if not isinstance(optimizer_states, list):
            optimizer_states: object | None = optimizer_state_aliases
        if not isinstance(optimizer_states, list):
            optimizer_states: object | None = (
                [optimizer_state] if isinstance(optimizer_state, dict) else []
            )
        distributed_state: dict[str, object] = {"state_dict": model_state}
        for index, state in enumerate(optimizer_states):
            if not isinstance(state, dict):
                raise TypeError("Optimizer checkpoint state must be a dictionary.")
            distributed_state[f"optimizer_{index}"] = state
        return distributed_state, metadata

    def _rekey_optimizer_state(
        self,
        optimizer_state: dict[str, object],
        model: nn.Module
    ) -> dict[str, object]:
        state: object | None = optimizer_state.get("state")
        if not isinstance(state, dict) or not state:
            return optimizer_state
        if not isinstance(next(iter(state)), int):
            return optimizer_state
        from torch.distributed.fsdp import FullyShardedDataParallel, OptimStateKeyType

        return cast(
            dict[str, object],
            FullyShardedDataParallel.rekey_optim_state_dict(
                optimizer_state,
                OptimStateKeyType.PARAM_NAME,
                model
            )
        )

    def _state_dict_options(self) -> StateDictOptions:
        return StateDictOptions(
            full_state_dict=self._checkpoint_format == "full",
            cpu_offload=True
        )
