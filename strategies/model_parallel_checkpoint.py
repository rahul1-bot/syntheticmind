from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
import torch.distributed.checkpoint as distributed_checkpoint
from torch import nn
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    StorageReader,
    StorageWriter
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict
)

from syntheticmind.utilities.checkpoint import load_checkpoint, save_checkpoint
from syntheticmind.utilities.filesystem import (
    is_directory,
    is_file,
    is_local_path,
    join_path,
    make_directory,
    normalize_path,
    remove_path
)
from syntheticmind.utilities.types import CheckpointDict

__all__: list[str] = []

_METADATA_FILENAME: str = "meta.pt"


def is_sharded_checkpoint(path: str | Path) -> bool:
    return is_directory(path) and is_file(join_path(path, _METADATA_FILENAME))


def is_model_parallel_checkpoint(path: str | Path) -> bool:
    return is_file(path) or is_sharded_checkpoint(path)


def load_model_parallel_metadata(
    path: str | Path,
    weights_only: bool | None
) -> CheckpointDict:
    if is_sharded_checkpoint(path):
        return load_checkpoint(
            join_path(path, _METADATA_FILENAME),
            weights_only=weights_only
        )
    if is_file(path):
        return load_checkpoint(path, weights_only=weights_only)
    raise ValueError(
        f"The path {str(path)!r} does not point to a full or sharded checkpoint."
    )


def save_model_parallel_checkpoint(
    checkpoint: CheckpointDict,
    path: str | Path,
    save_distributed_checkpoint: bool,
    is_global_zero: bool,
    barrier: Callable[[], None]
) -> None:
    if save_distributed_checkpoint:
        if is_global_zero:
            remove_path(path, recursive=True)
            make_directory(path)
        barrier()
        distributed_state, metadata = _split_checkpoint(checkpoint)
        _save_distributed_state(distributed_state, path)
        if is_global_zero:
            save_checkpoint(
                cast(CheckpointDict, metadata),
                join_path(path, _METADATA_FILENAME)
            )
        barrier()
        return
    if is_directory(path) and not is_sharded_checkpoint(path):
        raise IsADirectoryError(f"The checkpoint path exists and is a directory: {path}")
    if is_sharded_checkpoint(path):
        if is_global_zero:
            remove_path(path, recursive=True)
        barrier()
    if is_global_zero:
        save_checkpoint(checkpoint, path)


def load_model_parallel_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizers: list[torch.optim.Optimizer],
    strict: bool,
    weights_only: bool | None,
    preloaded_checkpoint: CheckpointDict | None = None
) -> CheckpointDict:
    if is_sharded_checkpoint(path):
        state: dict[str, object] = {
            "state_dict": get_model_state_dict(model)
        }
        for index, optimizer in enumerate(optimizers):
            state[f"optimizer_{index}"] = get_optimizer_state_dict(
                model,
                optimizer
            )
        _load_distributed_state(state, path)
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
        return load_model_parallel_metadata(path, weights_only)
    if not is_file(path):
        raise ValueError(
            f"The path {str(path)!r} does not point to a full or sharded checkpoint."
        )
    checkpoint: CheckpointDict = (
        load_checkpoint(path, weights_only=weights_only)
        if preloaded_checkpoint is None
        else preloaded_checkpoint
    )
    distributed_state, _ = _split_checkpoint(checkpoint)
    options = StateDictOptions(full_state_dict=True, strict=strict)
    set_model_state_dict(
        model,
        cast(dict[str, object], distributed_state["state_dict"]),
        options=options
    )
    for index, optimizer in enumerate(optimizers):
        optimizer_state: object | None = distributed_state.get(
            f"optimizer_{index}"
        )
        if not isinstance(optimizer_state, dict):
            continue
        set_optimizer_state_dict(
            model,
            optimizer,
            _rekey_optimizer_state(optimizer_state, model),
            options=options
        )
    returned_checkpoint: CheckpointDict = dict(checkpoint)
    returned_checkpoint.pop("state_dict", None)
    returned_checkpoint.pop("model_state_dict", None)
    return returned_checkpoint


def model_parallel_model_state(
    model: nn.Module,
    full_state_dict: bool
) -> dict[str, object]:
    options: StateDictOptions = StateDictOptions(
        full_state_dict=full_state_dict,
        cpu_offload=True
    )
    return cast(dict[str, object], get_model_state_dict(model, options=options))


def model_parallel_optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    full_state_dict: bool
) -> dict[str, object]:
    options: StateDictOptions = StateDictOptions(
        full_state_dict=full_state_dict,
        cpu_offload=True
    )
    return cast(
        dict[str, object],
        get_optimizer_state_dict(model, optimizer, options=options)
    )


def _split_checkpoint(
    checkpoint: CheckpointDict
) -> tuple[dict[str, object], dict[str, object]]:
    payload: dict[str, object] = dict(checkpoint)
    model_state: object | None = payload.pop("state_dict", None)
    model_state_alias: object | None = payload.pop("model_state_dict", None)
    if not isinstance(model_state, dict):
        model_state = model_state_alias
    if not isinstance(model_state, dict):
        raise KeyError("Checkpoint is missing a model `state_dict`.")
    optimizer_states: object | None = payload.pop("optimizer_states", None)
    optimizer_state_aliases: object | None = payload.pop(
        "optimizer_state_dicts",
        None
    )
    optimizer_state: object | None = payload.pop("optimizer_state_dict", None)
    if not isinstance(optimizer_states, list):
        optimizer_states = optimizer_state_aliases
    if not isinstance(optimizer_states, list):
        optimizer_states = [optimizer_state] if isinstance(optimizer_state, dict) else []
    distributed_state: dict[str, object] = {"state_dict": model_state}
    for index, state in enumerate(optimizer_states):
        if not isinstance(state, dict):
            raise TypeError("Optimizer checkpoint state must be a dictionary.")
        distributed_state[f"optimizer_{index}"] = state
    return distributed_state, payload


def _rekey_optimizer_state(
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


def _save_distributed_state(
    state: dict[str, object],
    path: str | Path
) -> None:
    writer: StorageWriter
    if is_local_path(path):
        writer = FileSystemWriter(normalize_path(path))
    else:
        from torch.distributed.checkpoint._fsspec_filesystem import FsspecWriter

        writer = FsspecWriter(path)
    distributed_checkpoint.save(state, storage_writer=writer)


def _load_distributed_state(
    state: dict[str, object],
    path: str | Path
) -> None:
    reader: StorageReader
    if is_local_path(path):
        reader = FileSystemReader(normalize_path(path))
    else:
        from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader

        reader = FsspecReader(path)
    distributed_checkpoint.load(state, storage_reader=reader)
