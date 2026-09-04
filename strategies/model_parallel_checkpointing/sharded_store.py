import warnings
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import cast

import torch.distributed.checkpoint as distributed_checkpoint
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint import StorageReader, StorageWriter
from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader, FsspecWriter

from syntheticmind.plugins.checkpoint_io import TorchCheckpointIO
from syntheticmind.strategies.model_parallel_checkpointing.generation import (
    CheckpointBroadcast,
    _CheckpointGenerationIndex,
    _CheckpointGenerationPointer
)
from syntheticmind.exceptions import CheckpointError
from syntheticmind.storage.filesystem import StoragePath
from syntheticmind.types import CheckpointDict, DistributedCheckpointState

__all__: list[str] = []

class _ShardedCheckpointStore:
    def __init__(self) -> None:
        self._checkpoint_io: TorchCheckpointIO = TorchCheckpointIO()
        self._distributed_metadata_filename: str = ".metadata"
        self._generation_index: _CheckpointGenerationIndex = (
            _CheckpointGenerationIndex()
        )
        self._generations_directory_name: str = "generations"
        self._metadata_filename: str = "meta.pt"

    def is_checkpoint(self, path: str | Path) -> bool:
        if not StoragePath(path).is_directory():
            return False
        return self._generation_index.exists(path) or self._is_complete_generation(path)

    def load_metadata(
        self,
        path: str | Path,
        weights_only: bool | None
    ) -> CheckpointDict:
        generation_path: str = self._active_generation_path(path)
        return self._checkpoint_io.load_checkpoint(
            StoragePath(generation_path).join(self._metadata_filename),
            weights_only=weights_only
        )

    def save(
        self,
        state: DistributedCheckpointState,
        metadata: CheckpointDict,
        path: str | Path,
        global_rank: int,
        barrier: Callable[[], None],
        broadcast: CheckpointBroadcast
    ) -> None:
        generation_name: str = self._generation_index.create_generation_id(
            global_rank,
            broadcast
        )
        generation_path: str = self._generation_path(path, generation_name)
        preparation: Callable[[], None] = partial(
            self._prepare_generation,
            path,
            generation_path
        )
        self._run_rank_zero_operation(
            preparation,
            global_rank,
            broadcast,
            "prepare checkpoint generation"
        )
        barrier()
        writer: StorageWriter = self._storage_writer(generation_path)
        try:
            distributed_checkpoint.save(state, storage_writer=writer)
        except BaseException as exception:
            raise CheckpointError(
                f"Failed to write distributed checkpoint generation: {generation_path}"
            ) from exception
        metadata_write: Callable[[], None] = partial(
            self._checkpoint_io.save_checkpoint,
            metadata,
            StoragePath(generation_path).join(self._metadata_filename)
        )
        self._run_rank_zero_operation(
            metadata_write,
            global_rank,
            broadcast,
            "write checkpoint metadata"
        )
        publication: Callable[[], None] = partial(
            self._publish_generation,
            path,
            generation_name
        )
        self._run_rank_zero_operation(
            publication,
            global_rank,
            broadcast,
            "publish checkpoint generation"
        )
        if global_rank == 0:
            self._prune_generations(path)
        barrier()

    def load_state(self, state: DistributedCheckpointState, path: str | Path) -> None:
        generation_path: str = self._active_generation_path(path)
        reader: StorageReader = self._storage_reader(generation_path)
        distributed_checkpoint.load(state, storage_reader=reader)

    def remove(self, path: str | Path) -> None:
        StoragePath(path).remove(recursive=True)

    def _prepare_generation(
        self,
        checkpoint_path: str | Path,
        generation_path: str | Path
    ) -> None:
        if StoragePath(checkpoint_path).is_file():
            raise FileExistsError(
                f"A full checkpoint file already exists at {checkpoint_path}."
            )
        StoragePath(generation_path).make_directory()

    def _publish_generation(
        self,
        checkpoint_path: str | Path,
        generation_name: str
    ) -> None:
        generation_path: str = self._generation_path(
            checkpoint_path,
            generation_name
        )
        if not self._is_complete_generation(generation_path):
            raise CheckpointError(
                f"Checkpoint generation is incomplete and cannot be published: {generation_path}"
            )
        self._generation_index.publish(checkpoint_path, generation_name)

    def _run_rank_zero_operation(
        self,
        operation: Callable[[], None],
        global_rank: int,
        broadcast: CheckpointBroadcast,
        operation_name: str
    ) -> None:
        local_error: BaseException | None = None
        local_message: str | None = None
        if global_rank == 0:
            try:
                operation()
            except BaseException as exception:
                local_error: BaseException | None = exception
                detail: str = " ".join(str(exception).split())
                local_message: str | None = (
                    f"{type(exception).__name__}: {detail}"
                    if detail
                    else type(exception).__name__
                )
        broadcast_message: str | None = broadcast(local_message, 0)
        if broadcast_message is None:
            return
        if not isinstance(broadcast_message, str):
            raise CheckpointError(
                f"Rank-zero failure broadcast for {operation_name} was not a string."
            )
        error: CheckpointError = CheckpointError(
            f"Failed to {operation_name}: {broadcast_message}"
        )
        if local_error is not None:
            raise error from local_error
        raise error

    def _active_generation_path(self, path: str | Path) -> str:
        pointer: _CheckpointGenerationPointer | None = self._generation_index.read(path)
        if pointer is None:
            if self._is_complete_generation(path):
                return StoragePath(path).normalized
            raise CheckpointError(
                f"Sharded checkpoint has no published generation: {path}"
            )
        generation_name: str = pointer.generation_id
        generation_path: str = self._generation_path(path, generation_name)
        if not self._is_complete_generation(generation_path):
            raise CheckpointError(
                f"Published checkpoint generation is incomplete: {generation_path}"
            )
        return generation_path

    def _generation_path(
        self,
        checkpoint_path: str | Path,
        generation_name: str
    ) -> str:
        return StoragePath(checkpoint_path).join(
            self._generations_directory_name,
            generation_name
        )

    def _is_complete_generation(self, path: str | Path) -> bool:
        distributed_metadata_path: str = StoragePath(path).join(
            self._distributed_metadata_filename
        )
        trainer_metadata_path: str = StoragePath(path).join(
            self._metadata_filename
        )
        return StoragePath(distributed_metadata_path).is_file() and StoragePath(
            trainer_metadata_path
        ).is_file()

    def _prune_generations(
        self,
        checkpoint_path: str | Path
    ) -> None:
        try:
            pointer: _CheckpointGenerationPointer | None = (
                self._generation_index.read(checkpoint_path)
            )
            if pointer is None:
                return
            retained_generation_names: frozenset[str] = pointer.retained_generation_ids
            generations_path: str = StoragePath(checkpoint_path).join(
                self._generations_directory_name
            )
            generation_path: str
            for generation_path in StoragePath(generations_path).list_directory():
                generation_storage_path: StoragePath = StoragePath(generation_path)
                if not generation_storage_path.is_directory():
                    continue
                if generation_storage_path.name in retained_generation_names:
                    continue
                generation_storage_path.remove(recursive=True)
        except BaseException as exception:
            warnings.warn(
                f"Published checkpoint but could not prune stale generations: {exception}",
                RuntimeWarning,
                stacklevel=3
            )

    def _storage_writer(self, path: str | Path) -> StorageWriter:
        storage_path: StoragePath = StoragePath(path)
        if storage_path.is_local:
            return FileSystemWriter(storage_path.normalized)
        return FsspecWriter(path)

    def _storage_reader(self, path: str | Path) -> StorageReader:
        storage_path: StoragePath = StoragePath(path)
        if storage_path.is_local:
            return FileSystemReader(storage_path.normalized)
        return FsspecReader(path)
