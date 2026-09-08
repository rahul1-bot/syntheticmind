from collections.abc import Callable
from concurrent.futures import Future
from functools import partial
from pathlib import Path
from typing import cast, override

import torch.distributed.checkpoint as distributed_checkpoint
from torch.distributed.checkpoint import StorageWriter
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.state_dict_saver import AsyncSaveResponse

from syntheticmind.strategies.model_parallel_checkpointing.generation import (
    CheckpointBroadcast
)
from syntheticmind.strategies.model_parallel_checkpointing.sharded_store import (
    _ShardedCheckpointStore
)
from syntheticmind.primitives.exceptions import CheckpointError
from syntheticmind.storage.filesystem import StoragePath
from syntheticmind.primitives.types import CheckpointDict, DistributedCheckpointState

__all__: list[str] = []

type _UploadFuture = Future[Metadata] | Future[None]


class _AsyncShardedCheckpointStore(_ShardedCheckpointStore):
    def __init__(self) -> None:
        super().__init__()
        self._pending_upload: _UploadFuture | None = None
        self._pending_checkpoint_path: str | None = None
        self._pending_generation_name: str | None = None
        self._pending_global_rank: int | None = None
        self._pending_barrier: Callable[[], None] | None = None
        self._pending_broadcast: CheckpointBroadcast | None = None
        self._pending_publication: bool = False

    @override
    def is_checkpoint(self, path: str | Path) -> bool:
        self._finish_pending()
        return super().is_checkpoint(path)

    @override
    def load_metadata(
        self,
        path: str | Path,
        weights_only: bool | None
    ) -> CheckpointDict:
        self._finish_pending()
        return super().load_metadata(path, weights_only)

    @override
    def save(
        self,
        state: DistributedCheckpointState,
        metadata: CheckpointDict,
        path: str | Path,
        global_rank: int,
        barrier: Callable[[], None],
        broadcast: CheckpointBroadcast
    ) -> None:
        self._finish_pending()
        checkpoint_path: str = StoragePath(path).normalized
        generation_name: str = self._generation_index.create_generation_id(
            global_rank,
            broadcast
        )
        generation_path: str = self._generation_path(
            checkpoint_path,
            generation_name
        )
        preparation: Callable[[], None] = partial(
            self._prepare_generation,
            checkpoint_path,
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
        self._start_upload(state, writer)
        self._pending_checkpoint_path: str | None = checkpoint_path
        self._pending_generation_name: str | None = generation_name
        self._pending_global_rank: int | None = global_rank
        self._pending_barrier: Callable[[], None] | None = barrier
        self._pending_broadcast: CheckpointBroadcast | None = broadcast
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
        self._pending_publication: bool = True

    @override
    def load_state(
        self,
        state: DistributedCheckpointState,
        path: str | Path
    ) -> None:
        self._finish_pending()
        super().load_state(state, path)

    @override
    def remove(self, path: str | Path) -> None:
        self._finish_pending()
        super().remove(path)

    def teardown(self) -> None:
        self._finish_pending()

    def _start_upload(
        self,
        state: DistributedCheckpointState,
        writer: StorageWriter
    ) -> None:
        try:
            save_response: Future[Metadata] | AsyncSaveResponse = cast(
                Future[Metadata] | AsyncSaveResponse,
                distributed_checkpoint.async_save(state, storage_writer=writer)
            )
            if isinstance(save_response, AsyncSaveResponse):
                save_response.staging_completion.result()
                self._pending_upload: _UploadFuture | None = (
                    save_response.upload_completion
                )
                return
            self._pending_upload: _UploadFuture | None = save_response
        except BaseException as exception:
            raise CheckpointError(
                "Failed to stage asynchronous distributed checkpoint state."
            ) from exception

    def _finish_pending(self) -> None:
        pending_upload: _UploadFuture | None = self._pending_upload
        if pending_upload is None:
            return
        try:
            pending_upload.result()
            if self._pending_publication:
                self._publish_pending()
        except BaseException as exception:
            if isinstance(exception, CheckpointError):
                raise
            raise CheckpointError(
                "Asynchronous distributed checkpoint upload failed."
            ) from exception
        finally:
            self._clear_pending()

    def _publish_pending(self) -> None:
        checkpoint_path: str | None = self._pending_checkpoint_path
        generation_name: str | None = self._pending_generation_name
        global_rank: int | None = self._pending_global_rank
        barrier: Callable[[], None] | None = self._pending_barrier
        broadcast: CheckpointBroadcast | None = self._pending_broadcast
        if (
            checkpoint_path is None
            or generation_name is None
            or global_rank is None
            or barrier is None
            or broadcast is None
        ):
            raise CheckpointError(
                "Asynchronous checkpoint publication state is incomplete."
            )
        publication: Callable[[], None] = partial(
            self._publish_generation,
            checkpoint_path,
            generation_name
        )
        self._run_rank_zero_operation(
            publication,
            global_rank,
            broadcast,
            "publish checkpoint generation"
        )
        if global_rank == 0:
            self._prune_generations(checkpoint_path)
        barrier()

    def _clear_pending(self) -> None:
        self._pending_upload: _UploadFuture | None = None
        self._pending_checkpoint_path: str | None = None
        self._pending_generation_name: str | None = None
        self._pending_global_rank: int | None = None
        self._pending_barrier: Callable[[], None] | None = None
        self._pending_broadcast: CheckpointBroadcast | None = None
        self._pending_publication: bool = False
