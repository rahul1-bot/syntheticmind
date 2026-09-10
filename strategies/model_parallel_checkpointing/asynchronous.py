# This module:
# 1. Owns asynchronous staging, distributed writing, and generation publication
# 2. Keeps background checkpoint collectives separate from training collectives
#
# Design decisions:
# - BlockingAsyncStager makes an independent CPU snapshot before save returns
# - One worker writes DCP shards and publishes their generation in one transaction
# - A dedicated Gloo group owns background consensus, including failure reporting
# - Readers, removal, subsequent saves, and teardown fence the publication future
# - A failed future remains observable until teardown; it cannot become success
#
# Author: Rahul Sawhney

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import cast, override

import torch.distributed as dist
import torch.distributed.checkpoint as distributed_checkpoint
from torch.distributed.checkpoint import StorageWriter
from torch.distributed.checkpoint.staging import BlockingAsyncStager

from syntheticmind.primitives.exceptions import CheckpointError
from syntheticmind.primitives.types import CheckpointDict, DistributedCheckpointState
from syntheticmind.storage.filesystem import StoragePath
from syntheticmind.strategies.model_parallel_checkpointing.generation import CheckpointBroadcast
from syntheticmind.strategies.model_parallel_checkpointing.sharded_store import _ShardedCheckpointStore

__all__: list[str] = []


class _AsyncShardedCheckpointStore(_ShardedCheckpointStore):
    # One stage-scoped writer owns its CPU snapshot, worker, communication group,
    # and publication future. Training never uses the writer's process group.
    @override
    def __init__(self) -> None:
        super().__init__()
        self._pending: Future[None] | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._group: dist.ProcessGroup | None = None
        self._stager: BlockingAsyncStager = BlockingAsyncStager()

    @override
    def is_checkpoint(self, path: str | Path) -> bool:
        self._finish_pending()
        return super().is_checkpoint(path)

    @override
    def load_metadata(self, path: str | Path, weights_only: bool | None) -> CheckpointDict:
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
        # Snapshot and metadata preparation complete on the caller. Only shard
        # I/O and commit continue asynchronously; at most one save is outstanding.
        self._finish_pending()
        self._setup_writer()
        checkpoint_path: str = StoragePath(path).normalized
        generation_name: str = self._generation_index.create_generation_id(global_rank, broadcast)
        generation_path: str = self._generation_path(checkpoint_path, generation_name)
        self._run_rank_zero_operation(
            partial(self._prepare_generation, checkpoint_path, generation_path),
            global_rank, broadcast, "prepare checkpoint generation"
        )
        self._run_rank_zero_operation(
            partial(
                self._checkpoint_io.save_checkpoint,
                metadata,
                StoragePath(generation_path).join(self._metadata_filename)
            ),
            global_rank, broadcast, "write checkpoint metadata"
        )
        staged: DistributedCheckpointState | None = None
        writer: StorageWriter | None = None
        local_error: BaseException | None = None
        try:
            staged: DistributedCheckpointState | None = cast(
                DistributedCheckpointState, self._stager.stage(state)
            )
            writer: StorageWriter | None = self._storage_writer(generation_path)
        except BaseException as exception:
            local_error: BaseException | None = exception
        self._agree_failure(local_error, "stage checkpoint")
        if staged is None or writer is None or self._executor is None:
            raise CheckpointError("Checkpoint staging completed without writer state.")
        barrier()
        self._pending: Future[None] | None = self._executor.submit(
            self._write_and_publish,
            staged, writer, checkpoint_path, generation_name, global_rank
        )

    @override
    def load_state(self, state: DistributedCheckpointState, path: str | Path) -> None:
        self._finish_pending()
        super().load_state(state, path)

    @override
    def remove(self, path: str | Path) -> None:
        # Waiting here is local: every rank's worker already participates in the
        # background transaction, so rank-zero-only retention cannot deadlock.
        self._finish_pending()
        super().remove(path)

    @override
    def teardown(self) -> None:
        # Drain before destroying process groups. Cleanup failures never replace
        # the original asynchronous failure and repeated teardown is harmless.
        failure: BaseException | None = None
        try:
            self._finish_pending()
        except BaseException as exception:
            failure: BaseException | None = exception
        operations: list[Callable[[], None]] = [self._close_worker, self._stager.close, self._close_group]
        operation: Callable[[], None]
        for operation in operations:
            try:
                operation()
            except BaseException as exception:
                if failure is None:
                    failure: BaseException | None = exception
                else:
                    failure.add_note(f"Checkpoint cleanup also failed: {exception}")
        self._pending: Future[None] | None = None
        if failure is not None:
            raise failure

    def _setup_writer(self) -> None:
        if self._group is None and dist.is_initialized():
            self._group: dist.ProcessGroup | None = dist.new_group(
                backend="gloo", timeout=timedelta(seconds=60)
            )
        if self._executor is None:
            self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=1)

    def _write_and_publish(
        self,
        state: DistributedCheckpointState,
        writer: StorageWriter,
        checkpoint_path: str,
        generation_name: str,
        global_rank: int
    ) -> None:
        local_error: BaseException | None = None
        try:
            distributed_checkpoint.save(
                state, storage_writer=writer, process_group=self._group
            )
        except BaseException as exception:
            local_error: BaseException | None = exception
        self._agree_failure(local_error, "write checkpoint shards")
        self._run_rank_zero_operation(
            partial(self._publish_generation, checkpoint_path, generation_name),
            global_rank, self._broadcast_writer, "publish checkpoint generation"
        )
        if global_rank == 0:
            self._prune_generations(checkpoint_path)
        if self._group is not None:
            dist.barrier(group=self._group)

    def _agree_failure(self, local_error: BaseException | None, operation: str) -> None:
        message: str | None = None if local_error is None else f"{type(local_error).__name__}: {local_error}"
        messages: list[str | None] = [message]
        if self._group is not None:
            messages: list[str | None] = [None] * dist.get_world_size(self._group)
            dist.all_gather_object(messages, message, group=self._group)
        if any(message is not None for message in messages):
            raise CheckpointError(f"Failed to {operation}: {messages}") from local_error

    def _broadcast_writer(self, value: str | None, source_rank: int) -> str | None:
        if self._group is None:
            return value
        values: list[str | None] = [value]
        dist.broadcast_object_list(values, src=source_rank, group=self._group)
        return values[0]

    def _finish_pending(self) -> None:
        if self._pending is not None:
            self._pending.result()
            self._pending: Future[None] | None = None

    def _close_worker(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor: ThreadPoolExecutor | None = None

    def _close_group(self) -> None:
        if self._group is not None:
            group: dist.ProcessGroup = self._group
            self._group: dist.ProcessGroup | None = None
            dist.destroy_process_group(group)
