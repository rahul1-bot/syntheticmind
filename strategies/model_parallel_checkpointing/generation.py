# This module:
# 1. Defines the strict versioned pointer that publishes one immutable sharded
#    checkpoint generation and retains its immediate predecessor
# 2. Creates UUIDv7 generation identities, validates pointers, and atomically
#    publishes local or transactional remote pointer updates
#
# Design decisions:
# - A non-empty distributed checkpoint directory cannot be atomically replaced
#   portably, so immutable generation directories hold data and one small pointer
#   is the publication commit point
# - The pointer records active and previous published generations explicitly;
#   selecting the newest directory would allow a complete but unpublished failed
#   generation to displace the last valid recovery point
# - UUIDv7 provides sortable unique generation identity and strict schema
#   constraints reject malformed or future pointer formats
# - Local publication writes, flushes, fsyncs, atomically replaces, and fsyncs the
#   parent directory; remote publication uses the resolved fsspec transaction
# - Failed local pointer writes remove their temporary file and preserve the
#   previously published pointer
#
# Author: Rahul Sawhney

import io
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, ClassVar, Literal
from uuid import uuid7

from fsspec.spec import AbstractFileSystem
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from syntheticmind.primitives.exceptions import CheckpointError
from syntheticmind.storage.filesystem import StoragePath

__all__: list[str] = []

type CheckpointBroadcast = Callable[[str | None, int], str | None]
type CheckpointGenerationId = Annotated[
    str,
    StringConstraints(
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        ascii_only=True
    )
]


class _CheckpointGenerationPointer(BaseModel):
    # Immutable publication record naming the readable generation and the prior
    # published recovery generation retained during pruning.
    schema_version: Literal[1] = 1
    generation_id: CheckpointGenerationId
    previous_generation_id: CheckpointGenerationId | None = None
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        validate_default=True
    )

    @property
    def retained_generation_ids(self) -> frozenset[str]:
        # Unique non-null generation identities protected from pruning.
        retained_generation_ids: set[str] = set()
        generation_id: str | None
        for generation_id in (self.generation_id, self.previous_generation_id):
            if generation_id is not None:
                retained_generation_ids.add(generation_id)
        return frozenset(retained_generation_ids)


class _CheckpointGenerationIndex:
    # Publication-index owner for one checkpoint root.
    def __init__(self) -> None:
        # Fixes the sole pointer identity shared by local and remote backends.
        self._pointer_filename: str = "active_generation"

    def exists(self, checkpoint_path: str | Path) -> bool:
        # Tests for the publication marker rather than guessing from shard files.
        return StoragePath(self._pointer_path(checkpoint_path)).is_file()

    def create_generation_id(
        self,
        global_rank: int,
        broadcast: CheckpointBroadcast
    ) -> str:
        # Creates UUIDv7 on rank zero, broadcasts it, and validates canonical form
        # on every rank before generation paths are constructed.
        local_generation_id: str | None = str(uuid7()) if global_rank == 0 else None
        broadcast_generation_id: str | None = broadcast(local_generation_id, 0)
        if not isinstance(broadcast_generation_id, str):
            raise CheckpointError("Checkpoint generation broadcast did not return a string.")
        pointer: _CheckpointGenerationPointer = _CheckpointGenerationPointer(
            generation_id=broadcast_generation_id
        )
        return pointer.generation_id

    def read(
        self,
        checkpoint_path: str | Path
    ) -> _CheckpointGenerationPointer | None:
        # Reads and strictly validates the current pointer, returning None for a
        # legacy flat sharded checkpoint with no pointer.
        pointer_path: str = self._pointer_path(checkpoint_path)
        if not StoragePath(pointer_path).is_file():
            return None
        with StoragePath(pointer_path).open_text("r") as pointer_file:
            pointer_payload: str = pointer_file.read()
        try:
            return _CheckpointGenerationPointer.model_validate_json(pointer_payload)
        except ValidationError as exception:
            raise CheckpointError(
                f"Checkpoint generation pointer is invalid: {pointer_path}"
            ) from exception

    def publish(
        self,
        checkpoint_path: str | Path,
        generation_id: CheckpointGenerationId
    ) -> _CheckpointGenerationPointer:
        # Builds the next pointer from current active identity and commits it
        # through the backend-specific atomic write boundary.
        previous_pointer: _CheckpointGenerationPointer | None = self.read(
            checkpoint_path
        )
        pointer: _CheckpointGenerationPointer = _CheckpointGenerationPointer(
            generation_id=generation_id,
            previous_generation_id=(
                previous_pointer.generation_id
                if previous_pointer is not None
                else None
            )
        )
        self._write(self._pointer_path(checkpoint_path), pointer)
        return pointer

    def _pointer_path(self, checkpoint_path: str | Path) -> str:
        # Resolves the pointer beneath the checkpoint root through StoragePath.
        return StoragePath(checkpoint_path).join(self._pointer_filename)

    def _write(
        self,
        pointer_path: str | Path,
        pointer: _CheckpointGenerationPointer
    ) -> None:
        # Commits remote JSON transactionally or local JSON through fsynced
        # temporary-file replacement and parent-directory synchronization.
        pointer_payload: str = pointer.model_dump_json()
        storage_path: StoragePath = StoragePath(pointer_path)
        if not storage_path.is_local:
            filesystem: AbstractFileSystem = storage_path.filesystem
            with filesystem.transaction:
                with storage_path.open_text("w") as opened_file:
                    opened_file.write(pointer_payload)
            return
        local_pointer_path: Path = Path(storage_path.normalized)
        local_pointer_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_file: tuple[int, str] = tempfile.mkstemp(
            dir=str(local_pointer_path.parent),
            prefix=f".{self._pointer_filename}-"
        )
        file_descriptor: int = temporary_file[0]
        temporary_path_string: str = temporary_file[1]
        temporary_path: Path = Path(temporary_path_string)
        try:
            pointer_file: io.TextIOBase = os.fdopen(
                file_descriptor,
                "w",
                encoding="utf-8"
            )
            with pointer_file:
                pointer_file.write(pointer_payload)
                pointer_file.flush()
                os.fsync(pointer_file.fileno())
            temporary_path.replace(local_pointer_path)
            directory_descriptor: int = os.open(
                local_pointer_path.parent,
                os.O_RDONLY
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except BaseException:
            if temporary_path.exists():
                temporary_path.unlink()
            raise
