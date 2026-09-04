import io
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal, cast
from uuid import uuid7

from fsspec.spec import AbstractFileSystem
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from syntheticmind.utilities.exceptions import CheckpointError
from syntheticmind.utilities.filesystem import get_filesystem, is_file, is_local_path
from syntheticmind.utilities.filesystem import join_path, normalize_path

__all__: list[str] = []

type CheckpointBroadcast = Callable[[object, int], object]
type CheckpointGenerationId = Annotated[
    str,
    StringConstraints(
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        ascii_only=True
    )
]


class _CheckpointGenerationPointer(BaseModel):
    schema_version: Literal[1] = 1
    generation_id: CheckpointGenerationId
    previous_generation_id: CheckpointGenerationId | None = None
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        validate_default=True
    )

    @property
    def retained_generation_ids(self) -> frozenset[str]:
        return frozenset(
            generation_id
            for generation_id in (
                self.generation_id,
                self.previous_generation_id
            )
            if generation_id is not None
        )


class _CheckpointGenerationIndex:
    def __init__(self) -> None:
        self._pointer_filename: str = "active_generation"

    def exists(self, checkpoint_path: str | Path) -> bool:
        return is_file(self._pointer_path(checkpoint_path))

    def create_generation_id(
        self,
        global_rank: int,
        broadcast: CheckpointBroadcast
    ) -> str:
        local_generation_id: object = str(uuid7()) if global_rank == 0 else None
        broadcast_generation_id: object = broadcast(local_generation_id, 0)
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
        pointer_path: str = self._pointer_path(checkpoint_path)
        if not is_file(pointer_path):
            return None
        with get_filesystem(pointer_path).open(
            os.fspath(pointer_path),
            "r",
            encoding="utf-8"
        ) as pointer_file:
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
        return join_path(checkpoint_path, self._pointer_filename)

    def _write(
        self,
        pointer_path: str | Path,
        pointer: _CheckpointGenerationPointer
    ) -> None:
        pointer_payload: str = pointer.model_dump_json()
        if not is_local_path(pointer_path):
            filesystem: AbstractFileSystem = get_filesystem(pointer_path)
            with filesystem.transaction:
                opened_file: object = filesystem.open(
                    os.fspath(pointer_path),
                    "w",
                    encoding="utf-8"
                )
                with cast(io.TextIOBase, opened_file) as pointer_file:
                    pointer_file.write(pointer_payload)
            return
        local_pointer_path: Path = Path(normalize_path(pointer_path))
        local_pointer_path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor: int
        temporary_path_string: str
        file_descriptor, temporary_path_string = tempfile.mkstemp(
            dir=str(local_pointer_path.parent),
            prefix=f".{self._pointer_filename}-"
        )
        temporary_path: Path = Path(temporary_path_string)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as pointer_file:
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
