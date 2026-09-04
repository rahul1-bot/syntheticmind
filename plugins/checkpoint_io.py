from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import tempfile
from typing import Protocol, cast, override, runtime_checkable

import numpy as np
import torch
from fsspec.spec import AbstractFileSystem

from syntheticmind.storage.filesystem import StoragePath
from syntheticmind.types import CheckpointDict, CheckpointValue, MapLocation

__all__: list[str] = [
    "AsyncCheckpointIO",
    "CheckpointIO",
    "CheckpointPath",
    "CheckpointStorageOptions",
    "TorchCheckpointIO"
]

type CheckpointPath = str | Path
type CheckpointStorageOptions = Mapping[str, CheckpointValue] | None


@runtime_checkable
class CheckpointIO(Protocol):
    def save_checkpoint(
        self,
        checkpoint: CheckpointDict,
        path: CheckpointPath,
        storage_options: CheckpointStorageOptions = None
    ) -> None: ...

    def load_checkpoint(
        self,
        path: CheckpointPath,
        map_location: MapLocation = None,
        weights_only: bool | None = None
    ) -> CheckpointDict: ...

    def remove_checkpoint(self, path: CheckpointPath) -> None: ...

    def teardown(self) -> None: ...


class TorchCheckpointIO(CheckpointIO):
    @override
    def save_checkpoint(
        self,
        checkpoint: CheckpointDict,
        path: CheckpointPath,
        storage_options: CheckpointStorageOptions = None
    ) -> None:
        if storage_options is not None:
            raise TypeError(
                "storage_options are unsupported by TorchCheckpointIO. "
                "Use a custom CheckpointIO implementation for storage-specific options."
            )
        if StoragePath(path).is_local:
            self._save_local(checkpoint, path)
            return
        self._save_remote(checkpoint, path)

    @override
    def load_checkpoint(
        self,
        path: CheckpointPath,
        map_location: MapLocation = None,
        weights_only: bool | None = None
    ) -> CheckpointDict:
        if not StoragePath(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        resolved_map_location: MapLocation = (
            "cpu" if map_location is None else map_location
        )
        checkpoint: CheckpointValue
        if StoragePath(path).is_local:
            checkpoint = cast(
                CheckpointValue,
                torch.load(
                    Path(StoragePath(path).normalized),
                    map_location=resolved_map_location,
                    weights_only=False if weights_only is None else weights_only
                )
            )
        else:
            checkpoint = self._load_remote(
                path,
                resolved_map_location,
                weights_only
            )
        if not isinstance(checkpoint, dict):
            raise TypeError(
                f"Checkpoint payload must be a dictionary, got {type(checkpoint).__name__}."
            )
        return cast(CheckpointDict, checkpoint)

    @override
    def remove_checkpoint(self, path: CheckpointPath) -> None:
        StoragePath(path).remove(recursive=True)

    @override
    def teardown(self) -> None:
        pass

    def _save_local(self, checkpoint: CheckpointDict, path: CheckpointPath) -> None:
        local_path: Path = Path(StoragePath(path).normalized)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor: int
        temporary_path_string: str
        file_descriptor, temporary_path_string = tempfile.mkstemp(
            dir=str(local_path.parent),
            suffix=".tmp"
        )
        os.close(file_descriptor)
        temporary_path: Path = Path(temporary_path_string)
        try:
            torch.save(checkpoint, temporary_path)
            temporary_path.replace(local_path)
        except BaseException:
            if temporary_path.exists():
                temporary_path.unlink()
            raise

    def _save_remote(self, checkpoint: CheckpointDict, path: CheckpointPath) -> None:
        storage_path: StoragePath = StoragePath(path)
        StoragePath(storage_path.parent).make_directory()
        filesystem: AbstractFileSystem = storage_path.filesystem
        with tempfile.TemporaryFile() as temporary_file:
            torch.save(checkpoint, temporary_file)
            temporary_file.seek(0)
            with filesystem.transaction:
                with storage_path.open_binary(
                    "wb",
                    filesystem=filesystem
                ) as destination:
                    shutil.copyfileobj(temporary_file, destination)

    def _load_remote(
        self,
        path: CheckpointPath,
        map_location: MapLocation,
        weights_only: bool | None
    ) -> CheckpointValue:
        storage_path: StoragePath = StoragePath(path)
        with (
            storage_path.open_binary("rb") as source,
            tempfile.TemporaryFile() as temporary_file
        ):
            shutil.copyfileobj(source, temporary_file)
            temporary_file.seek(0)
            return cast(
                CheckpointValue,
                torch.load(
                    temporary_file,
                    map_location=map_location,
                    weights_only=False if weights_only is None else weights_only
                )
            )


class _WrappingCheckpointIO(CheckpointIO):
    def __init__(self, checkpoint_io: CheckpointIO | None = None) -> None:
        self._checkpoint_io: CheckpointIO | None = checkpoint_io
        self._base_checkpoint_io_configured: bool = checkpoint_io is not None
        if isinstance(checkpoint_io, _WrappingCheckpointIO):
            self._base_checkpoint_io_configured: bool = (
                checkpoint_io._base_checkpoint_io_configured
            )

    @override
    def save_checkpoint(
        self,
        checkpoint: CheckpointDict,
        path: CheckpointPath,
        storage_options: CheckpointStorageOptions = None
    ) -> None:
        self._require_checkpoint_io().save_checkpoint(
            checkpoint,
            path,
            storage_options
        )

    @override
    def load_checkpoint(
        self,
        path: CheckpointPath,
        map_location: MapLocation = None,
        weights_only: bool | None = None
    ) -> CheckpointDict:
        return self._require_checkpoint_io().load_checkpoint(
            path,
            map_location,
            weights_only
        )

    @override
    def remove_checkpoint(self, path: CheckpointPath) -> None:
        self._require_checkpoint_io().remove_checkpoint(path)

    @override
    def teardown(self) -> None:
        if self._checkpoint_io is not None:
            self._checkpoint_io.teardown()

    @property
    def checkpoint_io(self) -> CheckpointIO | None:
        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, checkpoint_io: CheckpointIO) -> None:
        if isinstance(checkpoint_io, _WrappingCheckpointIO):
            raise TypeError("A wrapping CheckpointIO cannot be installed as its own base.")
        if self._checkpoint_io is None:
            self._checkpoint_io: CheckpointIO | None = checkpoint_io
            self._base_checkpoint_io_configured: bool = True
            return
        if (
            isinstance(self._checkpoint_io, _WrappingCheckpointIO)
            and not self._base_checkpoint_io_configured
        ):
            self._checkpoint_io.checkpoint_io: CheckpointIO = checkpoint_io
            self._base_checkpoint_io_configured: bool = True

    def _require_checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:
            raise RuntimeError("The wrapping CheckpointIO has no base implementation.")
        return self._checkpoint_io


class AsyncCheckpointIO(_WrappingCheckpointIO):
    def __init__(self, checkpoint_io: CheckpointIO | None = None) -> None:
        super().__init__(checkpoint_io)
        self._executor: ThreadPoolExecutor | None = None
        self._error: BaseException | None = None

    @override
    def save_checkpoint(
        self,
        checkpoint: CheckpointDict,
        path: CheckpointPath,
        storage_options: CheckpointStorageOptions = None
    ) -> None:
        self._raise_error()
        snapshot: CheckpointDict = {}
        checkpoint_name: str
        checkpoint_value: CheckpointValue
        for checkpoint_name, checkpoint_value in checkpoint.items():
            snapshot[checkpoint_name]: CheckpointValue = self._clone_value(
                checkpoint_value
            )
        self._ensure_setup().submit(
            self._save_snapshot,
            snapshot,
            path,
            storage_options
        )
        self._raise_error()

    @override
    def load_checkpoint(
        self,
        path: CheckpointPath,
        map_location: MapLocation = None,
        weights_only: bool | None = None
    ) -> CheckpointDict:
        self._wait()
        return super().load_checkpoint(path, map_location, weights_only)

    @override
    def remove_checkpoint(self, path: CheckpointPath) -> None:
        self._wait()
        super().remove_checkpoint(path)

    @override
    def teardown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor: ThreadPoolExecutor | None = None
        asynchronous_error: BaseException | None = self._error
        try:
            super().teardown()
        except BaseException as teardown_error:
            if asynchronous_error is None:
                raise
            detail: str = " ".join(str(teardown_error).split())
            description: str = type(teardown_error).__name__
            if detail:
                description: str = f"{description}: {detail}"
            asynchronous_error.add_note(
                f"CheckpointIO teardown also failed: {description}"
            )
        if asynchronous_error is not None:
            raise asynchronous_error

    def _ensure_setup(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
                max_workers=1
            )
        return self._executor

    def _raise_error(self) -> None:
        if self._error is not None:
            raise self._error

    def _wait(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor: ThreadPoolExecutor | None = None
        self._raise_error()

    def _save_snapshot(
        self,
        snapshot: CheckpointDict,
        path: CheckpointPath,
        storage_options: CheckpointStorageOptions
    ) -> None:
        try:
            self._require_checkpoint_io().save_checkpoint(
                snapshot,
                path,
                storage_options
            )
        except BaseException as exception:
            self._error: BaseException | None = exception

    def _clone_value(self, value: CheckpointValue) -> CheckpointValue:
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, list):
            cloned_list: list[CheckpointValue] = []
            list_value: CheckpointValue
            for list_value in value:
                cloned_list.append(self._clone_value(list_value))
            return cloned_list
        if isinstance(value, tuple):
            cloned_values: list[CheckpointValue] = []
            tuple_value: CheckpointValue
            for tuple_value in value:
                cloned_values.append(self._clone_value(tuple_value))
            return tuple(cloned_values)
        if isinstance(value, dict):
            cloned_mapping: dict[str, CheckpointValue] = {}
            mapping_name: str
            mapping_value: CheckpointValue
            for mapping_name, mapping_value in value.items():
                cloned_mapping[mapping_name]: CheckpointValue = self._clone_value(
                    mapping_value
                )
            return cloned_mapping
        return value
