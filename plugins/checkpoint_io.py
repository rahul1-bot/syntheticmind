# This module:
# 1. Defines CheckpointIO, the interchangeable transport contract for complete
#    checkpoint dictionaries
# 2. Implements local and fsspec-backed torch persistence with atomic local
#    publication and remote transaction boundaries
# 3. Implements wrapper composition and asynchronous snapshot persistence
#
# Design decisions:
# - CheckpointIO owns bytes and storage transport only; Trainer assembles the
#   trajectory, Strategy owns model and optimizer representation, and callbacks
#   own cadence, ranking, and retention
# - Local saves write a same-directory temporary file and atomically replace the
#   destination, so readers observe the previous complete artifact or the new
#   complete artifact rather than a partially written checkpoint
# - Remote saves serialize into a temporary local file, then copy within the
#   fsspec transaction exposed by the resolved backend
# - Wrapping implementations receive their concrete base after Trainer resolves
#   plugins, allowing AsyncCheckpointIO to decorate default or custom storage
#   without claiming another persistence format
# - AsyncCheckpointIO clones the complete payload on the calling thread before
#   submission; later model or optimizer mutation cannot change the background
#   artifact
# - One background worker preserves save order, and load, remove, or teardown
#   waits for pending work and re-raises the original asynchronous failure
#
# Author: Rahul Sawhney

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
from syntheticmind.primitives.types import CheckpointDict, CheckpointValue, MapLocation

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
    # Structural checkpoint transport supplied through Trainer plugins. A custom
    # client implementation persists complete CheckpointDict values and owns any
    # storage-specific options, resources, and teardown it introduces.
    # Persists one complete snapshot to the requested path. Implementations must
    # either honor storage_options or reject them explicitly.
    def save_checkpoint(
        self,
        checkpoint: CheckpointDict,
        path: CheckpointPath,
        storage_options: CheckpointStorageOptions = None
    ) -> None: ...

    # Loads one complete checkpoint mapping under the requested device and
    # weights-only policy.
    def load_checkpoint(
        self,
        path: CheckpointPath,
        map_location: MapLocation = None,
        weights_only: bool | None = None
    ) -> CheckpointDict: ...

    # Removes the artifact represented by path; retention policy remains outside.
    def remove_checkpoint(self, path: CheckpointPath) -> None: ...

    # Releases transport-owned workers, sessions, handles, or deferred failures.
    def teardown(self) -> None: ...


class TorchCheckpointIO(CheckpointIO):
    # Default torch.save and torch.load transport for local files and fsspec
    # resources. It accepts no storage-specific option mapping because silent
    # option dropping would misrepresent a custom backend contract.
    @override
    def save_checkpoint(
        self,
        checkpoint: CheckpointDict,
        path: CheckpointPath,
        storage_options: CheckpointStorageOptions = None
    ) -> None:
        # Selects atomic local publication or transaction-scoped remote copying
        # after rejecting unsupported storage options.
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
        # Loads a complete mapping on CPU by default and honors explicit
        # map_location and weights_only controls for both local and remote paths.
        if not StoragePath(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        resolved_map_location: MapLocation = (
            "cpu" if map_location is None else map_location
        )
        if StoragePath(path).is_local:
            checkpoint: CheckpointValue = cast(
                CheckpointValue,
                torch.load(
                    Path(StoragePath(path).normalized),
                    map_location=resolved_map_location,
                    weights_only=False if weights_only is None else weights_only
                )
            )
        else:
            checkpoint: CheckpointValue = self._load_remote(
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
        # Removes a file or checkpoint directory through StoragePath. The caller
        # owns retention and rank policy.
        StoragePath(path).remove(recursive=True)

    @override
    def teardown(self) -> None:
        # Default transport owns no persistent resource.
        pass

    def _save_local(self, checkpoint: CheckpointDict, path: CheckpointPath) -> None:
        # Serializes beside the destination and publishes through atomic replace;
        # any failed temporary file is removed before the causal error propagates.
        local_path: Path = Path(StoragePath(path).normalized)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_file: tuple[int, str] = tempfile.mkstemp(
            dir=str(local_path.parent),
            suffix=".tmp"
        )
        file_descriptor: int = temporary_file[0]
        temporary_path_string: str = temporary_file[1]
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
        # Serializes once into a seekable temporary file and copies it inside the
        # resolved fsspec transaction.
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
        # Stages remote bytes into a seekable temporary file required by
        # torch.load, then returns the deserialized value for mapping validation.
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
    # Delegating checkpoint transport whose concrete base may be supplied at
    # construction or injected by the Trainer connector. Nested wrappers accept
    # one eventual base without forming wrapper cycles.
    def __init__(self, checkpoint_io: CheckpointIO | None = None) -> None:
        # Records whether a client explicitly selected the base so default
        # injection cannot overwrite intentional composition.
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
        # Forwards one complete snapshot to the resolved concrete transport.
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
        # Wait-free delegation for wrappers whose concrete subclasses do not add
        # ordering semantics.
        return self._require_checkpoint_io().load_checkpoint(
            path,
            map_location,
            weights_only
        )

    @override
    def remove_checkpoint(self, path: CheckpointPath) -> None:
        # Delegates removal without acquiring retention ownership.
        self._require_checkpoint_io().remove_checkpoint(path)

    @override
    def teardown(self) -> None:
        # Releases the resolved base exactly once when one was attached.
        if self._checkpoint_io is not None:
            self._checkpoint_io.teardown()

    @property
    def checkpoint_io(self) -> CheckpointIO | None:
        # Exposes the resolved transport for connector composition inspection.
        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, checkpoint_io: CheckpointIO) -> None:
        # Installs the first concrete base or forwards it into an unresolved
        # nested wrapper; a wrapper cannot install itself as its own base.
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
        # Resolves the concrete transport before a delegated operation begins.
        if self._checkpoint_io is None:
            raise RuntimeError("The wrapping CheckpointIO has no base implementation.")
        return self._checkpoint_io


class AsyncCheckpointIO(_WrappingCheckpointIO):
    # Single-worker asynchronous checkpoint decorator. Client code supplies it
    # through Trainer plugins; the connector injects TorchCheckpointIO when no
    # explicit base transport was provided.
    @override
    def __init__(self, checkpoint_io: CheckpointIO | None = None) -> None:
        # Starts detached from any worker; the executor is acquired only by the
        # first submitted save and preserves one stored causal failure.
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
        # Raises any prior background failure, clones the complete payload in the
        # caller, schedules one ordered save, and rechecks synchronous submission
        # errors before returning.
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
        # Waits for pending saves before reading so the returned artifact cannot
        # race an earlier asynchronous publication.
        self._wait()
        return super().load_checkpoint(path, map_location, weights_only)

    @override
    def remove_checkpoint(self, path: CheckpointPath) -> None:
        # Waits for pending saves before removal to preserve operation order.
        self._wait()
        super().remove_checkpoint(path)

    @override
    def teardown(self) -> None:
        # Joins the worker, tears down the base transport, preserves the original
        # asynchronous error, and attaches any secondary teardown failure as a
        # causal note.
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
        # Lazily creates one worker so unused asynchronous plugins allocate no
        # thread and all saves retain submission order.
        if self._executor is None:
            self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
                max_workers=1
            )
        return self._executor

    def _raise_error(self) -> None:
        # Propagates the original background failure at the next synchronous
        # public boundary without wrapping away its type or traceback.
        if self._error is not None:
            raise self._error

    def _wait(self) -> None:
        # Joins and releases the current executor, then propagates stored failure.
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
        # Background operation that stores the causal failure for deterministic
        # propagation at the next public boundary.
        try:
            self._require_checkpoint_io().save_checkpoint(
                snapshot,
                path,
                storage_options
            )
        except BaseException as exception:
            self._error: BaseException | None = exception

    def _clone_value(self, value: CheckpointValue) -> CheckpointValue:
        # Recursively snapshots tensors, NumPy arrays, and containers while
        # retaining immutable scalar values directly.
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
