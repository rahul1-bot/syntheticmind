# This module:
# 1. Defines StoragePath, the path-bound owner for local and fsspec-backed remote
#    filesystem identity, resources, metadata, traversal, mutation, and staging
#
# Design decisions:
# - Construction resolves the fsspec backend, backend-local path, and protocol
#   once; every later operation therefore acts on one stable resource identity
# - Public training APIs retain str or Path inputs and construct StoragePath only
#   at the persistence boundary, preventing filesystem objects from leaking into
#   model, callback, logger, or profiler contracts
# - Local paths use operating-system normalization and joining, while remote
#   paths retain protocol-qualified POSIX semantics
# - Write-capable resource methods create the parent directory before opening,
#   giving checkpoint, logger, and profiler owners one consistent preparation
#   rule without moving artifact semantics into storage
# - local_artifact yields a direct local path or an isolated temporary staging
#   path whose completed file uploads on successful context exit; producers can
#   use libraries that require local files without implementing remote copying
# - Containment and equality compare normalized identities and never infer
#   ownership from unresolved relative string prefixes
#
# Author: Rahul Sawhney

import io
import os
import posixpath
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import cast, override

import fsspec.utils
from fsspec.core import url_to_fs
from fsspec.spec import AbstractFileSystem

__all__: list[str] = ["StoragePath"]


class StoragePath(os.PathLike[str]):
    # One resolved local or remote path. The class owns filesystem mechanics only;
    # checkpoint, logger, profiler, and Trainer components retain decisions about
    # what an artifact means, when it is committed, and whether it may be removed.
    def __init__(self, path: str | Path) -> None:
        # Resolves protocol and backend identity through fsspec without touching
        # the resource itself.
        value: str = os.fspath(path)
        resolved: tuple[AbstractFileSystem, str] = cast(
            tuple[AbstractFileSystem, str],
            url_to_fs(value)
        )
        self._value: str = value
        self._filesystem: AbstractFileSystem = resolved[0]
        self._backend_path: str = resolved[1]
        self._protocol: str = fsspec.utils.get_protocol(value)

    @override
    def __fspath__(self) -> str:
        # Original protocol-qualified value for os.PathLike interoperability.
        return self._value

    @property
    def value(self) -> str:
        # Original client-supplied path string.
        return self._value

    @property
    def filesystem(self) -> AbstractFileSystem:
        # Resolved fsspec backend used by transaction-aware consumers.
        return self._filesystem

    @property
    def is_local(self) -> bool:
        # Whether the resolved protocol is the local file backend.
        return self._protocol == "file"

    @property
    def normalized(self) -> str:
        # Expanded normalized local path or unchanged protocol-qualified remote
        # identity.
        if not self.is_local:
            return self._value
        return os.path.normpath(os.path.expanduser(self._backend_path))

    @property
    def parent(self) -> str:
        # Parent path under local operating-system or remote POSIX semantics.
        if self.is_local:
            return os.path.dirname(self.normalized)
        separator_index: int = self._value.rfind("/")
        protocol_index: int = self._value.find("://")
        if separator_index <= protocol_index + 2:
            return self._value
        return self._value[:separator_index]

    @property
    def name(self) -> str:
        # Final path component independent of backend protocol.
        return self._value.rstrip("/").rsplit("/", 1)[-1]

    @property
    def size(self) -> int:
        # Validated byte size reported by the backend.
        size_value: int | float = cast(
            int | float,
            self._filesystem.size(self._backend_path)
        )
        if isinstance(size_value, bool) or not isinstance(size_value, int | float):
            raise TypeError(
                f"Filesystem size for {self._value!r} must be numeric, got "
                f"{type(size_value).__name__}."
            )
        return int(size_value)

    @property
    def modified_time(self) -> float:
        # Backend modification time normalized to Unix seconds.
        modified_value: datetime | int | float = cast(
            datetime | int | float,
            self._filesystem.modified(self._backend_path)
        )
        if isinstance(modified_value, datetime):
            return modified_value.timestamp()
        if isinstance(modified_value, bool) or not isinstance(
            modified_value,
            int | float
        ):
            raise TypeError(
                f"Filesystem modification time for {self._value!r} must be "
                f"numeric or datetime, got {type(modified_value).__name__}."
            )
        return float(modified_value)

    def join(self, *parts: str) -> str:
        # Joins non-empty components without discarding a remote protocol prefix.
        if self.is_local:
            return os.path.join(self.normalized, *parts)
        normalized_parts: tuple[str, ...] = tuple(
            part.strip("/") for part in parts if part
        )
        if not normalized_parts:
            return self._value
        return posixpath.join(self._value.rstrip("/"), *normalized_parts)

    def exists(self) -> bool:
        # Whether any resource exists at this path.
        return bool(self._filesystem.exists(self._backend_path))

    def is_file(self) -> bool:
        # Whether the backend identifies the path as a regular file.
        return bool(self._filesystem.isfile(self._backend_path))

    def is_directory(self) -> bool:
        # Whether the backend identifies the path as a directory.
        return bool(self._filesystem.isdir(self._backend_path))

    def make_directory(self) -> None:
        # Creates the path and missing parents idempotently; an empty backend path
        # requires no mutation.
        if not self._backend_path:
            return
        self._filesystem.makedirs(self._backend_path, exist_ok=True)

    def list_directory(self) -> list[str]:
        # Lists direct children as protocol-correct full paths and validates that
        # the backend returns string identities.
        if not self.is_directory():
            return []
        listed: list[str] = cast(
            list[str],
            self._filesystem.ls(self._backend_path, detail=False)
        )
        if not isinstance(listed, list):
            raise TypeError(
                f"Filesystem listing for {self._value!r} must be a list, got "
                f"{type(listed).__name__}."
            )
        paths: list[str] = []
        listed_path: str
        for listed_path in listed:
            if not isinstance(listed_path, str):
                raise TypeError(
                    "Filesystem listing entries must be strings, got "
                    f"{type(listed_path).__name__}."
                )
            paths.append(self.join(StoragePath(listed_path).name))
        return paths

    def open_text(
        self,
        mode: str,
        newline: str | None = None
    ) -> io.TextIOWrapper:
        # Opens UTF-8 text and prepares the parent for write, append, or exclusive
        # creation modes.
        if any(marker in mode for marker in ("w", "a", "x")):
            StoragePath(self.parent).make_directory()
        opened: io.TextIOWrapper = cast(
            io.TextIOWrapper,
            self._filesystem.open(
                self._backend_path,
                mode,
                encoding="utf-8",
                newline=newline
            )
        )
        return opened

    def open_binary(
        self,
        mode: str,
        filesystem: AbstractFileSystem | None = None
    ) -> io.BufferedIOBase:
        # Opens binary data through this backend or an explicitly supplied
        # transaction-bound backend instance.
        if any(marker in mode for marker in ("w", "a", "x")):
            StoragePath(self.parent).make_directory()
        resolved_filesystem: AbstractFileSystem = (
            self._filesystem if filesystem is None else filesystem
        )
        return cast(
            io.BufferedIOBase,
            resolved_filesystem.open(self._backend_path, mode)
        )

    def read_text(self) -> str:
        # Reads the complete UTF-8 text resource with deterministic closure.
        with self.open_text("r") as file:
            return file.read()

    def write_text(self, value: str) -> None:
        # Replaces the complete UTF-8 text resource.
        with self.open_text("w") as file:
            file.write(value)

    def append_text(self, value: str) -> None:
        # Appends natively on local storage; remote backends without dependable
        # append semantics are rewritten from their existing complete value.
        if self.is_local:
            with self.open_text("a") as file:
                file.write(value)
            return
        existing_value: str = self.read_text() if self.exists() else ""
        self.write_text(f"{existing_value}{value}")

    def remove(self, recursive: bool = False) -> None:
        # Removes an existing resource through the resolved backend. Callers own
        # the semantic decision and must request recursive directory deletion
        # explicitly.
        if self.exists():
            self._filesystem.rm(self._backend_path, recursive=recursive)

    def copy_from_local(self, source: Path) -> None:
        # Uploads one completed local file after ensuring the destination parent.
        StoragePath(self.parent).make_directory()
        self._filesystem.put_file(str(source), self._backend_path)

    def same_as(self, other: str | Path) -> bool:
        # Compares absolute normalized local identities or normalized remote
        # protocol identities without filesystem mutation.
        other_path: StoragePath = StoragePath(other)
        if self.is_local and other_path.is_local:
            return Path(self.normalized).absolute() == Path(
                other_path.normalized
            ).absolute()
        return self.normalized.rstrip("/") == other_path.normalized.rstrip("/")

    def is_within(self, directory: str | Path) -> bool:
        # Tests strict descendant containment for path-escape protection; a path
        # is not considered within itself.
        directory_path: StoragePath = StoragePath(directory)
        if self.is_local and directory_path.is_local:
            absolute_path: Path = Path(self.normalized).absolute()
            absolute_directory: Path = Path(directory_path.normalized).absolute()
            return absolute_directory in absolute_path.parents
        prefix: str = f"{directory_path.normalized.rstrip('/')}/"
        return self.normalized.startswith(prefix)

    @contextmanager
    def local_artifact(self) -> Iterator[Path]:
        # Yields a direct local destination or an isolated temporary file for a
        # remote artifact. A produced remote file uploads only after the client
        # context exits successfully.
        if self.is_local:
            local_path: Path = Path(self.normalized)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            yield local_path
            return
        with tempfile.TemporaryDirectory(
            prefix="syntheticmind-artifact-"
        ) as temporary_directory:
            temporary_path: Path = Path(temporary_directory) / self.name
            yield temporary_path
            if temporary_path.is_file():
                self.copy_from_local(temporary_path)
