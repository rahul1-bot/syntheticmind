import io
import os
import posixpath
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import cast

import fsspec.utils
from fsspec.core import url_to_fs
from fsspec.spec import AbstractFileSystem

__all__: list[str] = [
    "append_text",
    "basename",
    "copy_local_file",
    "dirname",
    "exists",
    "file_size",
    "get_filesystem",
    "is_directory",
    "is_file",
    "is_local_path",
    "is_path_within",
    "join_path",
    "list_directory",
    "local_artifact_path",
    "make_directory",
    "modified_time",
    "normalize_path",
    "open_binary",
    "open_text",
    "read_text",
    "remove_path",
    "same_path",
    "write_text"
]


def get_filesystem(path: str | Path) -> AbstractFileSystem:
    resolved: tuple[AbstractFileSystem, str] = cast(
        tuple[AbstractFileSystem, str],
        url_to_fs(os.fspath(path))
    )
    return resolved[0]


def is_local_path(path: str | Path) -> bool:
    return fsspec.utils.get_protocol(os.fspath(path)) == "file"


def normalize_path(path: str | Path) -> str:
    value: str = os.fspath(path)
    if not is_local_path(value):
        return value
    if value.startswith("file://"):
        resolved: tuple[AbstractFileSystem, str] = cast(
            tuple[AbstractFileSystem, str],
            url_to_fs(value)
        )
        value: str = resolved[1]
    return os.path.normpath(os.path.expanduser(value))


def join_path(path: str | Path, *parts: str) -> str:
    value: str = os.fspath(path)
    if is_local_path(value):
        return os.path.join(value, *parts)
    normalized_parts: tuple[str, ...] = tuple(
        part.strip("/") for part in parts if part
    )
    if not normalized_parts:
        return value
    return posixpath.join(value.rstrip("/"), *normalized_parts)


def dirname(path: str | Path) -> str:
    value: str = os.fspath(path)
    if is_local_path(value):
        return os.path.dirname(value)
    separator_index: int = value.rfind("/")
    protocol_index: int = value.find("://")
    if separator_index <= protocol_index + 2:
        return value
    return value[:separator_index]


def basename(path: str | Path) -> str:
    return os.fspath(path).rstrip("/").rsplit("/", 1)[-1]


def exists(path: str | Path) -> bool:
    return bool(get_filesystem(path).exists(os.fspath(path)))


def is_file(path: str | Path) -> bool:
    return bool(get_filesystem(path).isfile(os.fspath(path)))


def is_directory(path: str | Path) -> bool:
    return bool(get_filesystem(path).isdir(os.fspath(path)))


def make_directory(path: str | Path) -> None:
    value: str = os.fspath(path)
    if not value:
        return
    get_filesystem(value).makedirs(value, exist_ok=True)


def list_directory(path: str | Path) -> list[str]:
    value: str = os.fspath(path)
    if not is_directory(value):
        return []
    listed: object = get_filesystem(value).ls(value, detail=False)
    if not isinstance(listed, list):
        raise TypeError(
            f"Filesystem listing for {value!r} must be a list, got "
            f"{type(listed).__name__}."
        )
    paths: list[str] = []
    listed_path: object
    for listed_path in listed:
        if not isinstance(listed_path, str):
            raise TypeError(
                f"Filesystem listing entries must be strings, got "
                f"{type(listed_path).__name__}."
            )
        paths.append(join_path(value, basename(listed_path)))
    return paths


def file_size(path: str | Path) -> int:
    size: object = get_filesystem(path).size(os.fspath(path))
    if isinstance(size, bool) or not isinstance(size, int | float):
        raise TypeError(
            f"Filesystem size for {os.fspath(path)!r} must be numeric, got "
            f"{type(size).__name__}."
        )
    return int(size)


def modified_time(path: str | Path) -> float:
    value: object = get_filesystem(path).modified(os.fspath(path))
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(
            f"Filesystem modification time for {os.fspath(path)!r} must be "
            f"numeric or datetime, got {type(value).__name__}."
        )
    return float(value)


def open_text(
    path: str | Path,
    mode: str,
    newline: str | None = None
) -> io.TextIOWrapper:
    value: str = os.fspath(path)
    if any(marker in mode for marker in ("w", "a", "x")):
        make_directory(dirname(value))
    opened: object = get_filesystem(value).open(
        value,
        mode,
        encoding="utf-8",
        newline=newline
    )
    return cast(io.TextIOWrapper, opened)


def open_binary(
    path: str | Path,
    mode: str,
    filesystem: AbstractFileSystem | None = None
) -> io.BufferedIOBase:
    value: str = os.fspath(path)
    if any(marker in mode for marker in ("w", "a", "x")):
        make_directory(dirname(value))
    resolved_filesystem: AbstractFileSystem = (
        get_filesystem(value) if filesystem is None else filesystem
    )
    opened: object = resolved_filesystem.open(value, mode)
    return cast(io.BufferedIOBase, opened)


def read_text(path: str | Path) -> str:
    with open_text(path, "r") as file:
        return file.read()


def write_text(path: str | Path, value: str) -> None:
    with open_text(path, "w") as file:
        file.write(value)


def append_text(path: str | Path, value: str) -> None:
    if is_local_path(path):
        with open_text(path, "a") as file:
            file.write(value)
        return
    existing_value: str = read_text(path) if exists(path) else ""
    write_text(path, f"{existing_value}{value}")


def remove_path(path: str | Path, recursive: bool = False) -> None:
    value: str = os.fspath(path)
    filesystem: AbstractFileSystem = get_filesystem(value)
    if filesystem.exists(value):
        filesystem.rm(value, recursive=recursive)


def copy_local_file(source: Path, destination: str | Path) -> None:
    value: str = os.fspath(destination)
    make_directory(dirname(value))
    get_filesystem(value).put_file(str(source), value)


def same_path(first: str | Path, second: str | Path) -> bool:
    first_value: str = normalize_path(first)
    second_value: str = normalize_path(second)
    if is_local_path(first_value) and is_local_path(second_value):
        return Path(first_value).absolute() == Path(second_value).absolute()
    return first_value.rstrip("/") == second_value.rstrip("/")


def is_path_within(path: str | Path, directory: str | Path) -> bool:
    path_value: str = normalize_path(path)
    directory_value: str = normalize_path(directory)
    if is_local_path(path_value) and is_local_path(directory_value):
        absolute_path: Path = Path(path_value).absolute()
        absolute_directory: Path = Path(directory_value).absolute()
        return absolute_directory in absolute_path.parents
    prefix: str = f"{directory_value.rstrip('/')}/"
    return path_value.startswith(prefix)


@contextmanager
def local_artifact_path(path: str | Path) -> Iterator[Path]:
    value: str = os.fspath(path)
    if is_local_path(value):
        local_path: Path = Path(normalize_path(value))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        yield local_path
        return
    with tempfile.TemporaryDirectory(prefix="syntheticmind-artifact-") as temporary_directory:
        temporary_path: Path = Path(temporary_directory) / basename(value)
        yield temporary_path
        if temporary_path.is_file():
            copy_local_file(temporary_path, value)
