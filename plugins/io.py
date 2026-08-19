from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import override

import numpy as np
import torch

from syntheticmind.utilities.checkpoint import load_checkpoint, save_checkpoint
from syntheticmind.utilities.filesystem import remove_path
from syntheticmind.utilities.types import CheckpointDict, CheckpointValue, MapLocation

__all__: list[str] = [
    "AsyncCheckpointIO",
    "CheckpointIO",
    "TorchCheckpointIO"
]

type CheckpointPath = str | Path


class CheckpointIO:
    def save_checkpoint(
        self,
        checkpoint: CheckpointDict,
        path: CheckpointPath,
        storage_options: object | None = None
    ) -> None:
        raise NotImplementedError

    def load_checkpoint(
        self,
        path: CheckpointPath,
        map_location: MapLocation = None,
        weights_only: bool | None = None
    ) -> CheckpointDict:
        raise NotImplementedError

    def remove_checkpoint(self, path: CheckpointPath) -> None:
        raise NotImplementedError

    def teardown(self) -> None:
        pass


class TorchCheckpointIO(CheckpointIO):
    @override
    def save_checkpoint(
        self,
        checkpoint: CheckpointDict,
        path: CheckpointPath,
        storage_options: object | None = None
    ) -> None:
        if storage_options is not None:
            raise TypeError(
                "storage_options are unsupported by TorchCheckpointIO. "
                "Use a custom CheckpointIO implementation for storage-specific options."
            )
        save_checkpoint(checkpoint, path)

    @override
    def load_checkpoint(
        self,
        path: CheckpointPath,
        map_location: MapLocation = None,
        weights_only: bool | None = None
    ) -> CheckpointDict:
        return load_checkpoint(
            path,
            map_location="cpu" if map_location is None else map_location,
            weights_only=weights_only
        )

    @override
    def remove_checkpoint(self, path: CheckpointPath) -> None:
        remove_path(path, recursive=True)


class _WrappingCheckpointIO(CheckpointIO):
    def __init__(self, checkpoint_io: CheckpointIO | None = None) -> None:
        self._checkpoint_io: CheckpointIO | None = checkpoint_io
        self._base_checkpoint_io_configured: bool = checkpoint_io is not None
        if isinstance(checkpoint_io, _WrappingCheckpointIO):
            self._base_checkpoint_io_configured = (
                checkpoint_io._base_checkpoint_io_configured
            )

    @property
    def checkpoint_io(self) -> CheckpointIO | None:
        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, checkpoint_io: CheckpointIO) -> None:
        if isinstance(checkpoint_io, _WrappingCheckpointIO):
            raise TypeError("A wrapping CheckpointIO cannot be installed as its own base.")
        if self._checkpoint_io is None:
            self._checkpoint_io = checkpoint_io
            self._base_checkpoint_io_configured = True
            return
        if (
            isinstance(self._checkpoint_io, _WrappingCheckpointIO)
            and not self._base_checkpoint_io_configured
        ):
            self._checkpoint_io.checkpoint_io = checkpoint_io
            self._base_checkpoint_io_configured = True

    def _require_checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:
            raise RuntimeError("The wrapping CheckpointIO has no base implementation.")
        return self._checkpoint_io

    @override
    def save_checkpoint(
        self,
        checkpoint: CheckpointDict,
        path: CheckpointPath,
        storage_options: object | None = None
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


class AsyncCheckpointIO(_WrappingCheckpointIO):
    def __init__(self, checkpoint_io: CheckpointIO | None = None) -> None:
        super().__init__(checkpoint_io)
        self._executor: ThreadPoolExecutor | None = None
        self._error: BaseException | None = None

    def _ensure_setup(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)
        return self._executor

    def _raise_error(self) -> None:
        if self._error is not None:
            raise self._error

    def _wait(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._raise_error()

    @override
    def save_checkpoint(
        self,
        checkpoint: CheckpointDict,
        path: CheckpointPath,
        storage_options: object | None = None
    ) -> None:
        self._raise_error()
        snapshot: CheckpointDict = {
            key: _clone_checkpoint_value(value)
            for key, value in checkpoint.items()
        }

        def save() -> None:
            try:
                self._require_checkpoint_io().save_checkpoint(
                    snapshot,
                    path,
                    storage_options
                )
            except BaseException as exception:
                self._error = exception

        self._ensure_setup().submit(save)
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
            self._executor = None
        asynchronous_error: BaseException | None = self._error
        try:
            super().teardown()
        except BaseException as teardown_error:
            if asynchronous_error is None:
                raise
            detail: str = " ".join(str(teardown_error).split())
            description: str = type(teardown_error).__name__
            if detail:
                description = f"{description}: {detail}"
            asynchronous_error.add_note(
                f"CheckpointIO teardown also failed: {description}"
            )
        if asynchronous_error is not None:
            raise asynchronous_error


def _clone_checkpoint_value(value: CheckpointValue) -> CheckpointValue:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list):
        return [_clone_checkpoint_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_checkpoint_value(item) for item in value)
    if isinstance(value, dict):
        return {
            key: _clone_checkpoint_value(item)
            for key, item in value.items()
        }
    return value
