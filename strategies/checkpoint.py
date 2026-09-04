import warnings
from collections.abc import Mapping
from typing import Protocol, cast, override, runtime_checkable

import torch
from torch import nn

from syntheticmind.plugins.checkpoint_io import (
    CheckpointIO,
    CheckpointPath,
    CheckpointStorageOptions,
    TorchCheckpointIO,
    _WrappingCheckpointIO
)
from syntheticmind.storage.filesystem import StoragePath
from syntheticmind.types import (
    CheckpointDict,
    CheckpointValue,
    ModelState,
    OptimizerState
)

__all__: list[str] = ["StrategyCheckpoint"]


@runtime_checkable
class StrategyCheckpoint(Protocol):
    def connect_model(self, model: nn.Module) -> None: ...

    def connect_optimizers(self, optimizers: list[torch.optim.Optimizer]) -> None: ...

    def save(
        self,
        checkpoint: CheckpointDict,
        filepath: CheckpointPath,
        storage_options: CheckpointStorageOptions,
        global_rank: int
    ) -> None: ...

    def load(
        self,
        checkpoint_path: CheckpointPath,
        weights_only: bool | None
    ) -> CheckpointDict: ...

    def load_metadata(
        self,
        checkpoint_path: CheckpointPath,
        weights_only: bool | None
    ) -> CheckpointDict: ...

    def is_checkpoint_path(self, checkpoint_path: CheckpointPath) -> bool: ...

    def model_state_dict(self) -> ModelState: ...

    def optimizer_state(
        self,
        optimizer: torch.optim.Optimizer
    ) -> OptimizerState: ...

    def restore(
        self,
        checkpoint: Mapping[str, CheckpointValue],
        strict: bool,
        restore_optimizers: bool
    ) -> None: ...

    def remove(self, filepath: CheckpointPath, global_rank: int) -> None: ...

    def teardown(self) -> None: ...

    def attach_checkpoint_io(self, checkpoint_io: CheckpointIO) -> None: ...

    @property
    def configured_checkpoint_io(self) -> CheckpointIO | None: ...


class _SingleFileCheckpoint(StrategyCheckpoint):
    def __init__(self, checkpoint_io: CheckpointIO | None) -> None:
        if checkpoint_io is not None:
            self._validate_checkpoint_io(checkpoint_io)
        self._checkpoint_io: CheckpointIO | None = checkpoint_io
        self._configured_checkpoint_io: CheckpointIO | None = checkpoint_io
        self._model: nn.Module | None = None
        self._optimizers: list[torch.optim.Optimizer] = []
        self._preloaded_path: str | None = None
        self._preloaded_checkpoint: CheckpointDict | None = None

    @override
    def connect_model(self, model: nn.Module) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("Checkpoint model must be a torch.nn.Module.")
        self._model: nn.Module | None = model

    @override
    def connect_optimizers(self, optimizers: list[torch.optim.Optimizer]) -> None:
        if not isinstance(optimizers, list):
            raise TypeError("Checkpoint optimizers must be a list.")
        optimizer: torch.optim.Optimizer
        for optimizer in optimizers:
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise TypeError("Checkpoint optimizers must contain Optimizer instances.")
        self._optimizers: list[torch.optim.Optimizer] = list(optimizers)

    @override
    def save(
        self,
        checkpoint: CheckpointDict,
        filepath: CheckpointPath,
        storage_options: CheckpointStorageOptions,
        global_rank: int
    ) -> None:
        if global_rank == 0:
            self._resolve_checkpoint_io().save_checkpoint(
                checkpoint,
                filepath,
                storage_options
            )

    @override
    def load(
        self,
        checkpoint_path: CheckpointPath,
        weights_only: bool | None
    ) -> CheckpointDict:
        normalized_path: str = StoragePath(checkpoint_path).normalized
        checkpoint: CheckpointDict | None = (
            self._preloaded_checkpoint
            if self._preloaded_path == normalized_path
            else None
        )
        self._preloaded_path: str | None = None
        self._preloaded_checkpoint: CheckpointDict | None = None
        if checkpoint is not None:
            return checkpoint
        return self._read(checkpoint_path, weights_only)

    @override
    def load_metadata(
        self,
        checkpoint_path: CheckpointPath,
        weights_only: bool | None
    ) -> CheckpointDict:
        checkpoint: CheckpointDict = self._read(checkpoint_path, weights_only)
        self._preloaded_path: str | None = StoragePath(checkpoint_path).normalized
        self._preloaded_checkpoint: CheckpointDict | None = checkpoint
        return checkpoint

    @override
    def is_checkpoint_path(self, checkpoint_path: CheckpointPath) -> bool:
        return StoragePath(checkpoint_path).is_file()

    @override
    def model_state_dict(self) -> ModelState:
        return cast(ModelState, self._require_model().state_dict())

    @override
    def optimizer_state(
        self,
        optimizer: torch.optim.Optimizer
    ) -> OptimizerState:
        return cast(OptimizerState, optimizer.state_dict())

    @override
    def restore(
        self,
        checkpoint: Mapping[str, CheckpointValue],
        strict: bool,
        restore_optimizers: bool
    ) -> None:
        state_dict: CheckpointValue | None = checkpoint.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise KeyError("Checkpoint is missing a model `state_dict`.")
        incompatible_keys: tuple[list[str], list[str]] = cast(
            tuple[list[str], list[str]],
            self._require_model().load_state_dict(
                cast(ModelState, state_dict),
                strict=strict
            )
        )
        missing_keys: list[str] = incompatible_keys[0]
        unexpected_keys: list[str] = incompatible_keys[1]
        if not strict:
            self._warn_incompatible_keys(missing_keys, unexpected_keys)
        if restore_optimizers:
            self._restore_optimizer_state(checkpoint)

    @override
    def remove(self, filepath: CheckpointPath, global_rank: int) -> None:
        if global_rank == 0:
            self._resolve_checkpoint_io().remove_checkpoint(filepath)

    @override
    def teardown(self) -> None:
        self._preloaded_path: str | None = None
        self._preloaded_checkpoint: CheckpointDict | None = None
        if self._checkpoint_io is not None:
            self._checkpoint_io.teardown()

    @override
    def attach_checkpoint_io(self, checkpoint_io: CheckpointIO) -> None:
        self._validate_checkpoint_io(checkpoint_io)
        self._checkpoint_io: CheckpointIO | None = checkpoint_io

    @property
    @override
    def configured_checkpoint_io(self) -> CheckpointIO | None:
        return self._configured_checkpoint_io

    def _read(
        self,
        checkpoint_path: CheckpointPath,
        weights_only: bool | None
    ) -> CheckpointDict:
        return self._resolve_checkpoint_io().load_checkpoint(
            checkpoint_path,
            weights_only=weights_only
        )

    def _restore_optimizer_state(
        self,
        checkpoint: Mapping[str, CheckpointValue]
    ) -> None:
        optimizer_states: CheckpointValue | None = checkpoint.get("optimizer_states")
        if isinstance(optimizer_states, list):
            if len(optimizer_states) != len(self._optimizers):
                raise KeyError(
                    "Checkpoint optimizer state count does not match the configured optimizers."
                )
            optimizer: torch.optim.Optimizer
            serialized_optimizer_state: CheckpointValue
            for optimizer, serialized_optimizer_state in zip(
                self._optimizers,
                optimizer_states,
                strict=True
            ):
                if not isinstance(serialized_optimizer_state, dict):
                    raise TypeError("Optimizer checkpoint state must be a dictionary.")
                optimizer.load_state_dict(cast(OptimizerState, serialized_optimizer_state))
            return
        if self._optimizers:
            raise KeyError("Checkpoint is missing resumable optimizer state.")

    def _require_model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("Checkpoint state requires a connected model.")
        return self._model

    def _resolve_checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:
            self._checkpoint_io: CheckpointIO | None = TorchCheckpointIO()
        elif isinstance(self._checkpoint_io, _WrappingCheckpointIO):
            self._checkpoint_io.checkpoint_io: CheckpointIO = TorchCheckpointIO()
        return self._checkpoint_io

    def _validate_checkpoint_io(self, checkpoint_io: CheckpointIO) -> None:
        if not isinstance(checkpoint_io, CheckpointIO):
            raise TypeError("checkpoint_io must be a CheckpointIO instance.")

    def _warn_incompatible_keys(
        self,
        missing_keys: list[str],
        unexpected_keys: list[str]
    ) -> None:
        if missing_keys:
            warnings.warn(
                "Found keys that are in the model state dict but not in the "
                f"checkpoint: {missing_keys}",
                UserWarning,
                stacklevel=3
            )
        if unexpected_keys:
            warnings.warn(
                "Found keys that are not in the model state dict but in the "
                f"checkpoint: {unexpected_keys}",
                UserWarning,
                stacklevel=3
            )
