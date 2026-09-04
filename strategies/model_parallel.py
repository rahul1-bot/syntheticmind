from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal, override

import torch
from torch import nn
from torch.distributed.constants import default_pg_timeout
from torch.distributed.device_mesh import init_device_mesh

from syntheticmind.core.hooks import ModelHooks
from syntheticmind.core.module import Module
from syntheticmind.plugins.io import CheckpointIO, CheckpointPath
from syntheticmind.strategies.ddp import DDPStrategy
from syntheticmind.strategies.model_parallel_checkpointing.checkpoint_io import (
    _ModelParallelCheckpointIO
)
from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.filesystem import normalize_path
from syntheticmind.utilities.model_helpers import is_overridden
from syntheticmind.utilities.model_initialization import _ModuleInitializer
from syntheticmind.utilities.types import CheckpointDict

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

__all__: list[str] = ["ModelParallelStrategy"]


class ModelParallelStrategy(DDPStrategy):
    def __init__(
        self,
        data_parallel_size: Literal["auto"] | int = "auto",
        tensor_parallel_size: Literal["auto"] | int = "auto",
        save_distributed_checkpoint: bool = True,
        process_group_backend: str | None = None,
        timeout: timedelta | None = default_pg_timeout
    ) -> None:
        self._validate_parallel_size("data_parallel_size", data_parallel_size)
        self._validate_parallel_size("tensor_parallel_size", tensor_parallel_size)
        if type(save_distributed_checkpoint) is not bool:
            raise TypeError("`save_distributed_checkpoint` must be a bool.")
        super().__init__(
            process_group_backend=process_group_backend,
            timeout=timeout,
            start_method="popen"
        )
        self._data_parallel_size: Literal["auto"] | int = data_parallel_size
        self._tensor_parallel_size: Literal["auto"] | int = tensor_parallel_size
        self._model_parallel_checkpoint_io: _ModelParallelCheckpointIO = (
            _ModelParallelCheckpointIO(
                "sharded" if save_distributed_checkpoint else "full"
            )
        )
        self._device_mesh: DeviceMesh | None = None
        self._preloaded_checkpoint_path: str | None = None
        self._preloaded_checkpoint: CheckpointDict | None = None

    @override
    def connect(self, model: nn.Module) -> None:
        super().connect(model)
        if self._device_mesh is not None and isinstance(model, Module):
            model._device_mesh: DeviceMesh | None = self._device_mesh

    @property
    def device_mesh(self) -> DeviceMesh:
        if self._device_mesh is None:
            raise RuntimeError(
                "Accessing the device mesh before processes have initialized is not allowed."
            )
        return self._device_mesh

    @property
    @override
    def distributed_sampler_kwargs(self) -> dict[str, int]:
        data_parallel_mesh: DeviceMesh = self.device_mesh["data_parallel"]
        return {
            "num_replicas": data_parallel_mesh.size(),
            "rank": data_parallel_mesh.get_local_rank()
        }

    @property
    @override
    def restore_checkpoint_after_setup(self) -> bool:
        return True

    @property
    @override
    def checkpoint_io(self) -> CheckpointIO:
        raise NotImplementedError(
            "ModelParallelStrategy owns distributed checkpointing and does not use CheckpointIO."
        )

    @override
    def validate_configuration(self) -> None:
        if self._checkpoint_io is not None:
            raise MisconfigurationError(
                "ModelParallelStrategy does not support a CheckpointIO plugin."
            )
        if self.layer_sync is not None:
            raise MisconfigurationError(
                "ModelParallelStrategy does not support a LayerSync plugin or sync_batchnorm."
            )
        supported_precisions: tuple[str, ...] = (
            "32-true",
            "bf16-mixed",
            "bf16-true",
            "16-true"
        )
        if self.precision_plugin.precision not in supported_precisions:
            raise MisconfigurationError(
                f"ModelParallelStrategy does not support precision={self.precision_plugin.precision!r}. "
                f"Choose one of {supported_precisions}."
            )

    @override
    def setup_environment(self) -> None:
        super().setup_environment()
        data_parallel_size: int = (
            self.num_nodes
            if self._data_parallel_size == "auto"
            else self._data_parallel_size
        )
        tensor_parallel_size: int = (
            self.num_processes
            if self._tensor_parallel_size == "auto"
            else self._tensor_parallel_size
        )
        if data_parallel_size * tensor_parallel_size != self.world_size:
            raise RuntimeError(
                f"data_parallel_size={data_parallel_size} and "
                f"tensor_parallel_size={tensor_parallel_size} must multiply to "
                f"world_size={self.world_size}."
            )
        self._device_mesh: DeviceMesh | None = init_device_mesh(
            self.root_device.type,
            (data_parallel_size, tensor_parallel_size),
            mesh_dim_names=("data_parallel", "tensor_parallel")
        )
        if not isinstance(self._module, Module):
            raise RuntimeError("ModelParallelStrategy requires a connected Module.")
        self._module._device_mesh: DeviceMesh | None = self._device_mesh

    @override
    def setup(self, model: nn.Module, device: torch.device) -> nn.Module:
        from torch.distributed.fsdp import FullyShardedDataParallel

        if not isinstance(model, Module):
            raise TypeError("ModelParallelStrategy requires a SyntheticMind Module.")
        if not is_overridden("configure_model", model, ModelHooks):
            raise TypeError(
                "ModelParallelStrategy requires the Module to override configure_model() "
                "and apply its parallelization plan there."
            )
        if any(
            isinstance(submodule, FullyShardedDataParallel)
            for submodule in model.modules()
        ):
            raise TypeError(
                "ModelParallelStrategy supports composable FSDP2, not "
                "torch.distributed.fsdp.FullyShardedDataParallel."
            )
        module_initializer: _ModuleInitializer = _ModuleInitializer(enabled=False)
        uninitialized_modules: set[str] = module_initializer.materialize_distributed(
            model,
            device
        )
        if uninitialized_modules and self.is_global_zero:
            warnings.warn(
                "Parameter initialization is incomplete because these modules do not "
                f"define reset_parameters(): {', '.join(sorted(uninitialized_modules))}.",
                UserWarning,
                stacklevel=3
            )
        placed_model: nn.Module = model.to(device)
        self._device: torch.device = device
        self.connect(placed_model)
        self.barrier()
        return placed_model

    @contextmanager
    @override
    def tensor_init_context(
        self,
        empty_init: bool | None = None
    ) -> Iterator[None]:
        if empty_init is not None and type(empty_init) is not bool:
            raise TypeError("`empty_init` must be a bool or None.")
        empty_context: AbstractContextManager[object] = (
            torch.device("meta") if empty_init else nullcontext()
        )
        with empty_context, self.precision_plugin.tensor_init_context():
            yield

    @override
    def sync_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer

    @override
    def model_state_dict(self) -> dict[str, object]:
        if self.model is None:
            raise RuntimeError("ModelParallelStrategy has no connected model.")
        return self._model_parallel_checkpoint_io.model_state(self.model)

    @override
    def optimizer_state(
        self,
        optimizer: torch.optim.Optimizer
    ) -> dict[str, object]:
        if self.model is None:
            raise RuntimeError("ModelParallelStrategy has no connected model.")
        return self._model_parallel_checkpoint_io.optimizer_state(
            self.model,
            optimizer
        )

    @override
    def load_model_state_dict(
        self,
        checkpoint: Mapping[str, object],
        strict: bool = True
    ) -> None:
        del checkpoint, strict

    @override
    def load_optimizer_state_dict(
        self,
        checkpoint: Mapping[str, object]
    ) -> None:
        del checkpoint

    @override
    def save_checkpoint(
        self,
        checkpoint: CheckpointDict,
        filepath: CheckpointPath,
        storage_options: object | None = None
    ) -> None:
        if storage_options is not None:
            raise TypeError(
                "ModelParallelStrategy.save_checkpoint(..., storage_options=...) is "
                "unsupported because the strategy does not use CheckpointIO."
            )
        path: str = normalize_path(self.broadcast(filepath))
        self._model_parallel_checkpoint_io.save(
            checkpoint,
            path,
            self.global_rank,
            self.barrier,
            self.broadcast
        )

    @override
    def load_checkpoint_metadata(
        self,
        checkpoint_path: CheckpointPath,
        weights_only: bool | None = None
    ) -> CheckpointDict:
        path: str = normalize_path(self.broadcast(checkpoint_path))
        checkpoint: CheckpointDict = self._model_parallel_checkpoint_io.load_metadata(
            path,
            weights_only
        )
        if Path(path).is_file():
            self._preloaded_checkpoint_path: str | None = path
            self._preloaded_checkpoint: CheckpointDict | None = checkpoint
        return checkpoint

    @override
    def load_checkpoint(
        self,
        checkpoint_path: CheckpointPath,
        weights_only: bool | None = None
    ) -> CheckpointDict:
        if self.model is None or not isinstance(self._module, Module):
            raise RuntimeError(
                "ModelParallelStrategy can load model state only after setup."
            )
        path: str = normalize_path(self.broadcast(checkpoint_path))
        preloaded_checkpoint: CheckpointDict | None = (
            self._preloaded_checkpoint
            if self._preloaded_checkpoint_path == path
            else None
        )
        self._preloaded_checkpoint_path: str | None = None
        self._preloaded_checkpoint: CheckpointDict | None = None
        return self._model_parallel_checkpoint_io.load(
            path,
            self.model,
            self.optimizers,
            self._module.strict_loading,
            weights_only,
            preloaded_checkpoint
        )

    @override
    def is_checkpoint_path(self, checkpoint_path: CheckpointPath) -> bool:
        return self._model_parallel_checkpoint_io.is_checkpoint(checkpoint_path)

    @override
    def remove_checkpoint(self, filepath: CheckpointPath) -> None:
        if self.is_global_zero:
            self._model_parallel_checkpoint_io.remove(filepath)

    @override
    def on_exception(self, exception: BaseException) -> None:
        self._device_mesh: DeviceMesh | None = None
        if isinstance(self._module, Module):
            self._module._device_mesh: DeviceMesh | None = None
        super().on_exception(exception)

    def _validate_parallel_size(
        self,
        name: str,
        value: Literal["auto"] | int
    ) -> None:
        if value == "auto":
            return
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"`{name}` must be 'auto' or a positive integer.")
        if value < 1:
            raise ValueError(f"`{name}` must be greater than zero.")
