import warnings
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from datetime import timedelta
from typing import Literal, override

import torch
from torch import nn
from torch.distributed.constants import default_pg_timeout
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel

from syntheticmind.core.hooks import ModelHooks
from syntheticmind.core.module import Module
from syntheticmind.plugins.precision.precision import Precision
from syntheticmind.strategies.checkpoint import StrategyCheckpoint
from syntheticmind.strategies.model_parallel_checkpointing.checkpoint import (
    _ModelParallelCheckpoint
)
from syntheticmind.strategies.parallel import ParallelStrategy
from syntheticmind.exceptions import MisconfigurationError
from syntheticmind.strategies.model_initialization import _ModuleInitializer

__all__: list[str] = ["ModelParallelStrategy"]


class ModelParallelStrategy(ParallelStrategy):
    @override
    def __init__(
        self,
        data_parallel_size: Literal["auto"] | int = "auto",
        tensor_parallel_size: Literal["auto"] | int = "auto",
        save_distributed_checkpoint: bool = True,
        process_group_backend: str | None = None,
        timeout: timedelta | None = default_pg_timeout
    ) -> None:
        super().__init__(
            process_group_backend=process_group_backend,
            timeout=timeout,
            start_method="popen"
        )
        self._validate_size("data_parallel_size", data_parallel_size)
        self._validate_size("tensor_parallel_size", tensor_parallel_size)
        if type(save_distributed_checkpoint) is not bool:
            raise TypeError("`save_distributed_checkpoint` must be a bool.")
        self._data_parallel_size: Literal["auto"] | int = data_parallel_size
        self._tensor_parallel_size: Literal["auto"] | int = tensor_parallel_size
        self._device_mesh: DeviceMesh | None = None
        self.checkpoint: StrategyCheckpoint = _ModelParallelCheckpoint(
            "sharded" if save_distributed_checkpoint else "full",
            self.barrier,
            self.broadcast
        )

    @override
    def connect(self, model: nn.Module) -> None:
        super().connect(model)
        if self._device_mesh is not None:
            self._require_module()._device_mesh: DeviceMesh | None = self._device_mesh

    @override
    def setup_environment(self) -> None:
        super().setup_environment()
        self._setup_device_mesh()

    @override
    def setup(self, model: nn.Module, device: torch.device) -> nn.Module:
        if not isinstance(model, Module):
            raise TypeError("ModelParallelStrategy requires a SyntheticMind Module.")
        if type(model).configure_model is ModelHooks.configure_model:
            raise TypeError(
                "ModelParallelStrategy requires Module.configure_model() to apply "
                "its parallelization plan."
            )
        submodule: nn.Module
        for submodule in model.modules():
            if isinstance(submodule, FullyShardedDataParallel):
                raise TypeError("ModelParallelStrategy supports FSDP2, not FSDP1.")
        initializer: _ModuleInitializer = _ModuleInitializer(enabled=False)
        uninitialized_modules: set[str] = initializer.materialize_distributed(
            model,
            device
        )
        if uninitialized_modules and self.is_global_zero:
            warnings.warn(
                "Parameter initialization is incomplete for modules without "
                f"reset_parameters(): {', '.join(sorted(uninitialized_modules))}.",
                UserWarning,
                stacklevel=3
            )
        placed_model: nn.Module = model.to(device)
        self.connect(placed_model)
        self.barrier()
        return placed_model

    @override
    def teardown(self) -> None:
        self._clear_device_mesh()
        super().teardown()

    @override
    def on_exception(self, exception: BaseException) -> None:
        self._clear_device_mesh()
        super().on_exception(exception)

    @contextmanager
    @override
    def tensor_init_context(
        self,
        empty_init: bool | None = None
    ) -> Iterator[None]:
        if empty_init is not None and type(empty_init) is not bool:
            raise TypeError("`empty_init` must be a bool or None.")
        empty_context: AbstractContextManager[None] = (
            torch.device("meta") if empty_init else nullcontext()
        )
        with empty_context, self.precision_plugin.tensor_init_context():
            yield

    @override
    def sync_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer

    @override
    def attach_precision_plugin(self, precision_plugin: Precision) -> None:
        if not isinstance(precision_plugin, Precision):
            raise TypeError("precision_plugin must be a Precision instance.")
        supported_precisions: tuple[str, ...] = (
            "32-true",
            "bf16-mixed",
            "bf16-true",
            "16-true"
        )
        if precision_plugin.precision not in supported_precisions:
            raise MisconfigurationError(
                f"ModelParallelStrategy does not support {precision_plugin.precision!r}. "
                f"Choose one of {supported_precisions}."
            )
        self.precision_plugin: Precision = precision_plugin

    @property
    @override
    def distributed_sampler_kwargs(self) -> dict[str, int] | None:
        data_parallel_mesh: DeviceMesh = self.device_mesh["data_parallel"]
        return {
            "num_replicas": data_parallel_mesh.size(),
            "rank": data_parallel_mesh.get_local_rank()
        }

    @property
    @override
    def data_parallel_size(self) -> int:
        if self._device_mesh is not None:
            return self._device_mesh["data_parallel"].size()
        return (
            self.num_nodes
            if self._data_parallel_size == "auto"
            else self._data_parallel_size
        )

    @property
    @override
    def checkpoint_requires_all_ranks(self) -> bool:
        return True

    @property
    def device_mesh(self) -> DeviceMesh:
        if self._device_mesh is None:
            raise RuntimeError("The device mesh is unavailable before process setup.")
        return self._device_mesh

    def _setup_device_mesh(self) -> None:
        if self._data_parallel_size == "auto":
            data_parallel_size: int = self.num_nodes
        else:
            data_parallel_size: int = self._data_parallel_size
        if self._tensor_parallel_size == "auto":
            tensor_parallel_size: int = self.num_processes
        else:
            tensor_parallel_size: int = self._tensor_parallel_size
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
        self._require_module()._device_mesh: DeviceMesh | None = self._device_mesh

    def _clear_device_mesh(self) -> None:
        if self._module is not None:
            self._module._device_mesh: DeviceMesh | None = None
        self._device_mesh: DeviceMesh | None = None

    def _validate_size(
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
