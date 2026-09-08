# This module:
# 1. Implements named data- and tensor-parallel device meshes for client-owned
#    tensor parallelism, composable FSDP2, or hybrid model plans
# 2. Owns distributed parameter materialization and full or sharded
#    model-parallel checkpoint representation
#
# Design decisions:
# - ModelParallelStrategy is an independent ParallelStrategy rather than a DDP
#   subclass because tensor-parallel and FSDP2 modules do not share replicated
#   DDP wrapping identity
# - Default execution preserves the two-dimensional DP/TP mesh. Extended CP/PP
#   axes require explicit DP/TP sizes and client-owned manual execution plans
# - Client code applies the actual parallelization plan in configure_model where
#   Module.device_mesh is available; Strategy supplies topology and lifecycle but
#   never guesses which layers should shard
# - Legacy FullyShardedDataParallel wrapping is rejected because the supported
#   path is composable FSDP2 over the client plan
# - Empty construction may place tensors on meta, after which materialization
#   allocates on the rank device and invokes each module's reset_parameters;
#   modules without a reset contract are reported explicitly
# - Model-parallel checkpoint operations require collective participation from
#   every rank even when only rank zero publishes metadata or a full artifact
# - Gradient synchronization belongs to the client's composable plan, so the DDP
#   tail-gradient algorithm is intentionally absent
#
# Author: Rahul Sawhney

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
from syntheticmind.strategies.model_parallel_checkpointing.checkpoint_io import (
    ModelParallelCheckpointWrite
)
from syntheticmind.strategies.parallel import ParallelStrategy
from syntheticmind.primitives.exceptions import MisconfigurationError
from syntheticmind.strategies.model_initialization import _ModuleInitializer

__all__: list[str] = ["ModelParallelStrategy"]


class ModelParallelStrategy(ParallelStrategy):
    # Researcher-controlled model-parallel execution policy. Clients must
    # override Module.configure_model and apply tensor-parallel, composable FSDP2,
    # or hybrid transformations against the named device mesh.
    #
    # Example::
    #
    #     from typing import override
    #
    #     from torch import nn
    #     from torch.distributed.tensor.parallel import (
    #         ColwiseParallel,
    #         parallelize_module
    #     )
    #
    #     from syntheticmind import Module
    #
    #     class ParallelEncoder(Module):
    #         def __init__(self) -> None:
    #             super().__init__()
    #             self.projection: nn.Linear = nn.Linear(1024, 4096)
    #
    #         @override
    #         def configure_model(self) -> None:
    #             parallelize_module(
    #                 self,
    #                 self.device_mesh["tensor_parallel"],
    #                 {"projection": ColwiseParallel()}
    #             )
    @override
    def __init__(
        self,
        data_parallel_size: Literal["auto"] | int = "auto",
        tensor_parallel_size: Literal["auto"] | int = "auto",
        save_distributed_checkpoint: bool = True,
        process_group_backend: str | None = None,
        timeout: timedelta | None = default_pg_timeout,
        *,
        context_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        checkpoint_write: ModelParallelCheckpointWrite = "synchronous"
    ) -> None:
        # Validates mesh dimensions and checkpoint format, selects collective
        # process launch, and installs the model-parallel checkpoint owner.
        super().__init__(
            process_group_backend=process_group_backend,
            timeout=timeout,
            start_method="popen"
        )
        self._validate_size("data_parallel_size", data_parallel_size)
        self._validate_size("tensor_parallel_size", tensor_parallel_size)
        if type(context_parallel_size) is not int or context_parallel_size < 1:
            raise ValueError("context_parallel_size must be a positive integer.")
        if type(pipeline_parallel_size) is not int or pipeline_parallel_size < 1:
            raise ValueError("pipeline_parallel_size must be a positive integer.")
        if type(save_distributed_checkpoint) is not bool:
            raise TypeError("`save_distributed_checkpoint` must be a bool.")
        if checkpoint_write not in {"synchronous", "asynchronous"}:
            raise ValueError(
                f"Unsupported model-parallel checkpoint write mode: {checkpoint_write!r}."
            )
        if not save_distributed_checkpoint and checkpoint_write == "asynchronous":
            raise ValueError(
                "Asynchronous model-parallel checkpointing requires sharded checkpoints."
            )
        self._data_parallel_size: Literal["auto"] | int = data_parallel_size
        self._tensor_parallel_size: Literal["auto"] | int = tensor_parallel_size
        self._context_parallel_size: int = context_parallel_size
        self._pipeline_parallel_size: int = pipeline_parallel_size
        self._validate_extended_mesh_configuration()
        self._device_mesh: DeviceMesh | None = None
        self.checkpoint: StrategyCheckpoint = _ModelParallelCheckpoint(
            "sharded" if save_distributed_checkpoint else "full",
            checkpoint_write,
            self.barrier,
            self.broadcast
        )

    @override
    def connect(self, model: nn.Module) -> None:
        # Connects the model and publishes an already initialized device mesh to
        # the client Module when stage ordering created the mesh first.
        super().connect(model)
        if self._device_mesh is not None:
            self._require_module()._device_mesh: DeviceMesh | None = self._device_mesh

    @override
    def setup_environment(self) -> None:
        # Initializes shared distributed environment, then creates the named
        # device mesh before client model configuration.
        super().setup_environment()
        self._setup_device_mesh()

    @override
    def setup(self, model: nn.Module, device: torch.device) -> nn.Module:
        # Requires a client parallelization plan, rejects FSDP1, materializes meta
        # tensors, places remaining state, reconnects checkpoint ownership, and
        # synchronizes ranks before execution.
        if not isinstance(model, Module):
            raise TypeError("ModelParallelStrategy requires a SyntheticMind Module.")
        if (self._context_parallel_size > 1 or self._pipeline_parallel_size > 1) and model.automatic_optimization:
            raise MisconfigurationError(
                "Extended CP/PP meshes require client-owned manual optimization, "
                "attention/gradient synchronization, and pipeline scheduling."
            )
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
        execution_model: nn.Module = self._compile_model(placed_model)
        self.model: nn.Module | None = execution_model
        self.barrier()
        return execution_model

    @override
    def teardown(self) -> None:
        # Removes mesh references from Strategy and Module while preserving shared
        # process-group lifetime for sequential distributed stages.
        self._clear_device_mesh()
        super().teardown()

    @override
    def on_exception(self, exception: BaseException) -> None:
        # Drain independent checkpoint workers before destroying training groups.
        try:
            self.checkpoint.teardown()
        except BaseException as cleanup_error:
            exception.add_note(f"Checkpoint drain also failed: {cleanup_error}")
        self._clear_device_mesh()
        super().on_exception(exception)

    @contextmanager
    @override
    def tensor_init_context(
        self,
        empty_init: bool | None = None
    ) -> Iterator[None]:
        # Selects meta allocation only when empty_init is true and composes it
        # with the active precision construction dtype.
        if empty_init is not None and type(empty_init) is not bool:
            raise TypeError("`empty_init` must be a bool or None.")
        empty_context: AbstractContextManager[None] = (
            torch.device("meta") if empty_init else nullcontext()
        )
        with empty_context, self.precision_plugin.tensor_init_context():
            yield

    @override
    def sync_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        # Performs no explicit tail all-reduce because DTensor and composable
        # sharding own gradient placement and synchronization semantics.
        del optimizer

    @override
    def attach_precision_plugin(self, precision_plugin: Precision) -> None:
        # Restricts numerical policies to those verified with DTensor and
        # composable sharding before attaching the plugin.
        if not isinstance(precision_plugin, Precision):
            raise TypeError("precision_plugin must be a Precision instance.")
        supported_precisions: tuple[str, ...] = (
            "32-true",
            "bf16-mixed",
            "bf16-true",
            "16-true",
            "torchao-float8"
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
        # Samples over the data-parallel mesh axis only; tensor-parallel ranks
        # cooperating on the same samples do not create extra replicas.
        data_parallel_mesh: DeviceMesh = self.device_mesh["data_parallel"]
        return {
            "num_replicas": data_parallel_mesh.size(),
            "rank": data_parallel_mesh.get_local_rank()
        }

    @property
    @override
    def data_parallel_size(self) -> int:
        # Resolved data-axis size from the live mesh or pre-setup configuration.
        if self._device_mesh is not None:
            return self._device_mesh["data_parallel"].size()
        return (
            self.num_nodes
            if self._data_parallel_size == "auto"
            else self._data_parallel_size
        )

    @property
    def context_parallel_size(self) -> int:
        return self._context_parallel_size

    @property
    def pipeline_parallel_size(self) -> int:
        return self._pipeline_parallel_size

    @property
    @override
    def checkpoint_requires_all_ranks(self) -> bool:
        # Full and sharded distributed state APIs require every model rank to
        # participate even when rank zero owns final publication.
        return True

    @property
    def device_mesh(self) -> DeviceMesh:
        # Named mesh available after process setup and during configure_model.
        if self._device_mesh is None:
            raise RuntimeError("The device mesh is unavailable before process setup.")
        return self._device_mesh

    def _setup_device_mesh(self) -> None:
        # Resolves automatic axis sizes, validates rectangular world coverage,
        # initializes the mesh, and attaches it to the connected Module.
        if self._data_parallel_size == "auto":
            data_parallel_size: int = self.num_nodes
        else:
            data_parallel_size: int = self._data_parallel_size
        if self._tensor_parallel_size == "auto":
            tensor_parallel_size: int = self.num_processes
        else:
            tensor_parallel_size: int = self._tensor_parallel_size
        context_parallel_size: int = self._context_parallel_size
        pipeline_parallel_size: int = self._pipeline_parallel_size
        mesh_world_size: int = (
            data_parallel_size
            * tensor_parallel_size
            * context_parallel_size
            * pipeline_parallel_size
        )
        if mesh_world_size != self.world_size:
            raise RuntimeError(
                f"data_parallel_size={data_parallel_size}, "
                f"tensor_parallel_size={tensor_parallel_size}, "
                f"context_parallel_size={context_parallel_size}, and "
                f"pipeline_parallel_size={pipeline_parallel_size} must multiply to "
                f"world_size={self.world_size}."
            )
        shape: tuple[int, ...] = (data_parallel_size, tensor_parallel_size)
        names: tuple[str, ...] = ("data_parallel", "tensor_parallel")
        if context_parallel_size > 1 or pipeline_parallel_size > 1:
            shape: tuple[int, ...] = (*shape, context_parallel_size, pipeline_parallel_size)
            names: tuple[str, ...] = (*names, "context_parallel", "pipeline_parallel")
        self._device_mesh: DeviceMesh | None = init_device_mesh(self.root_device.type, shape, mesh_dim_names=names)
        self._require_module()._device_mesh: DeviceMesh | None = self._device_mesh

    def _clear_device_mesh(self) -> None:
        # Removes Module and Strategy references without destroying the process
        # group that backs live distributed tensors.
        if self._module is not None:
            self._module._device_mesh: DeviceMesh | None = None
        self._device_mesh: DeviceMesh | None = None

    def _validate_extended_mesh_configuration(self) -> None:
        if self._context_parallel_size == 1 and self._pipeline_parallel_size == 1:
            return
        if self._data_parallel_size == "auto" or self._tensor_parallel_size == "auto":
            raise ValueError(
                "Context or pipeline parallel axes require explicit data_parallel_size "
                "and tensor_parallel_size so SyntheticMind does not guess topology."
            )

    def _validate_size(
        self,
        name: str,
        value: Literal["auto"] | int
    ) -> None:
        # Accepts ``"auto"`` or a positive non-bool integer for one mesh axis.
        if value == "auto":
            return
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"`{name}` must be 'auto' or a positive integer.")
        if value < 1:
            raise ValueError(f"`{name}` must be greater than zero.")
