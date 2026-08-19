# This module:
# 1. Defines the execution-strategy contract: environment setup, model placement,
#    the four step delegations, and the backward pass
# 2. Provides local single-process defaults for every collective operation so
#    non-distributed strategies inherit working no-ops
#
# Design decisions:
# - Collectives (barrier, broadcast, reduce_boolean_decision, all_gather) default
#   to identity or no-op behavior, so trainer and callback code calls them
#   unconditionally and only distributed strategies override them
# - The step methods take the model explicitly rather than reading self._model,
#   because a strategy may wrap the module (for example in a distributed
#   container) while the underlying step logic still belongs to the wrapped model
# - backward is strategy-owned because gradient synchronization differs between
#   local execution and distributed data parallel
# - teardown and on_exception default to no-ops so simple strategies opt into
#   cleanup only when they own resources
# - Batch transfer delegates to the shared recursive device-transfer helper,
#   imported lazily to keep this interface module light at import time
#
# Author: Rahul Sawhney

from __future__ import annotations

from abc import ABC, abstractmethod
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal, TypeVar, cast

import torch
from torch import nn

from syntheticmind.plugins.environments import ClusterEnvironment, LocalEnvironment
from syntheticmind.plugins.io import (
    CheckpointIO,
    CheckpointPath,
    TorchCheckpointIO,
    _WrappingCheckpointIO
)
from syntheticmind.plugins.layer_sync import LayerSync
from syntheticmind.precision.precision import Precision
from syntheticmind.utilities.model_initialization import _EmptyInit
from syntheticmind.utilities.types import (
    Batch,
    CheckpointDict,
    ModelOutput,
    StepOutput
)

if TYPE_CHECKING:
    from syntheticmind.accelerators.accelerator import Accelerator
    from syntheticmind.strategies.launcher import _ProcessLauncher

__all__: list[str] = ["Strategy"]

TBroadcast = TypeVar("TBroadcast")


class Strategy(ABC):
    # Abstract execution strategy. Owns the placement of the model onto its
    # root device and the execution of each step kind; the trainer talks to
    # this boundary so single-device and multi-process execution stay
    # interchangeable.
    def __init__(
        self,
        accelerator: Accelerator | None = None,
        parallel_devices: list[torch.device] | None = None,
        num_nodes: int = 1,
        cluster_environment: ClusterEnvironment | None = None,
        checkpoint_io: CheckpointIO | None = None,
        precision_plugin: Precision | None = None,
        layer_sync: LayerSync | None = None
    ) -> None:
        # Starts with no model attached and the CPU as the provisional root
        # device until setup binds the real placement.
        self._model: nn.Module | None = None
        self._module: nn.Module | None = None
        self._optimizers: list[torch.optim.Optimizer] = []
        self._accelerator: Accelerator | None = accelerator
        self._parallel_devices: list[torch.device] = list(parallel_devices or [])
        self._num_nodes: int = num_nodes
        self._cluster_environment: ClusterEnvironment = (
            cluster_environment or LocalEnvironment()
        )
        self._configured_cluster_environment: ClusterEnvironment | None = (
            cluster_environment
        )
        self._checkpoint_io: CheckpointIO | None = checkpoint_io
        self._configured_checkpoint_io: CheckpointIO | None = checkpoint_io
        self._precision_plugin: Precision = precision_plugin or Precision()
        self._configured_precision_plugin: Precision | None = precision_plugin
        self._layer_sync: LayerSync | None = layer_sync
        self._configured_layer_sync: LayerSync | None = layer_sync
        self._device: torch.device = (
            self._parallel_devices[0]
            if self._parallel_devices
            else torch.device("cpu")
        )

    def configure_topology(
        self,
        accelerator: Accelerator,
        parallel_devices: list[torch.device],
        num_nodes: int
    ) -> None:
        self._accelerator = accelerator
        self._parallel_devices = list(parallel_devices)
        self._num_nodes = num_nodes
        self._device = self._parallel_devices[0]
        self._configure_cluster_environment()

    def _configure_cluster_environment(self) -> None:
        if not self.is_distributed:
            return
        self._cluster_environment.validate_settings(
            self.num_processes,
            self.num_nodes
        )
        if self._cluster_environment.creates_processes_externally:
            return
        self._cluster_environment.set_world_size(
            self.num_nodes * self.num_processes
        )
        self._cluster_environment.set_global_rank(
            self._cluster_environment.node_rank() * self.num_processes
            + self._cluster_environment.local_rank()
        )

    def connect(self, model: nn.Module) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"`model` must be a torch.nn.Module, got {type(model).__name__}."
            )
        self._module = model
        self._model = model

    def validate_configuration(self) -> None:
        pass

    def save_checkpoint(
        self,
        checkpoint: CheckpointDict,
        filepath: CheckpointPath,
        storage_options: object | None = None
    ) -> None:
        if self.is_global_zero:
            self.checkpoint_io.save_checkpoint(
                checkpoint,
                filepath,
                storage_options
            )

    def load_checkpoint(
        self,
        checkpoint_path: CheckpointPath,
        weights_only: bool | None = None
    ) -> CheckpointDict:
        return self.checkpoint_io.load_checkpoint(
            checkpoint_path,
            weights_only=weights_only
        )

    def load_checkpoint_metadata(
        self,
        checkpoint_path: CheckpointPath,
        weights_only: bool | None = None
    ) -> CheckpointDict:
        return self.load_checkpoint(checkpoint_path, weights_only)

    def is_checkpoint_path(self, checkpoint_path: CheckpointPath) -> bool:
        from syntheticmind.utilities.filesystem import is_file

        return is_file(checkpoint_path)

    def model_state_dict(self) -> dict[str, object]:
        if self._module is None:
            raise RuntimeError("Strategy has no connected model.")
        return cast(dict[str, object], self._module.state_dict())

    def optimizer_state(
        self,
        optimizer: torch.optim.Optimizer
    ) -> dict[str, object]:
        return cast(dict[str, object], optimizer.state_dict())

    def load_model_state_dict(
        self,
        checkpoint: Mapping[str, object],
        strict: bool = True
    ) -> None:
        if self._module is None:
            raise RuntimeError("Strategy has no connected model.")
        state_dict: object | None = checkpoint.get(
            "state_dict",
            checkpoint.get("model_state_dict")
        )
        if not isinstance(state_dict, Mapping):
            raise KeyError("Checkpoint is missing a model `state_dict`.")
        incompatible_keys: object = self._module.load_state_dict(
            cast(Mapping[str, object], state_dict),
            strict=strict
        )
        if strict:
            return
        missing_keys: object = getattr(incompatible_keys, "missing_keys", [])
        unexpected_keys: object = getattr(incompatible_keys, "unexpected_keys", [])
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

    def load_optimizer_state_dict(
        self,
        checkpoint: Mapping[str, object]
    ) -> None:
        optimizer_states: object | None = checkpoint.get(
            "optimizer_states",
            checkpoint.get("optimizer_state_dicts")
        )
        if isinstance(optimizer_states, list):
            for optimizer, optimizer_state in zip(
                self.optimizers,
                optimizer_states,
                strict=False
            ):
                if not isinstance(optimizer_state, dict):
                    raise TypeError("Optimizer checkpoint state must be a dictionary.")
                optimizer.load_state_dict(optimizer_state)
            return
        optimizer_state: object | None = checkpoint.get("optimizer_state_dict")
        if isinstance(optimizer_state, dict) and self.optimizers:
            self.optimizers[0].load_state_dict(optimizer_state)
            return
        if self.optimizers:
            from loguru import logger as restore_log

            restore_log.warning(
                "Checkpoint missing optimizer_state_dict (partial/exception checkpoint); "
                "optimizer will start fresh."
            )

    def remove_checkpoint(self, filepath: CheckpointPath) -> None:
        if self.is_global_zero:
            self.checkpoint_io.remove_checkpoint(filepath)

    @abstractmethod
    def setup_environment(self) -> None:
        # Prepares process-level execution state (such as process groups)
        # before any model placement happens.
        raise NotImplementedError

    @abstractmethod
    def setup(self, model: nn.Module, device: torch.device) -> nn.Module:
        # Places the model on the target device, applies any strategy wrapping,
        # and returns the object the loops should drive.
        raise NotImplementedError

    @abstractmethod
    def training_step(
        self, model: nn.Module, batch: Batch, batch_idx: int
    ) -> StepOutput:
        # Executes one training step through the strategy's execution context.
        raise NotImplementedError

    @abstractmethod
    def validation_step(
        self, model: nn.Module, *args: object, **kwargs: object
    ) -> StepOutput:
        # Executes one validation step through the strategy's execution context.
        raise NotImplementedError

    @abstractmethod
    def test_step(
        self, model: nn.Module, *args: object, **kwargs: object
    ) -> StepOutput:
        # Executes one test step through the strategy's execution context.
        raise NotImplementedError

    @abstractmethod
    def predict_step(
        self, model: nn.Module, *args: object, **kwargs: object
    ) -> ModelOutput:
        # Executes one prediction step through the strategy's execution context.
        raise NotImplementedError

    def backward(
        self,
        closure_loss: torch.Tensor,
        optimizer: torch.optim.Optimizer | None = None,
        *args: object,
        **kwargs: object
    ) -> torch.Tensor:
        # Runs the backward pass, including any gradient synchronization the
        # strategy requires.
        return self._precision_backward(
            closure_loss,
            optimizer,
            *args,
            **kwargs
        )

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        closure: Callable[[], object],
        model: nn.Module | None = None,
        **kwargs: object
    ) -> object:
        from syntheticmind.core.module import Module

        resolved_model: nn.Module | None = model or self._module
        if not isinstance(resolved_model, Module):
            raise TypeError(
                f"Strategy optimizer_step requires Module, got {type(resolved_model).__name__}."
            )
        return self.precision_plugin.optimizer_step(
            optimizer,
            model=resolved_model,
            closure=closure,
            **kwargs
        )

    def teardown(self) -> None:
        # Releases strategy-owned resources after a run; default is a no-op.
        pass

    def teardown_checkpoint_io(self) -> None:
        if self._checkpoint_io is not None:
            self._checkpoint_io.teardown()

    def on_exception(self, exception: BaseException) -> None:
        # Hook for exception-time cleanup (such as tearing down process
        # groups); default is a no-op.
        pass

    def transfer_batch_to_device(self, batch: Batch, device: torch.device) -> Batch:
        # Moves a nested batch onto the target device via the shared recursive
        # transfer helper; imported lazily to keep interface imports light.
        from syntheticmind.utilities.data_transfer import move_data_to_device

        return move_data_to_device(batch, device)

    @contextmanager
    def tensor_init_context(
        self,
        empty_init: bool | None = None
    ) -> Iterator[None]:
        if empty_init is not None and type(empty_init) is not bool:
            raise TypeError("`empty_init` must be a bool or None.")
        with (
            _EmptyInit(enabled=bool(empty_init)),
            self.root_device,
            self.precision_plugin.tensor_init_context()
        ):
            yield

    def barrier(self) -> None:
        # Cross-process synchronization point; a no-op outside distributed
        # execution so call sites need no guards.
        pass

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        # Broadcasts from the source rank; local execution returns the input.
        return obj

    def reduce_boolean_decision(self, decision: bool) -> bool:
        # Agrees on a control decision across ranks; local execution returns
        # the local decision unchanged.
        return decision

    def all_gather(
        self,
        tensor: torch.Tensor,
        group: torch.distributed.ProcessGroup | None = None,
        sync_grads: bool = False
    ) -> torch.Tensor:
        # Gathers tensors from every rank; local execution returns the input.
        return tensor

    def sync_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        pass

    def pre_backward(self, closure_loss: torch.Tensor) -> None:
        del closure_loss

    def post_backward(self, closure_loss: torch.Tensor) -> None:
        del closure_loss

    def _forward_redirection(
        self,
        model: nn.Module,
        step_name: Literal[
            "training_step",
            "validation_step",
            "test_step",
            "predict_step"
        ],
        *step_args: object,
        **step_kwargs: object
    ) -> object:
        from syntheticmind.core.module import Module

        module: nn.Module | None = self._module
        if not isinstance(module, Module):
            raise TypeError(
                f"Strategy step execution requires Module, got {type(module).__name__}."
            )
        step: Callable[..., object] = cast(
            Callable[..., object],
            getattr(module, step_name)
        )
        if model is module:
            return step(*step_args, **step_kwargs)
        original_forward: Callable[..., object] = module.forward

        def redirected_forward(
            *args: object,
            **kwargs: object
        ) -> ModelOutput:
            module.forward = original_forward
            return cast(ModelOutput, step(*args, **kwargs))

        module.forward = redirected_forward
        try:
            return model(*step_args, **step_kwargs)
        finally:
            module.forward = original_forward

    def _precision_backward(
        self,
        closure_loss: torch.Tensor,
        optimizer: torch.optim.Optimizer | None,
        *args: object,
        **kwargs: object
    ) -> torch.Tensor:
        from syntheticmind.core.module import Module

        module: nn.Module | None = self._module
        if not isinstance(module, Module):
            raise TypeError(
                f"Strategy backward requires Module, got {type(module).__name__}."
            )
        self.pre_backward(closure_loss)
        prepared_loss: torch.Tensor = self.precision_plugin.pre_backward(
            closure_loss,
            module
        )
        self.precision_plugin.backward(
            prepared_loss,
            module,
            optimizer,
            *args,
            **kwargs
        )
        detached_loss: torch.Tensor = self.precision_plugin.post_backward(
            prepared_loss,
            module
        )
        self.post_backward(detached_loss)
        return detached_loss

    @property
    def is_distributed(self) -> bool:
        # Whether this strategy coordinates multiple processes; base default is
        # single-process.
        return False

    @property
    def restore_checkpoint_after_setup(self) -> bool:
        return False

    @property
    def distributed_sampler_kwargs(self) -> dict[str, int] | None:
        return None

    @property
    def launcher(self) -> _ProcessLauncher | None:
        return None

    @property
    def accelerator(self) -> Accelerator | None:
        return self._accelerator

    @property
    def cluster_environment(self) -> ClusterEnvironment:
        return self._cluster_environment

    @cluster_environment.setter
    def cluster_environment(
        self,
        cluster_environment: ClusterEnvironment
    ) -> None:
        if not isinstance(cluster_environment, ClusterEnvironment):
            raise TypeError(
                "cluster_environment must be a ClusterEnvironment instance, got "
                f"{type(cluster_environment).__name__}."
            )
        self._cluster_environment = cluster_environment
        self._configure_cluster_environment()
        self._configured_cluster_environment = cluster_environment

    def _attach_cluster_environment(
        self,
        cluster_environment: ClusterEnvironment
    ) -> None:
        if not isinstance(cluster_environment, ClusterEnvironment):
            raise TypeError(
                "cluster_environment must be a ClusterEnvironment instance, got "
                f"{type(cluster_environment).__name__}."
            )
        self._cluster_environment = cluster_environment
        self._configure_cluster_environment()

    @property
    def configured_cluster_environment(self) -> ClusterEnvironment | None:
        return self._configured_cluster_environment

    @property
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:
            self._checkpoint_io = TorchCheckpointIO()
        elif isinstance(self._checkpoint_io, _WrappingCheckpointIO):
            self._checkpoint_io.checkpoint_io = TorchCheckpointIO()
        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, checkpoint_io: CheckpointIO) -> None:
        if not isinstance(checkpoint_io, CheckpointIO):
            raise TypeError(
                f"checkpoint_io must be a CheckpointIO instance, got {type(checkpoint_io).__name__}."
            )
        self._checkpoint_io = checkpoint_io
        self._configured_checkpoint_io = checkpoint_io

    def _attach_checkpoint_io(self, checkpoint_io: CheckpointIO) -> None:
        if not isinstance(checkpoint_io, CheckpointIO):
            raise TypeError(
                f"checkpoint_io must be a CheckpointIO instance, got {type(checkpoint_io).__name__}."
            )
        self._checkpoint_io = checkpoint_io

    @property
    def configured_checkpoint_io(self) -> CheckpointIO | None:
        return self._configured_checkpoint_io

    @property
    def precision_plugin(self) -> Precision:
        return self._precision_plugin

    @precision_plugin.setter
    def precision_plugin(self, precision_plugin: Precision) -> None:
        if not isinstance(precision_plugin, Precision):
            raise TypeError(
                f"precision_plugin must be a Precision instance, got {type(precision_plugin).__name__}."
            )
        self._precision_plugin: Precision = precision_plugin
        self._configured_precision_plugin = precision_plugin

    def _attach_precision_plugin(self, precision_plugin: Precision) -> None:
        if not isinstance(precision_plugin, Precision):
            raise TypeError(
                f"precision_plugin must be a Precision instance, got {type(precision_plugin).__name__}."
            )
        self._precision_plugin = precision_plugin

    @property
    def configured_precision_plugin(self) -> Precision | None:
        return self._configured_precision_plugin

    @property
    def layer_sync(self) -> LayerSync | None:
        return self._layer_sync

    @layer_sync.setter
    def layer_sync(self, layer_sync: LayerSync | None) -> None:
        if layer_sync is not None and not isinstance(layer_sync, LayerSync):
            raise TypeError(
                f"layer_sync must be a LayerSync instance or None, got {type(layer_sync).__name__}."
            )
        self._layer_sync = layer_sync
        self._configured_layer_sync = layer_sync

    def _attach_layer_sync(self, layer_sync: LayerSync | None) -> None:
        if layer_sync is not None and not isinstance(layer_sync, LayerSync):
            raise TypeError(
                f"layer_sync must be a LayerSync instance or None, got {type(layer_sync).__name__}."
            )
        self._layer_sync = layer_sync

    @property
    def configured_layer_sync(self) -> LayerSync | None:
        return self._configured_layer_sync

    @property
    def optimizers(self) -> list[torch.optim.Optimizer]:
        return self._optimizers

    @optimizers.setter
    def optimizers(self, optimizers: list[torch.optim.Optimizer]) -> None:
        if not isinstance(optimizers, list) or not all(
            isinstance(optimizer, torch.optim.Optimizer)
            for optimizer in optimizers
        ):
            raise TypeError("`optimizers` must be a list of torch.optim.Optimizer instances.")
        self._optimizers = list(optimizers)

    @property
    def parallel_devices(self) -> list[torch.device]:
        return self._parallel_devices

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def num_processes(self) -> int:
        return len(self._parallel_devices) if self._parallel_devices else 1

    @property
    def global_rank(self) -> int:
        return 0

    @property
    def local_rank(self) -> int:
        return 0

    @property
    def node_rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1

    @property
    def is_global_zero(self) -> bool:
        return self.global_rank == 0

    @property
    def root_device(self) -> torch.device:
        # Device this strategy placed (or will place) the model on.
        return self._device

    @property
    def model(self) -> nn.Module | None:
        # The strategy-managed model object, or None before setup.
        return self._model

    @model.setter
    def model(self, model: nn.Module | None) -> None:
        self._model = model

    def __repr__(self) -> str:
        # Compact identity line for logs and interactive debugging.
        return f"{type(self).__name__}(device={self._device})"
