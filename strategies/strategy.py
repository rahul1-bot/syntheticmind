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
from typing import TYPE_CHECKING

import torch
from torch import nn

from syntheticmind.utilities.types import Batch, ModelOutput, StepOutput

if TYPE_CHECKING:
    from syntheticmind.accelerators.accelerator import Accelerator

__all__: list[str] = ["Strategy"]


class Strategy(ABC):
    # Abstract execution strategy. Owns the placement of the model onto its
    # root device and the execution of each step kind; the trainer talks to
    # this boundary so single-device and multi-process execution stay
    # interchangeable.
    def __init__(
        self,
        accelerator: Accelerator | None = None,
        parallel_devices: list[torch.device] | None = None,
        num_nodes: int = 1
    ) -> None:
        # Starts with no model attached and the CPU as the provisional root
        # device until setup binds the real placement.
        self._model: nn.Module | None = None
        self._accelerator: Accelerator | None = accelerator
        self._parallel_devices: list[torch.device] = list(parallel_devices or [])
        self._num_nodes: int = num_nodes
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
        self, model: nn.Module, batch: Batch, batch_idx: int
    ) -> StepOutput:
        # Executes one validation step through the strategy's execution context.
        raise NotImplementedError

    @abstractmethod
    def test_step(
        self, model: nn.Module, batch: Batch, batch_idx: int
    ) -> StepOutput:
        # Executes one test step through the strategy's execution context.
        raise NotImplementedError

    @abstractmethod
    def predict_step(
        self, model: nn.Module, batch: Batch, batch_idx: int
    ) -> ModelOutput:
        # Executes one prediction step through the strategy's execution context.
        raise NotImplementedError

    @abstractmethod
    def backward(
        self,
        loss: torch.Tensor,
        *args: object,
        **kwargs: object
    ) -> None:
        # Runs the backward pass, including any gradient synchronization the
        # strategy requires.
        raise NotImplementedError

    def teardown(self) -> None:
        # Releases strategy-owned resources after a run; default is a no-op.
        pass

    def on_exception(self, exception: BaseException) -> None:
        # Hook for exception-time cleanup (such as tearing down process
        # groups); default is a no-op.
        pass

    def transfer_batch_to_device(self, batch: Batch, device: torch.device) -> Batch:
        # Moves a nested batch onto the target device via the shared recursive
        # transfer helper; imported lazily to keep interface imports light.
        from syntheticmind.utilities.data_transfer import move_data_to_device

        return move_data_to_device(batch, device)

    def barrier(self) -> None:
        # Cross-process synchronization point; a no-op outside distributed
        # execution so call sites need no guards.
        pass

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        # Broadcasts from the source rank; local execution returns the input.
        return tensor

    def reduce_boolean_decision(self, decision: bool) -> bool:
        # Agrees on a control decision across ranks; local execution returns
        # the local decision unchanged.
        return decision

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        # Gathers tensors from every rank; local execution returns the input.
        return tensor

    @property
    def is_distributed(self) -> bool:
        # Whether this strategy coordinates multiple processes; base default is
        # single-process.
        return False

    @property
    def accelerator(self) -> Accelerator | None:
        return self._accelerator

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

    def __repr__(self) -> str:
        # Compact identity line for logs and interactive debugging.
        return f"{type(self).__name__}(device={self._device})"
