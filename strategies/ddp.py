# This module:
# 1. Implements DistributedDataParallel model wrapping over ParallelStrategy
# 2. Connects reversible layer synchronization and DDP no-sync accumulation
# 3. Specializes wrapped-forward redirection for manual optimization
#
# Design decisions:
# - DDP inherits process groups, ranks, launchers, collectives, samplers, and tail
#   synchronization from ParallelStrategy and owns only DDP-specific model policy
# - setup places the client model before wrapping and supplies CUDA device IDs
#   only for CUDA, preserving the CPU Gloo path
# - Stage steps execute through the DDP wrapper's __call__ path via inherited
#   redirection so reducer hooks are armed before client forward and backward
# - defer_backward_sync exposes DDP.no_sync to automatic accumulation and manual
#   managed optimizer contexts without leaking wrapper inspection into loops
# - LayerSync remains a separate reversible plugin applied around DDP wrapping;
#   teardown unwraps the client model before outer lifecycle restoration
# - Manual optimization resets DDP's backward synchronization marker around the
#   wrapper boundary because client code chooses its own backward operations
#
# Author: Rahul Sawhney

from datetime import timedelta
from contextlib import AbstractContextManager
from typing import override

import torch
from torch import nn
from torch.distributed.constants import default_pg_timeout
from torch.nn.parallel import DistributedDataParallel

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.core.module import Module
from syntheticmind.plugins.environments import ClusterEnvironment
from syntheticmind.plugins.checkpoint_io import CheckpointIO
from syntheticmind.plugins.layer_sync import LayerSync
from syntheticmind.plugins.precision.precision import Precision
from syntheticmind.strategies.launchers.launcher import ProcessStartMethod
from syntheticmind.strategies.parallel import ParallelStrategy
from syntheticmind.strategies.strategy import _ForwardRedirection

__all__: list[str] = ["DDPStrategy"]


class DDPStrategy(ParallelStrategy):
    # Replicated data-parallel execution with one process and model replica per
    # selected device. Clients choose ``"ddp"``, ``"ddp_spawn"``, or
    # ``"ddp_fork"`` through Trainer, or supply a configured DDPStrategy.
    @override
    def __init__(
        self,
        accelerator: Accelerator | None = None,
        parallel_devices: list[torch.device] | None = None,
        cluster_environment: ClusterEnvironment | None = None,
        checkpoint_io: CheckpointIO | None = None,
        precision_plugin: Precision | None = None,
        process_group_backend: str | None = None,
        timeout: timedelta | None = default_pg_timeout,
        start_method: ProcessStartMethod = "popen",
        layer_sync: LayerSync | None = None
    ) -> None:
        # Binds optional execution collaborators, rendezvous policy, local start
        # method, and reversible layer synchronization before topology resolution.
        super().__init__(
            accelerator,
            parallel_devices,
            cluster_environment,
            checkpoint_io,
            precision_plugin,
            process_group_backend,
            timeout,
            start_method
        )
        self._layer_sync: LayerSync | None = layer_sync
        self._configured_layer_sync: LayerSync | None = layer_sync
        self._forward_redirection: _ForwardRedirection = _DDPForwardRedirection()

    @override
    def setup(self, model: nn.Module, device: torch.device) -> nn.Module:
        # Places and wraps the model in DistributedDataParallel for this process.
        placed_model: nn.Module = model.to(device)
        wrapped_model: DistributedDataParallel = DistributedDataParallel(
            placed_model,
            device_ids=[device.index] if device.type == "cuda" else None
        )
        self.model: nn.Module | None = wrapped_model
        return wrapped_model

    @override
    def teardown(self) -> None:
        # Returns ownership to the underlying client model before shared Strategy
        # teardown; process-group lifetime remains with ParallelStrategy.
        if isinstance(self.model, DistributedDataParallel):
            self.model: nn.Module | None = self.model.module
        super().teardown()

    def attach_layer_sync(self, layer_sync: LayerSync | None) -> None:
        # Installs the validated reversible layer policy selected by the Trainer
        # connector without changing the originally configured plugin identity.
        if layer_sync is not None and not isinstance(layer_sync, LayerSync):
            raise TypeError("layer_sync must be a LayerSync instance or None.")
        self._layer_sync: LayerSync | None = layer_sync

    @override
    def defer_backward_sync(self) -> AbstractContextManager[None]:
        # Returns DDP.no_sync for non-boundary accumulation or explicit manual
        # optimizer contexts and fails before model wrapping.
        if not isinstance(self.model, DistributedDataParallel):
            raise RuntimeError("DDP gradient synchronization requires a wrapped model.")
        return self.model.no_sync()

    @property
    def layer_sync(self) -> LayerSync | None:
        # Returns the live reversible layer policy attached for this run.
        return self._layer_sync

    @property
    def configured_layer_sync(self) -> LayerSync | None:
        # Returns the client-configured identity used for connector conflict
        # detection independently from later attachment.
        return self._configured_layer_sync


class _DDPForwardRedirection(_ForwardRedirection):
    # DDP-specific wrapper-state adjustment around manual-optimization stage
    # redirection. Automatic optimization leaves DDP defaults untouched.
    @override
    def _after_inner_forward(self, wrapper: nn.Module, module: Module) -> None:
        # Prevents the surrounding wrapper boundary from retaining automatic
        # backward-sync intent after a client-driven manual step.
        if isinstance(wrapper, DistributedDataParallel) and not module.automatic_optimization:
            wrapper.require_backward_grad_sync: bool = False

    @override
    def _after_outer_forward(self, wrapper: nn.Module, module: Module) -> None:
        # Restores DDP synchronization intent for the next independently managed
        # step after the outer wrapper call completes.
        if isinstance(wrapper, DistributedDataParallel) and not module.automatic_optimization:
            wrapper.require_backward_grad_sync: bool = True
