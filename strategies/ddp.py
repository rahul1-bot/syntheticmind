from datetime import timedelta
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
        placed_model: nn.Module = model.to(device)
        wrapped_model: DistributedDataParallel = DistributedDataParallel(
            placed_model,
            device_ids=[device.index] if device.type == "cuda" else None
        )
        self.model: nn.Module | None = wrapped_model
        return wrapped_model

    @override
    def teardown(self) -> None:
        if isinstance(self.model, DistributedDataParallel):
            self.model: nn.Module | None = self.model.module
        super().teardown()

    def attach_layer_sync(self, layer_sync: LayerSync | None) -> None:
        if layer_sync is not None and not isinstance(layer_sync, LayerSync):
            raise TypeError("layer_sync must be a LayerSync instance or None.")
        self._layer_sync: LayerSync | None = layer_sync

    @property
    def layer_sync(self) -> LayerSync | None:
        return self._layer_sync

    @property
    def configured_layer_sync(self) -> LayerSync | None:
        return self._configured_layer_sync


class _DDPForwardRedirection(_ForwardRedirection):
    @override
    def _after_inner_forward(self, wrapper: nn.Module, module: Module) -> None:
        if isinstance(wrapper, DistributedDataParallel) and not module.automatic_optimization:
            wrapper.require_backward_grad_sync: bool = False

    @override
    def _after_outer_forward(self, wrapper: nn.Module, module: Module) -> None:
        if isinstance(wrapper, DistributedDataParallel) and not module.automatic_optimization:
            wrapper.require_backward_grad_sync: bool = True
