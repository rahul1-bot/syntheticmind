from typing import override

import torch

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.plugins.io import CheckpointIO
from syntheticmind.precision.precision import Precision
from syntheticmind.strategies.strategy import Strategy

__all__: list[str] = ["SingleDeviceStrategy"]


class SingleDeviceStrategy(Strategy):
    @override
    def __init__(
        self,
        accelerator: Accelerator | None = None,
        parallel_devices: list[torch.device] | None = None,
        checkpoint_io: CheckpointIO | None = None,
        precision_plugin: Precision | None = None
    ) -> None:
        super().__init__(
            accelerator,
            parallel_devices,
            checkpoint_io,
            precision_plugin
        )
