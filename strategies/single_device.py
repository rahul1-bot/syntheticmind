# This module:
# 1. Names the single-process, single-device Strategy policy
#
# Design decisions:
# - All behavior is inherited from Strategy because placement, full precision,
#   local collectives, and single-file checkpointing already have complete
#   single-device semantics
# - The concrete name remains public so Trainer configuration and client type
#   checks describe an execution policy rather than instantiating the base class
#
# Author: Rahul Sawhney

from typing import override

import torch

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.plugins.checkpoint_io import CheckpointIO
from syntheticmind.plugins.precision.precision import Precision
from syntheticmind.strategies.strategy import Strategy

__all__: list[str] = ["SingleDeviceStrategy"]


class SingleDeviceStrategy(Strategy):
    # Explicit local execution policy selected when one process owns one device.
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
