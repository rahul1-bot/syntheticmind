# This module:
# 1. Exposes single-device, shared parallel, DDP, and model-parallel execution
#    policies through their stateful Strategy hierarchy
#
# Design decisions:
# - Public exports name recognized ML execution concepts; launcher and checkpoint
#   collaborators remain internal to the selected Strategy
# - Strategy is the common behavior owner rather than an ABC or registry, and
#   concrete policies implement only topology-specific deltas
#
# Author: Rahul Sawhney

from syntheticmind.strategies.ddp import DDPStrategy
from syntheticmind.strategies.model_parallel import ModelParallelStrategy
from syntheticmind.strategies.parallel import ParallelStrategy
from syntheticmind.strategies.single_device import SingleDeviceStrategy
from syntheticmind.strategies.strategy import Strategy

__all__: list[str] = [
    "DDPStrategy",
    "ModelParallelStrategy",
    "ParallelStrategy",
    "SingleDeviceStrategy",
    "Strategy"
]
