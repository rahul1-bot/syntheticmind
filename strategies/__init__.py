# Package initializer for the execution strategies; modules are imported by full path.
#
# Author: Rahul Sawhney

from syntheticmind.strategies.ddp import DDPStrategy
from syntheticmind.strategies.model_parallel import ModelParallelStrategy
from syntheticmind.strategies.single_device import SingleDeviceStrategy
from syntheticmind.strategies.strategy import Strategy

__all__: list[str] = [
    "DDPStrategy",
    "ModelParallelStrategy",
    "SingleDeviceStrategy",
    "Strategy"
]
