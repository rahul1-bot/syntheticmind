# This module:
# 1. Exposes stateful execution policies and their explicit configuration contracts
#
# Design decisions:
# - Public names describe actual ML algorithms; checkpoint resources remain private
# - Optional TorchAO is not imported by ordinary Strategy construction
#
# Author: Rahul Sawhney

from syntheticmind.strategies.compiler import CompilationConfiguration, TorchCompiler
from syntheticmind.strategies.context_parallel import ContextParallelStrategy
from syntheticmind.strategies.ddp import DDPStrategy
from syntheticmind.strategies.model_parallel import ModelParallelStrategy
from syntheticmind.strategies.parallel import ParallelStrategy
from syntheticmind.strategies.parallel_configuration import (
    ContextParallelConfiguration, PipelineParallelConfiguration, PipelineStageShape
)
from syntheticmind.strategies.pipeline_parallel import PipelineParallelStrategy
from syntheticmind.strategies.single_device import SingleDeviceStrategy
from syntheticmind.strategies.strategy import Strategy

__all__: list[str] = [
    "CompilationConfiguration",
    "ContextParallelConfiguration",
    "ContextParallelStrategy",
    "DDPStrategy",
    "ModelParallelStrategy",
    "ParallelStrategy",
    "PipelineParallelConfiguration",
    "PipelineParallelStrategy",
    "PipelineStageShape",
    "SingleDeviceStrategy",
    "Strategy",
    "TorchCompiler"
]
