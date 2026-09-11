# This module:
# 1. Exposes PyTorchProfiler as the public operator-profiling policy
#
# Design decisions:
# - Schedule and module-recording collaborators remain private because client
#   configuration enters through PyTorchProfiler and torch.profiler callables
#
# Author: Rahul Sawhney

from syntheticmind.profilers.pytorch.profiler import PyTorchProfiler

__all__: list[str] = ["PyTorchProfiler"]
