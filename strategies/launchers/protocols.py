# This module:
# 1. Defines the narrow Strategy capability required by process launchers:
#    cluster environment, local process count, node count, and root device
#
# Design decisions:
# - Launchers depend on this structural projection rather than concrete DDP or
#   model-parallel classes, keeping process creation independent from wrapping
# - The contract contains only topology needed to construct worker environments
#   and never exposes Trainer execution or collective operations
#
# Author: Rahul Sawhney

from typing import Protocol, runtime_checkable

import torch

from syntheticmind.plugins.environments import ClusterEnvironment

__all__: list[str] = ["LaunchStrategy"]


@runtime_checkable
class LaunchStrategy(Protocol):
    # Structural topology view implemented by ParallelStrategy.
    num_nodes: int

    # Topology and rendezvous provider consumed during environment construction.
    @property
    def cluster_environment(self) -> ClusterEnvironment: ...

    # Number of worker processes created on the current node.
    @property
    def num_processes(self) -> int: ...

    # Process-local device assigned after rank specialization.
    @property
    def root_device(self) -> torch.device: ...
