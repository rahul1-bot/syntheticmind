from typing import Protocol, runtime_checkable

import torch

from syntheticmind.plugins.environments import ClusterEnvironment

__all__: list[str] = ["LaunchStrategy"]


@runtime_checkable
class LaunchStrategy(Protocol):
    num_nodes: int

    @property
    def cluster_environment(self) -> ClusterEnvironment: ...

    @property
    def num_processes(self) -> int: ...

    @property
    def root_device(self) -> torch.device: ...
