from typing import Protocol, runtime_checkable

from syntheticmind.plugins.environments import ClusterEnvironment
from syntheticmind.strategies.launchers.protocols import LaunchStrategy
from syntheticmind.strategies.strategy import Strategy

__all__: list[str] = ["ParallelStrategy"]


@runtime_checkable
class ParallelStrategy(Strategy, LaunchStrategy, Protocol):
    is_distributed: bool

    def attach_cluster_environment(
        self,
        cluster_environment: ClusterEnvironment
    ) -> None: ...

    @property
    def cluster_environment(self) -> ClusterEnvironment: ...

    @property
    def configured_cluster_environment(self) -> ClusterEnvironment | None: ...
