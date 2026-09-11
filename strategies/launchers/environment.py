# This module:
# 1. Owns construction, rank specialization, capture, and restoration of the
#    environment variables used by local distributed launchers
#
# Design decisions:
# - ClusterEnvironment supplies semantic topology and rendezvous values;
#   DistributedEnvironment converts them into the standard variables consumed by
#   torch.distributed and child interpreters
# - OMP_NUM_THREADS defaults to available CPUs divided across local processes,
#   preventing every worker from independently oversubscribing the host
# - Rank-specific values derive global rank from node rank and local process
#   index, preserving multi-node coordinates even though local launch itself is
#   restricted to one node
# - Capture records absence as well as values and restore removes variables that
#   did not exist, so a completed or failed launch leaves the parent environment
#   exactly as it began
#
# Author: Rahul Sawhney

import os
from typing import ClassVar

from syntheticmind.plugins.environments import ClusterEnvironment
from syntheticmind.strategies.launchers.protocols import LaunchStrategy
from syntheticmind.primitives.exceptions import MisconfigurationError

__all__: list[str] = []


class _DistributedEnvironment:
    # Reversible environment owner bound to the narrow launch topology supplied
    # by ParallelStrategy.
    _environment_keys: ClassVar[tuple[str, ...]] = (
        "MASTER_ADDR",
        "MASTER_PORT",
        "NODE_RANK",
        "GROUP_RANK",
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "OMP_NUM_THREADS"
    )

    def __init__(self, strategy: LaunchStrategy) -> None:
        # Binds the structural topology without mutating the parent environment.
        self._strategy: LaunchStrategy = strategy

    def local_values(self) -> dict[str, str]:
        # Validates rendezvous and thread policy and constructs values shared by
        # every worker on this node before local-rank specialization.
        cluster_environment: ClusterEnvironment = self._strategy.cluster_environment
        master_address: str = cluster_environment.main_address
        master_port: str = str(cluster_environment.main_port)
        if not master_address.strip():
            raise MisconfigurationError("MASTER_ADDR must not be empty.")
        try:
            numeric_port: int = int(master_port)
        except ValueError as exception:
            raise MisconfigurationError(
                f"MASTER_PORT must be an integer, got {master_port!r}."
            ) from exception
        if numeric_port < 1 or numeric_port > 65535:
            raise MisconfigurationError(
                f"MASTER_PORT must be in the range [1, 65535], got {numeric_port}."
            )
        cpu_count: int = os.cpu_count() or 1
        default_thread_count: int = max(1, cpu_count // self._strategy.num_processes)
        thread_value: str = os.environ.get(
            "OMP_NUM_THREADS",
            str(default_thread_count)
        )
        try:
            thread_count: int = int(thread_value)
        except ValueError as exception:
            raise MisconfigurationError(
                f"OMP_NUM_THREADS must be an integer, got {thread_value!r}."
            ) from exception
        if thread_count < 1:
            raise MisconfigurationError(
                f"OMP_NUM_THREADS must be at least one, got {thread_count}."
            )
        return {
            "MASTER_ADDR": master_address,
            "MASTER_PORT": str(numeric_port),
            "NODE_RANK": str(cluster_environment.node_rank()),
            "GROUP_RANK": str(cluster_environment.node_rank()),
            "WORLD_SIZE": str(
                self._strategy.num_nodes * self._strategy.num_processes
            ),
            "OMP_NUM_THREADS": str(thread_count)
        }

    def rank_values(
        self,
        environment: dict[str, str],
        local_rank: int
    ) -> dict[str, str]:
        # Adds local and global rank to a copied base environment for one worker.
        rank_environment: dict[str, str] = dict(environment)
        rank_environment["LOCAL_RANK"]: str = str(local_rank)
        node_rank: int = int(environment["NODE_RANK"])
        rank_environment["RANK"]: str = str(
            node_rank * self._strategy.num_processes + local_rank
        )
        return rank_environment

    def capture(self) -> dict[str, str | None]:
        # Snapshots every variable this launcher may mutate, including absence.
        snapshot: dict[str, str | None] = {}
        key: str
        for key in self._environment_keys:
            snapshot[key]: str | None = os.environ.get(key)
        return snapshot

    def restore(self, snapshot: dict[str, str | None]) -> None:
        # Restores exact parent values and removes keys absent in the snapshot.
        key: str
        value: str | None
        for key, value in snapshot.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key]: str = value

    @property
    def creates_processes_externally(self) -> bool:
        # Forwards external process ownership from the selected cluster provider.
        return self._strategy.cluster_environment.creates_processes_externally
