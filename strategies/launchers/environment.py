from __future__ import annotations

import os
from typing import TYPE_CHECKING, ClassVar

from syntheticmind.utilities.exceptions import MisconfigurationError

if TYPE_CHECKING:
    from syntheticmind.strategies.ddp import DDPStrategy

__all__: list[str] = []


class _DistributedEnvironment:
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
    def __init__(self, strategy: DDPStrategy) -> None:
        self._strategy: DDPStrategy = strategy

    def local_values(self) -> dict[str, str]:
        cluster_environment = self._strategy.cluster_environment
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
        rank_environment: dict[str, str] = dict(environment)
        rank_environment["LOCAL_RANK"] = str(local_rank)
        node_rank: int = int(environment["NODE_RANK"])
        rank_environment["RANK"] = str(
            node_rank * self._strategy.num_processes + local_rank
        )
        return rank_environment

    def capture(self) -> dict[str, str | None]:
        return {
            key: os.environ.get(key) for key in self._environment_keys
        }

    def restore(self, snapshot: dict[str, str | None]) -> None:
        for key, value in snapshot.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    @property
    def creates_processes_externally(self) -> bool:
        return self._strategy.cluster_environment.creates_processes_externally
