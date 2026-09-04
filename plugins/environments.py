# This module:
# 1. Defines ClusterEnvironment, the structural contract mapping scheduler or
#    launcher state into distributed ranks, world size, rendezvous, and process
#    ownership
# 2. Implements LocalEnvironment for ordinary scripts, local multiprocessing,
#    torchrun, and scheduler environments that expose standard variables
#
# Design decisions:
# - ClusterEnvironment owns topology facts supplied by the execution environment;
#   Strategy owns process groups and collective behavior
# - LocalEnvironment prefers standard environment variables over mutable local
#   defaults, allowing externally launched workers to reconstruct rank without a
#   separate scheduler-specific adapter
# - Node rank falls back from NODE_RANK to torchrun's GROUP_RANK convention
# - The rendezvous port honors MASTER_PORT, then STANDALONE_PORT, and otherwise
#   reserves a free local port for the upcoming launcher
# - creates_processes_externally is derived from LOCAL_RANK so ProcessLauncher
#   re-enters the stage directly instead of recursively launching more workers
#
# Author: Rahul Sawhney

import os
import socket
from typing import Protocol, override, runtime_checkable

__all__: list[str] = ["ClusterEnvironment", "LocalEnvironment"]


@runtime_checkable
class ClusterEnvironment(Protocol):
    # Interchangeable topology and rendezvous provider. Custom cluster clients
    # implement this Protocol and pass the instance through Trainer plugins.
    # Whether this environment implementation applies on the current process.
    @classmethod
    def detect(cls) -> bool: ...

    # Total launched process count visible to Strategy.
    def world_size(self) -> int: ...

    # Parent-side world-size configuration before worker export.
    def set_world_size(self, size: int) -> None: ...

    # Process-global rank across every node.
    def global_rank(self) -> int: ...

    # Parent-side global-rank configuration before worker export.
    def set_global_rank(self, rank: int) -> None: ...

    # Rank among processes launched on the current node.
    def local_rank(self) -> int: ...

    # Rank of the current node within the distributed job.
    def node_rank(self) -> int: ...

    # Validates requested local devices and node count against environment policy.
    def validate_settings(self, num_devices: int, num_nodes: int) -> None: ...

    # Releases environment-owned resources without destroying Strategy collectives.
    def teardown(self) -> None: ...

    # Whether torchrun or a scheduler already created this process.
    @property
    def creates_processes_externally(self) -> bool: ...

    # Rendezvous host visible to every participating process.
    @property
    def main_address(self) -> str: ...

    # Rendezvous TCP port visible to every participating process.
    @property
    def main_port(self) -> int: ...


class LocalEnvironment(ClusterEnvironment):
    # Standard environment-variable implementation used for local and torchrun
    # execution. Mutable defaults support parent-side topology before worker
    # environment variables exist.
    def __init__(self) -> None:
        self._main_port: int | None = None
        self._global_rank: int = 0
        self._world_size: int = 1

    @override
    def world_size(self) -> int:
        return int(os.environ.get("WORLD_SIZE", str(self._world_size)))

    @override
    def set_world_size(self, size: int) -> None:
        # Stores parent-side world size after strict positive validation.
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("Cluster world size must be a positive integer.")
        self._world_size: int = size

    @override
    def global_rank(self) -> int:
        return int(os.environ.get("RANK", str(self._global_rank)))

    @override
    def set_global_rank(self, rank: int) -> None:
        # Stores parent-side global rank before worker environment export.
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("Cluster global rank must be a non-negative integer.")
        self._global_rank: int = rank

    @override
    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", "0"))

    @override
    def node_rank(self) -> int:
        group_rank: str = os.environ.get("GROUP_RANK", "0")
        return int(os.environ.get("NODE_RANK", group_rank))

    @override
    def validate_settings(self, num_devices: int, num_nodes: int) -> None:
        # Rejects non-integral and empty process topology before rendezvous.
        if isinstance(num_devices, bool) or not isinstance(num_devices, int):
            raise TypeError("num_devices must be an integer.")
        if isinstance(num_nodes, bool) or not isinstance(num_nodes, int):
            raise TypeError("num_nodes must be an integer.")
        if num_devices < 1 or num_nodes < 1:
            raise ValueError("Distributed topology requires positive devices and nodes.")

    @override
    def teardown(self) -> None:
        pass

    @classmethod
    @override
    def detect(cls) -> bool:
        del cls
        return True

    @property
    @override
    def creates_processes_externally(self) -> bool:
        # LOCAL_RANK identifies a process already created by torchrun or a
        # scheduler and prevents nested local launch.
        return "LOCAL_RANK" in os.environ

    @property
    @override
    def main_address(self) -> str:
        return os.environ.get("MASTER_ADDR", "127.0.0.1")

    @property
    @override
    def main_port(self) -> int:
        # Lazily resolves one stable rendezvous port per environment instance.
        if self._main_port is None:
            self._main_port: int | None = self._resolve_main_port()
        return self._main_port

    def _resolve_main_port(self) -> int:
        # Applies standard and standalone environment precedence and validates
        # the complete TCP port range.
        configured_port: str | None = os.environ.get("MASTER_PORT")
        if configured_port is None:
            configured_port: str | None = os.environ.get("STANDALONE_PORT")
        if configured_port is None:
            return self._find_free_network_port()
        try:
            port: int = int(configured_port)
        except ValueError as exception:
            raise ValueError("Cluster main port must be an integer.") from exception
        if not 1 <= port <= 65535:
            raise ValueError("Cluster main port must be between 1 and 65535.")
        return port

    def _find_free_network_port(self) -> int:
        # Asks the operating system for an ephemeral local port and releases the
        # probe socket immediately for the subsequent process-group rendezvous.
        listener: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with listener:
            listener.bind(("", 0))
            return int(listener.getsockname()[1])
