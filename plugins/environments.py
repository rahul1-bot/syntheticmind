import os
import socket
from typing import Protocol, override, runtime_checkable

__all__: list[str] = ["ClusterEnvironment", "LocalEnvironment"]


@runtime_checkable
class ClusterEnvironment(Protocol):
    @staticmethod
    def detect() -> bool: ...

    def world_size(self) -> int: ...

    def set_world_size(self, size: int) -> None: ...

    def global_rank(self) -> int: ...

    def set_global_rank(self, rank: int) -> None: ...

    def local_rank(self) -> int: ...

    def node_rank(self) -> int: ...

    def validate_settings(self, num_devices: int, num_nodes: int) -> None: ...

    def teardown(self) -> None: ...

    @property
    def creates_processes_externally(self) -> bool: ...

    @property
    def main_address(self) -> str: ...

    @property
    def main_port(self) -> int: ...


class LocalEnvironment(ClusterEnvironment):
    def __init__(self) -> None:
        self._main_port: int | None = None
        self._global_rank: int = 0
        self._world_size: int = 1

    @override
    def world_size(self) -> int:
        return int(os.environ.get("WORLD_SIZE", str(self._world_size)))

    @override
    def set_world_size(self, size: int) -> None:
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("Cluster world size must be a positive integer.")
        self._world_size: int = size

    @override
    def global_rank(self) -> int:
        return int(os.environ.get("RANK", str(self._global_rank)))

    @override
    def set_global_rank(self, rank: int) -> None:
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
        if isinstance(num_devices, bool) or not isinstance(num_devices, int):
            raise TypeError("num_devices must be an integer.")
        if isinstance(num_nodes, bool) or not isinstance(num_nodes, int):
            raise TypeError("num_nodes must be an integer.")
        if num_devices < 1 or num_nodes < 1:
            raise ValueError("Distributed topology requires positive devices and nodes.")

    @override
    def teardown(self) -> None:
        pass

    @staticmethod
    @override
    def detect() -> bool:
        return True

    @property
    @override
    def creates_processes_externally(self) -> bool:
        return "LOCAL_RANK" in os.environ

    @property
    @override
    def main_address(self) -> str:
        return os.environ.get("MASTER_ADDR", "127.0.0.1")

    @property
    @override
    def main_port(self) -> int:
        if self._main_port is None:
            self._main_port: int | None = self._resolve_main_port()
        return self._main_port

    def _resolve_main_port(self) -> int:
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
        listener: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with listener:
            listener.bind(("", 0))
            return int(listener.getsockname()[1])
