import os
import socket
from typing import override

__all__: list[str] = ["ClusterEnvironment", "LocalEnvironment"]


class ClusterEnvironment:
    @property
    def creates_processes_externally(self) -> bool:
        raise NotImplementedError

    @property
    def main_address(self) -> str:
        raise NotImplementedError

    @property
    def main_port(self) -> int:
        raise NotImplementedError

    @staticmethod
    def detect() -> bool:
        raise NotImplementedError

    def world_size(self) -> int:
        raise NotImplementedError

    def set_world_size(self, size: int) -> None:
        raise NotImplementedError

    def global_rank(self) -> int:
        raise NotImplementedError

    def set_global_rank(self, rank: int) -> None:
        raise NotImplementedError

    def local_rank(self) -> int:
        raise NotImplementedError

    def node_rank(self) -> int:
        raise NotImplementedError

    def validate_settings(self, num_devices: int, num_nodes: int) -> None:
        del num_devices, num_nodes

    def teardown(self) -> None:
        pass


class LocalEnvironment(ClusterEnvironment):
    def __init__(self) -> None:
        self._main_port: int | None = None
        self._global_rank: int = 0
        self._world_size: int = 1

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
            self._main_port = (
                int(os.environ["MASTER_PORT"])
                if "MASTER_PORT" in os.environ
                else _find_free_network_port()
            )
        return self._main_port

    @staticmethod
    @override
    def detect() -> bool:
        return True

    @override
    def world_size(self) -> int:
        return int(os.environ.get("WORLD_SIZE", str(self._world_size)))

    @override
    def set_world_size(self, size: int) -> None:
        self._world_size = size

    @override
    def global_rank(self) -> int:
        return int(os.environ.get("RANK", str(self._global_rank)))

    @override
    def set_global_rank(self, rank: int) -> None:
        self._global_rank = rank

    @override
    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", "0"))

    @override
    def node_rank(self) -> int:
        group_rank: str = os.environ.get("GROUP_RANK", "0")
        return int(os.environ.get("NODE_RANK", group_rank))


def _find_free_network_port() -> int:
    if "STANDALONE_PORT" in os.environ:
        return int(os.environ["STANDALONE_PORT"])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("", 0))
        return int(listener.getsockname()[1])
