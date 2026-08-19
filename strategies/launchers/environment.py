from __future__ import annotations

import os
import socket
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
    _external_rank_keys: ClassVar[tuple[str, ...]] = (
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE"
    )
    _required_external_keys: ClassVar[tuple[str, ...]] = (
        "MASTER_ADDR",
        "MASTER_PORT",
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE"
    )

    def __init__(self, strategy: DDPStrategy) -> None:
        self._strategy: DDPStrategy = strategy

    @classmethod
    def _find_free_port(cls) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])

    def local_values(self) -> dict[str, str]:
        master_address: str = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port: str = os.environ.get("MASTER_PORT", str(self._find_free_port()))
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
            "NODE_RANK": "0",
            "GROUP_RANK": "0",
            "WORLD_SIZE": str(self._strategy.num_processes),
            "OMP_NUM_THREADS": str(thread_count)
        }

    def rank_values(
        self,
        environment: dict[str, str],
        local_rank: int
    ) -> dict[str, str]:
        rank_environment: dict[str, str] = dict(environment)
        rank_environment["LOCAL_RANK"] = str(local_rank)
        rank_environment["RANK"] = str(local_rank)
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
        present_rank_keys: list[str] = [
            key for key in self._external_rank_keys if key in os.environ
        ]
        if not present_rank_keys:
            return False
        missing_keys: list[str] = [
            key for key in self._required_external_keys if key not in os.environ
        ]
        if missing_keys:
            raise MisconfigurationError(
                f"The distributed environment is incomplete. Present rank variables: "
                f"{present_rank_keys}; missing required variables: {missing_keys}."
            )
        return True
