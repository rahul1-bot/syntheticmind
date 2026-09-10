# This module:
# 1. Reads and validates the topology and restart identity supplied by torchrun
#
# Design decisions:
# - TorchElastic owns worker restart; the checkpoint connector owns resume paths
# - Incomplete launcher state fails before a second worker group can be launched
# - Research configuration is not sourced from environment variables
#
# Author: Rahul Sawhney

import os
from typing import override

from syntheticmind.plugins.environments import LocalEnvironment

__all__: list[str] = ["TorchElasticEnvironment"]


class TorchElasticEnvironment(LocalEnvironment):
    # ParallelStrategy auto-selects this provider for an elastic torchrun job.
    # CheckpointConnector consumes restart_count to require durable continuation.
    @classmethod
    @override
    def detect(cls) -> bool:
        del cls
        name: str
        return all(
            name in os.environ
            for name in ("TORCHELASTIC_RUN_ID", "LOCAL_RANK", "RANK", "WORLD_SIZE")
        )

    @override
    def validate_settings(self, num_devices: int, num_nodes: int) -> None:
        super().validate_settings(num_devices, num_nodes)
        if not self.detect():
            raise RuntimeError("Incomplete TorchElastic topology; launch with torchrun.")
        if not os.environ.get("MASTER_ADDR") or "MASTER_PORT" not in os.environ:
            raise RuntimeError("TorchElastic requires MASTER_ADDR and MASTER_PORT.")
        if self.world_size() != num_devices * num_nodes:
            raise ValueError("TorchElastic world size does not match configured devices and nodes.")
        if not 0 <= self.local_rank() < num_devices:
            raise ValueError("TorchElastic LOCAL_RANK is outside the configured device range.")
        if not 0 <= self.global_rank() < self.world_size():
            raise ValueError("TorchElastic RANK is outside WORLD_SIZE.")
        if not 0 <= self.node_rank() < num_nodes:
            raise ValueError("TorchElastic node rank is outside configured num_nodes.")
        if self.restart_count > self.maximum_restarts:
            raise ValueError("TorchElastic restart count exceeds its restart budget.")
        if not self.run_id.strip():
            raise ValueError("TorchElastic run identity cannot be blank.")
        if not 1 <= self.main_port <= 65535:
            raise ValueError("TorchElastic MASTER_PORT is outside 1..65535.")

    @property
    @override
    def creates_processes_externally(self) -> bool:
        return True

    @property
    def restart_count(self) -> int:
        return self._restart_integer("TORCHELASTIC_RESTART_COUNT")

    @property
    def maximum_restarts(self) -> int:
        return self._restart_integer("TORCHELASTIC_MAX_RESTARTS")

    @property
    def run_id(self) -> str:
        run_id: str | None = os.environ.get("TORCHELASTIC_RUN_ID")
        if not run_id:
            raise RuntimeError("TorchElasticEnvironment requires TORCHELASTIC_RUN_ID.")
        return run_id

    def _restart_integer(self, name: str) -> int:
        value: int = int(os.environ.get(name, "0"))
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
        return value
