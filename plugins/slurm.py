import os
import subprocess
from typing import override

from syntheticmind.plugins.environments import ClusterEnvironment

__all__: list[str] = ["SLURMEnvironment"]


class SLURMEnvironment(ClusterEnvironment):
    def __init__(self) -> None:
        self._world_size: int = 1
        self._global_rank: int = 0
        self._main_address: str | None = None
        self._main_port: int | None = None

    @classmethod
    @override
    def detect(cls) -> bool:
        del cls
        return "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ

    @override
    def world_size(self) -> int:
        return int(os.environ.get("SLURM_NTASKS", str(self._world_size)))

    @override
    def set_world_size(self, size: int) -> None:
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(f"SLURM world size must be positive, got {size!r}.")
        self._world_size: int = size

    @override
    def global_rank(self) -> int:
        return int(os.environ.get("SLURM_PROCID", str(self._global_rank)))

    @override
    def set_global_rank(self, rank: int) -> None:
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError(f"SLURM global rank must be non-negative, got {rank!r}.")
        self._global_rank: int = rank

    @override
    def local_rank(self) -> int:
        return int(os.environ.get("SLURM_LOCALID", "0"))

    @override
    def node_rank(self) -> int:
        return int(os.environ.get("SLURM_NODEID", "0"))

    @override
    def validate_settings(self, num_devices: int, num_nodes: int) -> None:
        if isinstance(num_devices, bool) or not isinstance(num_devices, int):
            raise TypeError("num_devices must be an integer.")
        if isinstance(num_nodes, bool) or not isinstance(num_nodes, int):
            raise TypeError("num_nodes must be an integer.")
        if num_devices < 1 or num_nodes < 1:
            raise ValueError(
                f"SLURM topology requires positive devices and nodes, got "
                f"num_devices={num_devices}, num_nodes={num_nodes}."
            )
        launched_tasks: int = self.world_size()
        expected_tasks: int = num_devices * num_nodes
        if launched_tasks != expected_tasks:
            raise ValueError(
                f"SLURM launched {launched_tasks} tasks but SyntheticMind expects "
                f"{expected_tasks}. Set --ntasks-per-node={num_devices} and "
                f"--nodes={num_nodes}."
            )

    @override
    def teardown(self) -> None:
        self._main_address: str | None = None
        self._main_port: int | None = None

    @property
    @override
    def creates_processes_externally(self) -> bool:
        return self.detect()

    @property
    @override
    def main_address(self) -> str:
        if self._main_address is None:
            self._main_address: str | None = self._resolve_main_address()
        return self._main_address

    @property
    @override
    def main_port(self) -> int:
        if self._main_port is None:
            self._main_port: int | None = self._resolve_main_port()
        return self._main_port

    def _resolve_main_address(self) -> str:
        configured_address: str | None = os.environ.get("MASTER_ADDR")
        if configured_address:
            return configured_address
        node_list: str | None = os.environ.get("SLURM_NODELIST")
        if not node_list:
            raise RuntimeError(
                "SLURM rendezvous requires MASTER_ADDR or SLURM_NODELIST."
            )
        try:
            result: subprocess.CompletedProcess[str] = subprocess.run(
                ["scontrol", "show", "hostnames", node_list],
                check=True,
                capture_output=True,
                text=True
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exception:
            raise RuntimeError(
                "Could not resolve the first SLURM host. Set MASTER_ADDR explicitly."
            ) from exception
        hostnames: list[str] = [
            hostname.strip()
            for hostname in result.stdout.splitlines()
            if hostname.strip()
        ]
        if not hostnames:
            raise RuntimeError(
                f"SLURM_NODELIST={node_list!r} resolved to no hostnames."
            )
        return hostnames[0]

    def _resolve_main_port(self) -> int:
        configured_port: str | None = os.environ.get("MASTER_PORT")
        if configured_port is not None:
            port: int = int(configured_port)
            if not 1 <= port <= 65535:
                raise ValueError(f"MASTER_PORT must be in 1..65535, got {port}.")
            return port
        job_id_text: str | None = os.environ.get("SLURM_JOB_ID")
        if job_id_text is None:
            raise RuntimeError(
                "SLURM rendezvous requires MASTER_PORT or SLURM_JOB_ID."
            )
        job_id: int = int(job_id_text)
        return 10000 + job_id % 50000
