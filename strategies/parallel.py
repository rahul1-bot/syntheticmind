import atexit
import os
from datetime import timedelta
from typing import cast, override

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as distributed_all_gather

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.plugins.environments import ClusterEnvironment, LocalEnvironment
from syntheticmind.plugins.checkpoint_io import CheckpointIO
from syntheticmind.plugins.precision.precision import Precision
from syntheticmind.strategies.launchers.launcher import ProcessStartMethod, _ProcessLauncher
from syntheticmind.strategies.launchers.protocols import LaunchStrategy
from syntheticmind.strategies.strategy import Strategy
from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.types import OptimizerStateValue

__all__: list[str] = ["ParallelStrategy"]


class ParallelStrategy(Strategy, LaunchStrategy):
    @override
    def __init__(
        self,
        accelerator: Accelerator | None = None,
        parallel_devices: list[torch.device] | None = None,
        cluster_environment: ClusterEnvironment | None = None,
        checkpoint_io: CheckpointIO | None = None,
        precision_plugin: Precision | None = None,
        process_group_backend: str | None = None,
        timeout: timedelta | None = None,
        start_method: ProcessStartMethod = "popen"
    ) -> None:
        super().__init__(
            accelerator,
            parallel_devices,
            checkpoint_io,
            precision_plugin
        )
        if process_group_backend is not None and not isinstance(
            process_group_backend,
            str
        ):
            raise TypeError("`process_group_backend` must be a string or None.")
        if timeout is not None and not isinstance(timeout, timedelta):
            raise TypeError("`timeout` must be a datetime.timedelta or None.")
        if timeout is not None and timeout.total_seconds() < 0:
            raise ValueError("`timeout` must be non-negative.")
        self.is_distributed: bool = True
        self._cluster_environment: ClusterEnvironment = (
            cluster_environment or LocalEnvironment()
        )
        self._configured_cluster_environment: ClusterEnvironment | None = (
            cluster_environment
        )
        self._launcher: _ProcessLauncher = _ProcessLauncher(self, start_method)
        self._local_rank: int = 0
        self._owns_process_group: bool = False
        self._process_group_backend: str | None = (
            process_group_backend.lower()
            if process_group_backend is not None
            else None
        )
        self._process_group_cleanup_registered: bool = False
        self._timeout: timedelta | None = timeout

    @override
    def configure_topology(
        self,
        accelerator: Accelerator,
        parallel_devices: list[torch.device],
        num_nodes: int
    ) -> None:
        if not parallel_devices or num_nodes < 1:
            raise ValueError("Parallel execution requires devices and positive nodes.")
        self.accelerator: Accelerator | None = accelerator
        self.parallel_devices: list[torch.device] = list(parallel_devices)
        self.num_nodes: int = num_nodes
        self._configure_environment()

    @override
    def setup_environment(self) -> None:
        self._cluster_environment.validate_settings(
            self.num_processes,
            self.num_nodes
        )
        self._local_rank: int = self._cluster_environment.local_rank()
        if not 0 <= self._local_rank < self.num_processes:
            raise MisconfigurationError(
                f"local_rank={self._local_rank} is outside the configured device range."
            )
        backend: str = self._process_group_backend or (
            "nccl" if self.root_device.type == "cuda" else "gloo"
        )
        configured_world_size: int = self.num_nodes * self.num_processes
        launched_world_size: int = self._cluster_environment.world_size()
        global_rank: int = self._cluster_environment.global_rank()
        node_rank: int = self._cluster_environment.node_rank()
        self._validate_topology(
            configured_world_size,
            launched_world_size,
            global_rank,
            node_rank
        )
        if dist.is_initialized():
            self._validate_process_group(configured_world_size, backend)
            return
        if backend == "nccl" and not torch.cuda.is_available():
            raise MisconfigurationError("NCCL distributed execution requires CUDA.")
        os.environ["MASTER_ADDR"] = self._cluster_environment.main_address
        os.environ["MASTER_PORT"] = str(self._cluster_environment.main_port)
        os.environ["WORLD_SIZE"] = str(launched_world_size)
        os.environ["RANK"] = str(global_rank)
        os.environ["LOCAL_RANK"] = str(self._local_rank)
        dist.init_process_group(
            backend=backend,
            rank=global_rank,
            world_size=launched_world_size,
            timeout=self._timeout
        )
        self._owns_process_group: bool = True
        self._register_process_group_cleanup()

    @override
    def on_exception(self, exception: BaseException) -> None:
        del exception
        self._destroy_process_group()

    @override
    def barrier(self) -> None:
        if dist.is_initialized():
            dist.barrier()

    @override
    def broadcast[T](self, value: T, source_rank: int = 0) -> T:
        if not dist.is_initialized():
            return value
        values: list[T] = [value]
        dist.broadcast_object_list(
            values,
            src=source_rank,
            device=self.root_device if self.root_device.type == "cuda" else None
        )
        return cast(T, values[0])

    @override
    def reduce_boolean_decision(
        self,
        decision: bool,
        all: bool = True
    ) -> bool:
        if type(all) is not bool:
            raise TypeError("`all` must be a bool.")
        if not dist.is_initialized():
            return decision
        decision_tensor: torch.Tensor = torch.tensor(
            int(decision),
            device=self.root_device
        )
        dist.all_reduce(decision_tensor, op=dist.ReduceOp.SUM)
        decision_count: int = int(decision_tensor.item())
        return decision_count == self.world_size if all else decision_count > 0

    @override
    def all_gather(
        self,
        tensor: torch.Tensor,
        group: dist.ProcessGroup | None = None,
        sync_grads: bool = False
    ) -> torch.Tensor:
        if not dist.is_initialized():
            return tensor
        contiguous_tensor: torch.Tensor = tensor.contiguous()
        if sync_grads:
            gathered_tensors: tuple[torch.Tensor, ...] = cast(
                tuple[torch.Tensor, ...],
                distributed_all_gather(contiguous_tensor, group=group)
            )
            return torch.stack(gathered_tensors)
        world_size: int = dist.get_world_size(group)
        gathered: list[torch.Tensor] = []
        rank_index: int
        for rank_index in range(world_size):
            del rank_index
            gathered.append(torch.zeros_like(contiguous_tensor))
        dist.all_gather(gathered, contiguous_tensor, group=group)
        return torch.stack(gathered)

    @override
    def sync_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        if not dist.is_initialized():
            return
        synchronized_parameters: set[int] = set()
        parameter_groups: list[dict[str, OptimizerStateValue]] = cast(
            list[dict[str, OptimizerStateValue]],
            optimizer.param_groups
        )
        parameter_group: dict[str, OptimizerStateValue]
        for parameter_group in parameter_groups:
            parameters: OptimizerStateValue = parameter_group["params"]
            if not isinstance(parameters, list):
                raise TypeError("Optimizer parameter groups must contain a parameter list.")
            parameter: OptimizerStateValue
            for parameter in parameters:
                if not isinstance(parameter, torch.nn.Parameter):
                    raise TypeError("Optimizer parameter groups must contain Parameters.")
                parameter_identity: int = id(parameter)
                if parameter_identity in synchronized_parameters:
                    continue
                gradient_flags: torch.Tensor = torch.tensor(
                    [
                        int(parameter.grad is not None),
                        int(parameter.grad is not None and parameter.grad.is_sparse)
                    ],
                    device=parameter.device,
                    dtype=torch.int64
                )
                dist.all_reduce(gradient_flags, op=dist.ReduceOp.SUM)
                if int(gradient_flags[1].item()) > 0:
                    raise RuntimeError("Distributed tails do not support sparse gradients.")
                if int(gradient_flags[0].item()) > 0:
                    gradient: torch.Tensor = (
                        parameter.grad
                        if parameter.grad is not None
                        else torch.zeros_like(parameter)
                    )
                    dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
                    gradient.div_(self.world_size)
                    if parameter.grad is None:
                        parameter.grad: torch.Tensor | None = gradient
                synchronized_parameters.add(parameter_identity)

    def attach_cluster_environment(
        self,
        cluster_environment: ClusterEnvironment
    ) -> None:
        if not isinstance(cluster_environment, ClusterEnvironment):
            raise TypeError("cluster_environment must be a ClusterEnvironment.")
        self._cluster_environment: ClusterEnvironment = cluster_environment
        self._configure_environment()

    @property
    @override
    def distributed_sampler_kwargs(self) -> dict[str, int] | None:
        return {"num_replicas": self.world_size, "rank": self.global_rank}

    @property
    @override
    def launcher(self) -> _ProcessLauncher:
        return self._launcher

    @property
    @override
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices else 1

    @property
    @override
    def global_rank(self) -> int:
        if dist.is_initialized():
            return dist.get_rank()
        return self._cluster_environment.global_rank()

    @property
    @override
    def local_rank(self) -> int:
        return self._local_rank

    @property
    @override
    def node_rank(self) -> int:
        return self._cluster_environment.node_rank()

    @property
    @override
    def world_size(self) -> int:
        if dist.is_initialized():
            return dist.get_world_size()
        return self._cluster_environment.world_size()

    @property
    @override
    def root_device(self) -> torch.device:
        if not self.parallel_devices:
            return torch.device("cpu")
        return self.parallel_devices[self.local_rank]

    @property
    @override
    def cluster_environment(self) -> ClusterEnvironment:
        return self._cluster_environment

    @property
    def configured_cluster_environment(self) -> ClusterEnvironment | None:
        return self._configured_cluster_environment

    def _configure_environment(self) -> None:
        self._cluster_environment.validate_settings(
            self.num_processes,
            self.num_nodes
        )
        if self._cluster_environment.creates_processes_externally:
            return
        self._cluster_environment.set_world_size(
            self.num_nodes * self.num_processes
        )
        self._cluster_environment.set_global_rank(
            self._cluster_environment.node_rank() * self.num_processes
            + self._cluster_environment.local_rank()
        )

    def _validate_topology(
        self,
        configured_world_size: int,
        launched_world_size: int,
        global_rank: int,
        node_rank: int
    ) -> None:
        if launched_world_size != configured_world_size:
            raise MisconfigurationError(
                f"world_size={launched_world_size} does not match "
                f"configured_world_size={configured_world_size}."
            )
        if not 0 <= global_rank < launched_world_size:
            raise MisconfigurationError("The global rank is outside the launched world.")
        if not 0 <= node_rank < self.num_nodes:
            raise MisconfigurationError("The node rank is outside the configured nodes.")
        expected_global_rank: int = node_rank * self.num_processes + self._local_rank
        if expected_global_rank != global_rank:
            raise MisconfigurationError(
                f"global_rank={global_rank} does not match expected_rank={expected_global_rank}."
            )

    def _validate_process_group(self, world_size: int, backend: str) -> None:
        if dist.get_world_size() != world_size:
            raise MisconfigurationError("The active process group has the wrong world size.")
        active_backend: str = str(dist.get_backend()).lower()
        if active_backend != backend:
            raise MisconfigurationError(
                f"The active process group uses {active_backend!r}, expected {backend!r}."
            )

    def _register_process_group_cleanup(self) -> None:
        if self._process_group_cleanup_registered:
            return
        atexit.register(self._destroy_process_group)
        self._process_group_cleanup_registered: bool = True

    def _destroy_process_group(self) -> None:
        if self._owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
        self._owns_process_group: bool = False
