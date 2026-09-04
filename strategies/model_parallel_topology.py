from typing import Literal

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from syntheticmind.core.module import Module

__all__: list[str] = []


class _ModelParallelTopology:
    def __init__(
        self,
        data_parallel_size: Literal["auto"] | int,
        tensor_parallel_size: Literal["auto"] | int
    ) -> None:
        self._validate_size("data_parallel_size", data_parallel_size)
        self._validate_size("tensor_parallel_size", tensor_parallel_size)
        self._data_parallel_size: Literal["auto"] | int = data_parallel_size
        self._tensor_parallel_size: Literal["auto"] | int = tensor_parallel_size
        self._device_mesh: DeviceMesh | None = None
        self._module: Module | None = None

    def connect(self, module: Module) -> None:
        self._module: Module | None = module
        if self._device_mesh is not None:
            module._device_mesh: DeviceMesh | None = self._device_mesh

    def setup(
        self,
        world_size: int,
        num_nodes: int,
        num_processes: int,
        device_type: str
    ) -> None:
        data_parallel_size: int = (
            num_nodes
            if self._data_parallel_size == "auto"
            else self._data_parallel_size
        )
        tensor_parallel_size: int = (
            num_processes
            if self._tensor_parallel_size == "auto"
            else self._tensor_parallel_size
        )
        if data_parallel_size * tensor_parallel_size != world_size:
            raise RuntimeError(
                f"data_parallel_size={data_parallel_size} and "
                f"tensor_parallel_size={tensor_parallel_size} must multiply to "
                f"world_size={world_size}."
            )
        self._device_mesh: DeviceMesh | None = init_device_mesh(
            device_type,
            (data_parallel_size, tensor_parallel_size),
            mesh_dim_names=("data_parallel", "tensor_parallel")
        )
        self._require_module()._device_mesh: DeviceMesh | None = self._device_mesh

    def teardown(self) -> None:
        if self._module is not None:
            self._module._device_mesh: DeviceMesh | None = None
        self._device_mesh: DeviceMesh | None = None

    @property
    def device_mesh(self) -> DeviceMesh:
        if self._device_mesh is None:
            raise RuntimeError("The device mesh is unavailable before process setup.")
        return self._device_mesh

    @property
    def distributed_sampler_kwargs(self) -> dict[str, int]:
        data_parallel_mesh: DeviceMesh = self.device_mesh["data_parallel"]
        return {
            "num_replicas": data_parallel_mesh.size(),
            "rank": data_parallel_mesh.get_local_rank()
        }

    def _require_module(self) -> Module:
        if self._module is None:
            raise RuntimeError("Model-parallel topology requires a connected Module.")
        return self._module

    def _validate_size(
        self,
        name: str,
        value: Literal["auto"] | int
    ) -> None:
        if value == "auto":
            return
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"`{name}` must be 'auto' or a positive integer.")
        if value < 1:
            raise ValueError(f"`{name}` must be greater than zero.")
