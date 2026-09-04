from typing import override

import torch

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.accelerators.device_parser import DeviceInput, _AcceleratorDeviceParser
from syntheticmind.exceptions import MisconfigurationError

__all__: list[str] = ["CUDAAccelerator"]


class CUDAAccelerator(Accelerator):
    def __init__(self) -> None:
        self._device: torch.device | None = None

    @override
    def setup_device(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise MisconfigurationError(f"CUDAAccelerator cannot bind {device}.")
        self._device: torch.device | None = device
        torch.cuda.set_device(device)

    @override
    def get_device(self) -> torch.device:
        if self._device is None:
            return torch.device("cuda", 0)
        return self._device

    @override
    def get_device_stats(self, device: torch.device) -> dict[str, float]:
        if device.type != "cuda":
            raise MisconfigurationError(f"CUDAAccelerator cannot inspect {device}.")
        device_index: int = (
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        )
        memory_statistics: dict[str, int] = torch.cuda.memory_stats(device_index)
        selected_keys: tuple[str, ...] = (
            "allocated_bytes.all.current",
            "allocated_bytes.all.peak",
            "reserved_bytes.all.current",
            "reserved_bytes.all.peak",
            "active_bytes.all.current",
            "active_bytes.all.peak",
            "num_alloc_retries",
            "num_ooms"
        )
        statistics: dict[str, float] = {
            f"cuda/{key}": float(memory_statistics.get(key, 0))
            for key in selected_keys
        }
        free_bytes: int
        total_bytes: int
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        statistics["cuda/free_bytes"] = float(free_bytes)
        statistics["cuda/total_bytes"] = float(total_bytes)
        return statistics

    @classmethod
    @override
    def parse_devices(cls, devices: DeviceInput) -> int | list[int]:
        parser: _AcceleratorDeviceParser = _AcceleratorDeviceParser(
            cls.auto_device_count(),
            cls.__name__
        )
        return parser.parse(devices)

    @classmethod
    @override
    def get_parallel_devices(
        cls,
        devices: int | list[int]
    ) -> list[torch.device]:
        del cls
        if not isinstance(devices, list):
            raise TypeError("CUDAAccelerator requires parsed device IDs.")
        parallel_devices: list[torch.device] = []
        device_id: int
        for device_id in devices:
            parallel_devices.append(torch.device("cuda", device_id))
        return parallel_devices

    @classmethod
    @override
    def auto_device_count(cls) -> int:
        del cls
        return torch.cuda.device_count()

    @classmethod
    @override
    def is_available(cls) -> bool:
        del cls
        return torch.cuda.is_available()

    @classmethod
    @override
    def name(cls) -> str:
        del cls
        return "cuda"
