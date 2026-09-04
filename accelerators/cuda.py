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

    @staticmethod
    @override
    def parse_devices(devices: DeviceInput) -> int | list[int]:
        parser: _AcceleratorDeviceParser = _AcceleratorDeviceParser(
            CUDAAccelerator.auto_device_count(),
            "CUDAAccelerator"
        )
        return parser.parse(devices)

    @staticmethod
    @override
    def get_parallel_devices(devices: int | list[int]) -> list[torch.device]:
        if not isinstance(devices, list):
            raise TypeError("CUDAAccelerator requires parsed device IDs.")
        parallel_devices: list[torch.device] = []
        device_id: int
        for device_id in devices:
            parallel_devices.append(torch.device("cuda", device_id))
        return parallel_devices

    @staticmethod
    @override
    def auto_device_count() -> int:
        return torch.cuda.device_count()

    @staticmethod
    @override
    def is_available() -> bool:
        return torch.cuda.is_available()

    @staticmethod
    @override
    def name() -> str:
        return "cuda"
