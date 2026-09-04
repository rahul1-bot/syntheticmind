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
