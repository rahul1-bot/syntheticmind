from typing import override

import torch

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.accelerators.device_parser import DeviceInput, _AcceleratorDeviceParser
from syntheticmind.primitives.exceptions import MisconfigurationError

__all__: list[str] = ["MPSAccelerator"]


class MPSAccelerator(Accelerator):
    def __init__(self) -> None:
        self._device: torch.device = torch.device("mps")

    @override
    def setup_device(self, device: torch.device) -> None:
        if device.type != "mps":
            raise MisconfigurationError(f"MPSAccelerator cannot bind {device}.")
        self._device: torch.device = device

    @override
    def get_device(self) -> torch.device:
        return self._device

    @override
    def get_device_stats(self, device: torch.device) -> dict[str, float]:
        if device.type != "mps":
            raise MisconfigurationError(f"MPSAccelerator cannot inspect {device}.")
        return {
            "mps/current_allocated_bytes": float(torch.mps.current_allocated_memory()),
            "mps/driver_allocated_bytes": float(torch.mps.driver_allocated_memory()),
            "mps/recommended_max_bytes": float(torch.mps.recommended_max_memory())
        }

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
            raise TypeError("MPSAccelerator requires parsed device IDs.")
        parallel_devices: list[torch.device] = []
        device_id: int
        for device_id in devices:
            parallel_devices.append(torch.device("mps", device_id))
        return parallel_devices

    @classmethod
    @override
    def auto_device_count(cls) -> int:
        del cls
        return 1

    @classmethod
    @override
    def is_available(cls) -> bool:
        del cls
        return torch.backends.mps.is_available()

    @classmethod
    @override
    def name(cls) -> str:
        del cls
        return "mps"
