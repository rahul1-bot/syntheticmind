from typing import override

import torch

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.accelerators.device_parser import DeviceInput, _AcceleratorDeviceParser
from syntheticmind.exceptions import MisconfigurationError

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

    @staticmethod
    @override
    def parse_devices(devices: DeviceInput) -> int | list[int]:
        parser: _AcceleratorDeviceParser = _AcceleratorDeviceParser(
            MPSAccelerator.auto_device_count(),
            "MPSAccelerator"
        )
        return parser.parse(devices)

    @staticmethod
    @override
    def get_parallel_devices(devices: int | list[int]) -> list[torch.device]:
        if not isinstance(devices, list):
            raise TypeError("MPSAccelerator requires parsed device IDs.")
        parallel_devices: list[torch.device] = []
        device_id: int
        for device_id in devices:
            parallel_devices.append(torch.device("mps", device_id))
        return parallel_devices

    @staticmethod
    @override
    def auto_device_count() -> int:
        return 1

    @staticmethod
    @override
    def is_available() -> bool:
        return torch.backends.mps.is_available()

    @staticmethod
    @override
    def name() -> str:
        return "mps"
