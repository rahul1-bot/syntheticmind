from typing import override

import torch

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.accelerators.device_parser import DeviceInput
from syntheticmind.utilities.exceptions import MisconfigurationError

__all__: list[str] = ["CPUAccelerator"]


class CPUAccelerator(Accelerator):
    def __init__(self) -> None:
        self._device: torch.device = torch.device("cpu")

    @override
    def setup_device(self, device: torch.device) -> None:
        if device.type != "cpu":
            raise MisconfigurationError(f"CPUAccelerator cannot bind {device}.")
        self._device: torch.device = device

    @override
    def get_device(self) -> torch.device:
        return self._device

    @staticmethod
    @override
    def parse_devices(devices: DeviceInput) -> int | list[int]:
        normalized_devices: int | str | list[int] = devices
        if isinstance(normalized_devices, str) and normalized_devices.strip().isdigit():
            normalized_devices: int | str | list[int] = int(normalized_devices)
        if isinstance(normalized_devices, bool) or not isinstance(normalized_devices, int):
            raise TypeError("CPUAccelerator requires a positive integer device count.")
        if normalized_devices <= 0:
            raise TypeError("CPUAccelerator requires a positive integer device count.")
        return normalized_devices

    @staticmethod
    @override
    def get_parallel_devices(devices: int | list[int]) -> list[torch.device]:
        if not isinstance(devices, int):
            raise TypeError("CPUAccelerator requires an integer device count.")
        return [torch.device("cpu")] * devices

    @staticmethod
    @override
    def auto_device_count() -> int:
        return 1

    @staticmethod
    @override
    def is_available() -> bool:
        return True

    @staticmethod
    @override
    def name() -> str:
        return "cpu"
