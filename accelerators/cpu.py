import os
import time
from typing import override

import torch

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.accelerators.device_parser import DeviceInput
from syntheticmind.exceptions import MisconfigurationError

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

    @override
    def get_device_stats(self, device: torch.device) -> dict[str, float]:
        if device.type != "cpu":
            raise MisconfigurationError(f"CPUAccelerator cannot inspect {device}.")
        statistics: dict[str, float] = {
            "cpu/process_time_seconds": time.process_time(),
            "cpu/count": float(os.cpu_count() or 1)
        }
        if hasattr(os, "getloadavg"):
            load_average: tuple[float, float, float] = os.getloadavg()
            statistics["cpu/load_average_1m"] = load_average[0]
            statistics["cpu/load_average_5m"] = load_average[1]
            statistics["cpu/load_average_15m"] = load_average[2]
        return statistics

    @classmethod
    @override
    def parse_devices(cls, devices: DeviceInput) -> int | list[int]:
        del cls
        normalized_devices: int | str | list[int] = devices
        if isinstance(normalized_devices, str) and normalized_devices.strip().isdigit():
            normalized_devices: int | str | list[int] = int(normalized_devices)
        if isinstance(normalized_devices, bool) or not isinstance(normalized_devices, int):
            raise TypeError("CPUAccelerator requires a positive integer device count.")
        if normalized_devices <= 0:
            raise TypeError("CPUAccelerator requires a positive integer device count.")
        return normalized_devices

    @classmethod
    @override
    def get_parallel_devices(
        cls,
        devices: int | list[int]
    ) -> list[torch.device]:
        del cls
        if not isinstance(devices, int):
            raise TypeError("CPUAccelerator requires an integer device count.")
        return [torch.device("cpu")] * devices

    @classmethod
    @override
    def auto_device_count(cls) -> int:
        del cls
        return 1

    @classmethod
    @override
    def is_available(cls) -> bool:
        del cls
        return True

    @classmethod
    @override
    def name(cls) -> str:
        del cls
        return "cpu"
