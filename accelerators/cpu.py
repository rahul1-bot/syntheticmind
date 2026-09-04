# This module:
# 1. Implements Accelerator for CPU execution and multi-process CPU topology
# 2. Exposes process time, logical CPU count, and available system load averages
#    through the device-statistics contract
#
# Design decisions:
# - CPU is always available and is the guaranteed automatic fallback
# - A positive device count represents identical CPU worker slots rather than
#   indexed hardware, so parsing returns the count and parallel devices repeat
#   torch.device("cpu")
# - setup_device validates backend identity but requires no process-global
#   selection operation equivalent to torch.cuda.set_device
# - Load averages are reported only on operating systems that expose them, while
#   process time and CPU count remain portable baseline statistics
#
# Author: Rahul Sawhney

import os
import time
from typing import override

import torch

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.accelerators.device_parser import DeviceInput
from syntheticmind.primitives.exceptions import MisconfigurationError

__all__: list[str] = ["CPUAccelerator"]


class CPUAccelerator(Accelerator):
    # Host execution backend. Clients normally select it through
    # Trainer(accelerator="cpu"); custom code may also construct and pass the
    # instance directly.
    def __init__(self) -> None:
        # Establishes the immutable host placement used before and after setup.
        self._device: torch.device = torch.device("cpu")

    @override
    def setup_device(self, device: torch.device) -> None:
        # Binds only CPU devices and rejects cross-backend placement early.
        if device.type != "cpu":
            raise MisconfigurationError(f"CPUAccelerator cannot bind {device}.")
        self._device: torch.device = device

    @override
    def get_device(self) -> torch.device:
        # Returns the process-local host placement selected during setup.
        return self._device

    @override
    def get_device_stats(self, device: torch.device) -> dict[str, float]:
        # Collects backend-native host counters without allocating tensors or
        # synchronizing model execution.
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
        # Accepts a positive integer count or its numeric string form; explicit
        # CPU identifier lists have no meaningful hardware interpretation.
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
        # Creates one identical CPU placement entry per requested worker process.
        del cls
        if not isinstance(devices, int):
            raise TypeError("CPUAccelerator requires an integer device count.")
        return [torch.device("cpu")] * devices

    @classmethod
    @override
    def auto_device_count(cls) -> int:
        # Uses one process by default instead of interpreting host core count as
        # an automatic distributed-training request.
        del cls
        return 1

    @classmethod
    @override
    def is_available(cls) -> bool:
        # CPU remains the guaranteed backend for every supported host.
        del cls
        return True

    @classmethod
    @override
    def name(cls) -> str:
        # Returns the canonical Trainer configuration identity.
        del cls
        return "cpu"
