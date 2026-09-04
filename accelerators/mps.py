# This module:
# 1. Implements Accelerator for Apple Metal Performance Shaders discovery,
#    binding, selection, and memory statistics
#
# Design decisions:
# - Apple hosts expose one usable MPS accelerator to torch, so automatic count is
#   one and distributed MPS compatibility is rejected later by Trainer topology
# - setup_device records placement without a process-global selection API because
#   MPS has no torch.cuda.set_device equivalent
# - Device parsing still uses the shared GPU-style parser, making unavailable or
#   duplicate identifiers fail consistently with CUDA
# - Statistics retain Apple's current, driver, and recommended maximum memory
#   counters as raw bytes
#
# Author: Rahul Sawhney

from typing import override

import torch

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.accelerators.device_parser import DeviceInput, _AcceleratorDeviceParser
from syntheticmind.primitives.exceptions import MisconfigurationError

__all__: list[str] = ["MPSAccelerator"]


class MPSAccelerator(Accelerator):
    # Apple Silicon execution backend used for local macOS training and
    # evaluation when selected explicitly or through automatic resolution.
    def __init__(self) -> None:
        self._device: torch.device = torch.device("mps")

    @override
    def setup_device(self, device: torch.device) -> None:
        # Binds only MPS devices without process-global backend mutation.
        if device.type != "mps":
            raise MisconfigurationError(f"MPSAccelerator cannot bind {device}.")
        self._device: torch.device = device

    @override
    def get_device(self) -> torch.device:
        return self._device

    @override
    def get_device_stats(self, device: torch.device) -> dict[str, float]:
        # Reads the live MPS allocator, driver allocation, and recommended
        # working-set ceiling in bytes.
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
        # Resolves GPU-style public device input against the single visible MPS
        # slot.
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
        # Converts validated identifiers into MPS torch.device objects.
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
