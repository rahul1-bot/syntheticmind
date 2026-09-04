# This module:
# 1. Implements Accelerator for NVIDIA CUDA discovery, explicit device
#    selection, process-local binding, and allocator statistics
#
# Design decisions:
# - setup_device calls torch.cuda.set_device as well as recording the device so
#   context-free CUDA allocations inside client code follow the resolved rank
# - Before binding, get_device returns cuda:0 as the conventional provisional
#   device; Trainer replaces it after topology and local-rank resolution
# - Device parsing distinguishes counts, explicit identifiers, and all-visible
#   selection through the shared GPU parser and rejects unavailable IDs
# - Statistics expose allocator current and peak values, retry and out-of-memory
#   counts, plus free and total device bytes without converting units
#
# Author: Rahul Sawhney

from typing import override

import torch

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.accelerators.device_parser import DeviceInput, _AcceleratorDeviceParser
from syntheticmind.primitives.exceptions import MisconfigurationError

__all__: list[str] = ["CUDAAccelerator"]


class CUDAAccelerator(Accelerator):
    # NVIDIA execution backend selected by Trainer when CUDA is requested or
    # available under automatic accelerator resolution.
    def __init__(self) -> None:
        self._device: torch.device | None = None

    @override
    def setup_device(self, device: torch.device) -> None:
        # Validates CUDA identity and mutates this process's active CUDA device.
        if device.type != "cuda":
            raise MisconfigurationError(f"CUDAAccelerator cannot bind {device}.")
        self._device: torch.device | None = device
        torch.cuda.set_device(device)

    @override
    def get_device(self) -> torch.device:
        # Bound process-local device, or cuda:0 before topology setup.
        if self._device is None:
            return torch.device("cuda", 0)
        return self._device

    @override
    def get_device_stats(self, device: torch.device) -> dict[str, float]:
        # Reads allocator and hardware memory counters for the resolved device
        # index; callers synchronize execution before timing-sensitive sampling.
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
        memory_info: tuple[int, int] = torch.cuda.mem_get_info(device_index)
        free_bytes: int = memory_info[0]
        total_bytes: int = memory_info[1]
        statistics["cuda/free_bytes"] = float(free_bytes)
        statistics["cuda/total_bytes"] = float(total_bytes)
        return statistics

    @classmethod
    @override
    def parse_devices(cls, devices: DeviceInput) -> int | list[int]:
        # Resolves public count, explicit ID, comma-separated, and all-device
        # forms against torch.cuda.device_count.
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
        # Converts validated identifiers into ordered CUDA placements used by
        # Strategy local-rank selection.
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
