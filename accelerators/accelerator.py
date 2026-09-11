# This module:
# 1. Defines Accelerator, the structural backend contract for device discovery,
#    configuration parsing, process-local binding, and native statistics
#
# Design decisions:
# - Discovery and parsing are class-level operations because Trainer must resolve
#   availability and devices before a backend instance owns process-local state
# - setup_device is instance behavior because CUDA binding and current-device
#   selection are process-local mutations performed after rank resolution
# - Device statistics belong to the backend that defines their counters and
#   units; callbacks schedule observations without branching on device type
# - The Protocol contains no default behavior or implementation registry, keeping
#   concrete selection with the Trainer accelerator connector
#
# Author: Rahul Sawhney

from typing import Protocol, runtime_checkable

import torch

from syntheticmind.accelerators.device_parser import DeviceInput

__all__: list[str] = ["Accelerator"]


@runtime_checkable
class Accelerator(Protocol):
    # Interchangeable device backend consumed by Trainer and Strategy. A custom
    # client accelerator implements this complete contract and is supplied to
    # Trainer(accelerator=...), without registering global aliases.
    def setup_device(self, device: torch.device) -> None: ...

    # Process-local device selected after topology and rank resolution.
    def get_device(self) -> torch.device: ...

    # Backend-native scalar counters for the requested bound device.
    def get_device_stats(self, device: torch.device) -> dict[str, float]: ...

    # Validates public device input and resolves counts or explicit identifiers.
    @classmethod
    def parse_devices(cls, devices: DeviceInput) -> int | list[int]: ...

    # Ordered concrete devices assigned to local worker ranks.
    @classmethod
    def get_parallel_devices(
        cls,
        devices: int | list[int]
    ) -> list[torch.device]: ...

    # Default usable device count for automatic selection.
    @classmethod
    def auto_device_count(cls) -> int: ...

    # Whether the current torch build and host expose a usable backend.
    @classmethod
    def is_available(cls) -> bool: ...

    # Canonical lower-case configuration name.
    @classmethod
    def name(cls) -> str: ...
