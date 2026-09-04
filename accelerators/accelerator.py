from typing import Protocol, runtime_checkable

import torch

from syntheticmind.accelerators.device_parser import DeviceInput

__all__: list[str] = ["Accelerator"]


@runtime_checkable
class Accelerator(Protocol):
    def setup_device(self, device: torch.device) -> None: ...

    def get_device(self) -> torch.device: ...

    def get_device_stats(self, device: torch.device) -> dict[str, float]: ...

    @staticmethod
    def parse_devices(devices: DeviceInput) -> int | list[int]: ...

    @staticmethod
    def get_parallel_devices(devices: int | list[int]) -> list[torch.device]: ...

    @staticmethod
    def auto_device_count() -> int: ...

    @staticmethod
    def is_available() -> bool: ...

    @staticmethod
    def name() -> str: ...
