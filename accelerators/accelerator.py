from typing import Protocol, runtime_checkable

import torch

from syntheticmind.accelerators.device_parser import DeviceInput

__all__: list[str] = ["Accelerator"]


@runtime_checkable
class Accelerator(Protocol):
    def setup_device(self, device: torch.device) -> None: ...

    def get_device(self) -> torch.device: ...

    def get_device_stats(self, device: torch.device) -> dict[str, float]: ...

    @classmethod
    def parse_devices(cls, devices: DeviceInput) -> int | list[int]: ...

    @classmethod
    def get_parallel_devices(
        cls,
        devices: int | list[int]
    ) -> list[torch.device]: ...

    @classmethod
    def auto_device_count(cls) -> int: ...

    @classmethod
    def is_available(cls) -> bool: ...

    @classmethod
    def name(cls) -> str: ...
