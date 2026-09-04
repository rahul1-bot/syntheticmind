from typing import ClassVar

import torch
from pydantic import BaseModel, ConfigDict

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.accelerators.device_parser import DeviceInput

__all__: list[str] = ["DeviceTopology"]


class DeviceTopology(BaseModel):
    devices: int | tuple[int, ...]
    parallel_devices: tuple[torch.device, ...]
    num_nodes: int
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        strict=True,
        validate_default=True
    )

    @classmethod
    def resolve(
        cls: type[DeviceTopology],
        accelerator: Accelerator,
        devices: DeviceInput,
        num_nodes: int,
        configured_parallel_devices: list[torch.device] | None = None
    ) -> DeviceTopology:
        if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes < 1:
            raise ValueError(f"`num_nodes` must be positive, got {num_nodes!r}.")
        parsed_devices: int | list[int]
        if configured_parallel_devices:
            parsed_devices: int | list[int] = cls._parse_configured_devices(
                accelerator,
                configured_parallel_devices
            )
        else:
            resolved_devices: DeviceInput = (
                accelerator.auto_device_count()
                if devices == "auto"
                else devices
            )
            parsed_devices: int | list[int] = accelerator.parse_devices(
                resolved_devices
            )
        parallel_devices: list[torch.device] = accelerator.get_parallel_devices(
            parsed_devices
        )
        resolved_device_selection: int | tuple[int, ...] = (
            parsed_devices
            if isinstance(parsed_devices, int)
            else tuple(parsed_devices)
        )
        return cls(
            devices=resolved_device_selection,
            parallel_devices=tuple(parallel_devices),
            num_nodes=num_nodes
        )

    @classmethod
    def _parse_configured_devices(
        cls: type[DeviceTopology],
        accelerator: Accelerator,
        configured_parallel_devices: list[torch.device]
    ) -> int | list[int]:
        del cls
        expected_device_type: str = accelerator.name()
        actual_device_types: set[str] = set()
        device: torch.device
        for device in configured_parallel_devices:
            actual_device_types.add(device.type)
        if actual_device_types != {expected_device_type}:
            raise ValueError(
                f"Configured devices {configured_parallel_devices} do not match "
                f"accelerator={expected_device_type!r}."
            )
        if expected_device_type == "cpu":
            return accelerator.parse_devices(len(configured_parallel_devices))
        device_ids: list[int] = []
        device_index: int
        for device_index, device in enumerate(configured_parallel_devices):
            resolved_index: int = (
                device.index if device.index is not None else device_index
            )
            device_ids.append(resolved_index)
        return accelerator.parse_devices(device_ids)
