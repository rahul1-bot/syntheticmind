from dataclasses import dataclass

import torch

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.accelerators.device_parser import DeviceInput


@dataclass(frozen=True, slots=True)
class DeviceTopology:
    devices: int | list[int]
    parallel_devices: tuple[torch.device, ...]
    num_nodes: int


def resolve_device_topology(
    accelerator: Accelerator,
    devices: DeviceInput,
    num_nodes: int,
    configured_parallel_devices: list[torch.device] | None = None
) -> DeviceTopology:
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes < 1:
        raise ValueError(
            f"`num_nodes` must be a positive integer, but got {num_nodes!r}."
        )

    if configured_parallel_devices:
        expected_device_type: str = accelerator.name()
        actual_device_types: set[str] = {
            device.type for device in configured_parallel_devices
        }
        if actual_device_types != {expected_device_type}:
            raise ValueError(
                f"Configured parallel devices {configured_parallel_devices} do not match "
                f"accelerator={expected_device_type!r}."
            )
        if expected_device_type == "cpu":
            parsed_devices: int | list[int] = accelerator.parse_devices(
                len(configured_parallel_devices)
            )
        else:
            device_ids: list[int] = [
                device.index if device.index is not None else index
                for index, device in enumerate(configured_parallel_devices)
            ]
            parsed_devices = accelerator.parse_devices(device_ids)
    else:
        resolved_devices: DeviceInput = (
            accelerator.auto_device_count()
            if devices == "auto"
            else devices
        )
        parsed_devices = accelerator.parse_devices(resolved_devices)

    parallel_devices: list[torch.device] = accelerator.get_parallel_devices(
        parsed_devices
    )
    return DeviceTopology(
        devices=parsed_devices,
        parallel_devices=tuple(parallel_devices),
        num_nodes=num_nodes
    )
