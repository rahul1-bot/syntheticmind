from collections.abc import Callable
from typing import cast

import numpy as np
import torch

from syntheticmind.utilities.types import GatherData, GatherResult

__all__: list[str] = ["transform_tensor_collection"]


def transform_tensor_collection(
    data: GatherData,
    device: torch.device,
    transform: Callable[[torch.Tensor], torch.Tensor]
) -> GatherResult:
    tensor: torch.Tensor
    if isinstance(data, torch.Tensor):
        tensor = data.to(device)
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data).to(device)
    elif type(data) is bool:
        tensor = torch.tensor(data, device=device, dtype=torch.uint8)
    elif type(data) is int:
        tensor = torch.tensor(data, device=device, dtype=torch.int)
    elif type(data) is float:
        tensor = torch.tensor(data, device=device, dtype=torch.float)
    elif isinstance(data, dict):
        transformed_mapping: dict[object, GatherResult] = {
            key: transform_tensor_collection(value, device, transform)
            for key, value in data.items()
        }
        return transformed_mapping
    elif isinstance(data, list):
        transformed_list: list[GatherResult] = [
            transform_tensor_collection(value, device, transform)
            for value in data
        ]
        return transformed_list
    elif isinstance(data, tuple):
        transformed_tuple: tuple[GatherResult, ...] = tuple(
            transform_tensor_collection(value, device, transform)
            for value in data
        )
        if hasattr(data, "_fields"):
            maker: Callable[
                [tuple[GatherResult, ...]],
                tuple[GatherResult, ...]
            ] = cast(
                Callable[
                    [tuple[GatherResult, ...]],
                    tuple[GatherResult, ...]
                ],
                getattr(type(data), "_make")
            )
            return maker(transformed_tuple)
        return transformed_tuple
    else:
        raise TypeError(
            "`data` must be a tensor, NumPy array, bool, int, float, "
            "or a nested dict, list, or tuple containing those values, got "
            f"{type(data).__name__}."
        )
    return transform(tensor)
