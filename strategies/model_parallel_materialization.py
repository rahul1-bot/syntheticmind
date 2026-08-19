from itertools import chain

import torch
from torch import nn

__all__: list[str] = []


def materialize_distributed_module(
    module: nn.Module,
    device: torch.device
) -> set[str]:
    tensors = chain(module.parameters(), module.buffers())
    if not any(tensor.is_meta for tensor in tensors):
        return set()
    module.to_empty(device=device)
    uninitialized_modules: set[str] = set()
    for submodule in module.modules():
        local_tensors = chain(
            submodule.parameters(recurse=False),
            submodule.buffers(recurse=False)
        )
        if not any(True for _ in local_tensors):
            continue
        reset_parameters: object = getattr(submodule, "reset_parameters", None)
        if callable(reset_parameters):
            reset_parameters()
        else:
            uninitialized_modules.add(type(submodule).__name__)
    return uninitialized_modules
