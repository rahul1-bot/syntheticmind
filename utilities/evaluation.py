from contextlib import AbstractContextManager

import torch
import torch.distributed as dist

__all__: list[str] = ["evaluation_context"]


def evaluation_context(inference_mode: bool) -> AbstractContextManager[None]:
    if dist.is_available() and dist.is_initialized() and dist.get_backend() == "gloo":
        return torch.no_grad()
    if inference_mode:
        return torch.inference_mode()
    return torch.no_grad()
