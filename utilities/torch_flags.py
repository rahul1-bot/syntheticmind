import os
import warnings
from typing import Literal

import torch

from syntheticmind.utilities.exceptions import MisconfigurationError

__all__: list[str] = ["configure_torch_flags"]


def configure_torch_flags(
    deterministic: bool | Literal["warn"] | None,
    benchmark: bool | None
) -> None:
    if deterministic is not None and type(deterministic) is not bool and not (
        isinstance(deterministic, str) and deterministic == "warn"
    ):
        raise MisconfigurationError(
            "`deterministic` must be True, False, 'warn', or None."
        )
    if benchmark is not None and type(benchmark) is not bool:
        raise TypeError(f"`benchmark` must be a bool or None, got {benchmark!r}.")

    if deterministic:
        if benchmark is None:
            benchmark = False
        elif benchmark:
            warnings.warn(
                "`deterministic` and `benchmark=True` were both requested. "
                "cuDNN benchmark selection can prevent deterministic execution.",
                stacklevel=3
            )
    if benchmark is not None:
        torch.backends.cudnn.benchmark = benchmark

    if deterministic == "warn":
        torch.use_deterministic_algorithms(True, warn_only=True)
    elif type(deterministic) is bool:
        torch.use_deterministic_algorithms(deterministic)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
