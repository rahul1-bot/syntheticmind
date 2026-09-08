from typing import Literal, cast, override

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from syntheticmind.primitives.types import Batch, ModelOutput

__all__: list[str] = ["ActivationCheckpointing"]

type ActivationRngPolicy = Literal["preserve", "advance"]


class ActivationCheckpointing(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        rng_policy: ActivationRngPolicy = "preserve"
    ) -> None:
        super().__init__()
        if not isinstance(module, nn.Module):
            raise TypeError(
                f"ActivationCheckpointing requires torch.nn.Module, got "
                f"{type(module).__name__}."
            )
        if rng_policy not in {"preserve", "advance"}:
            raise ValueError(f"Unsupported activation RNG policy: {rng_policy!r}.")
        self._module: nn.Module = module
        self._rng_policy: ActivationRngPolicy = rng_policy

    @override
    def forward(
        self,
        *args: Batch,
        **kwargs: Batch
    ) -> ModelOutput:
        preserve_rng_state: bool = self._rng_policy == "preserve"
        output: ModelOutput = cast(
            ModelOutput,
            checkpoint(
                self._module,
                *args,
                use_reentrant=False,
                preserve_rng_state=preserve_rng_state,
                **kwargs
            )
        )
        return output

    @property
    def module(self) -> nn.Module:
        return self._module

    @property
    def rng_policy(self) -> ActivationRngPolicy:
        return self._rng_policy
