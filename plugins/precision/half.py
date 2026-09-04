# This module:
# 1. Implements true float16 and bfloat16 parameter, construction, input, and
#    forward execution policy without autocast or gradient scaling
#
# Design decisions:
# - True half precision converts model storage and floating inputs permanently;
#   clients requiring float32 master weights use MixedPrecision instead
# - Float16 and bfloat16 share one lifecycle because only their target dtype
#   differs and neither uses a GradScaler in true-precision mode
#
# Author: Rahul Sawhney

from contextlib import AbstractContextManager
from typing import Literal, override

import torch

from syntheticmind.plugins.precision.conversion import _DtypeContext
from syntheticmind.plugins.precision.precision import Precision
from syntheticmind.primitives.types import Batch, PrecisionType

__all__: list[str] = ["HalfPrecision"]


class HalfPrecision(Precision):
    # True half-storage policy selected through ``"16-true"`` or
    # ``"bf16-true"``.
    precision: PrecisionType = "16-true"

    def __init__(
        self,
        precision: Literal["16-true", "bf16-true"] = "16-true"
    ) -> None:
        # Validates the true-half mode and resolves its concrete torch dtype.
        if precision not in {"16-true", "bf16-true"}:
            raise ValueError(
                f"HalfPrecision requires '16-true' or 'bf16-true', got {precision!r}."
            )
        self.precision: PrecisionType = precision
        self._dtype: torch.dtype = (
            torch.float16 if precision == "16-true" else torch.bfloat16
        )

    @override
    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        # Converts floating model parameters and buffers to the selected dtype.
        return module.to(dtype=self._dtype)

    @override
    def tensor_init_context(self) -> AbstractContextManager[None]:
        # Selects the true-half dtype for Strategy-created tensors.
        return _DtypeContext(self._dtype)

    @override
    def module_init_context(self) -> AbstractContextManager[None]:
        # Selects the true-half dtype for configure_model construction.
        return _DtypeContext(self._dtype)

    @override
    def forward_context(self) -> AbstractContextManager[None]:
        # Selects the true-half dtype for free tensors created during forward.
        return _DtypeContext(self._dtype)

    @override
    def convert_input(self, data: Batch) -> Batch:
        # Recursively converts floating batch tensors to the selected dtype.
        return self._convert_floating_point(data, self._dtype)
