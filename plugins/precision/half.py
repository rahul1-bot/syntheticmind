from contextlib import AbstractContextManager
from typing import Literal, override

import torch

from syntheticmind.plugins.precision.conversion import _DtypeContext
from syntheticmind.plugins.precision.precision import Precision
from syntheticmind.utilities.types import Batch, PrecisionType

__all__: list[str] = ["HalfPrecision"]


class HalfPrecision(Precision):
    precision: PrecisionType = "16-true"

    def __init__(
        self,
        precision: Literal["16-true", "bf16-true"] = "16-true"
    ) -> None:
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
        return module.to(dtype=self._dtype)

    @override
    def tensor_init_context(self) -> AbstractContextManager[None]:
        return _DtypeContext(self._dtype)

    @override
    def module_init_context(self) -> AbstractContextManager[None]:
        return _DtypeContext(self._dtype)

    @override
    def forward_context(self) -> AbstractContextManager[None]:
        return _DtypeContext(self._dtype)

    @override
    def convert_input(self, data: Batch) -> Batch:
        return self._convert_floating_point(data, self._dtype)
