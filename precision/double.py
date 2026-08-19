from contextlib import AbstractContextManager
from typing import override

import torch

from syntheticmind.precision.conversion import _DtypeContext, _FloatingPointConverter
from syntheticmind.precision.precision import Precision
from syntheticmind.utilities.types import Batch, PrecisionType

__all__: list[str] = ["DoublePrecision"]


class DoublePrecision(Precision):
    precision: PrecisionType = "64-true"

    def __init__(self) -> None:
        self._converter: _FloatingPointConverter = _FloatingPointConverter()

    @override
    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        return module.double()

    @override
    def tensor_init_context(self) -> AbstractContextManager[None]:
        return _DtypeContext(torch.float64)

    @override
    def module_init_context(self) -> AbstractContextManager[None]:
        return _DtypeContext(torch.float64)

    @override
    def forward_context(self) -> AbstractContextManager[None]:
        return _DtypeContext(torch.float64)

    @override
    def convert_input(self, data: Batch) -> Batch:
        return self._converter.convert(data, torch.float64)
