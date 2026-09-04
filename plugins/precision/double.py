from contextlib import AbstractContextManager
from typing import override

import torch

from syntheticmind.plugins.precision.conversion import _DtypeContext
from syntheticmind.plugins.precision.precision import Precision
from syntheticmind.utilities.types import Batch, PrecisionType

__all__: list[str] = ["DoublePrecision"]


class DoublePrecision(Precision):
    precision: PrecisionType = "64-true"

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
        return self._convert_floating_point(data, torch.float64)
