# This module:
# 1. Implements true float64 parameter, tensor-construction, input, and forward
#    execution policy
#
# Design decisions:
# - True precision converts model storage and floating inputs rather than using
#   autocast, so parameters, activations, and client-created tensors share float64
# - Every default-dtype transition is reversible through _DtypeContext
#
# Author: Rahul Sawhney

from contextlib import AbstractContextManager
from typing import override

import torch

from syntheticmind.plugins.precision.conversion import _DtypeContext
from syntheticmind.plugins.precision.precision import Precision
from syntheticmind.primitives.types import Batch, PrecisionType

__all__: list[str] = ["DoublePrecision"]


class DoublePrecision(Precision):
    # True float64 policy selected through Trainer(precision="64-true").
    precision: PrecisionType = "64-true"

    @override
    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        # Converts floating model parameters and buffers to float64.
        return module.double()

    @override
    def tensor_init_context(self) -> AbstractContextManager[None]:
        # Selects float64 as the default for Strategy-created tensors.
        return _DtypeContext(torch.float64)

    @override
    def module_init_context(self) -> AbstractContextManager[None]:
        # Selects float64 for layers constructed inside configure_model.
        return _DtypeContext(torch.float64)

    @override
    def forward_context(self) -> AbstractContextManager[None]:
        # Selects float64 for free floating tensors created during forward.
        return _DtypeContext(torch.float64)

    @override
    def convert_input(self, data: Batch) -> Batch:
        # Recursively converts floating batch tensors to float64.
        return self._convert_floating_point(data, torch.float64)
