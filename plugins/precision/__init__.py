# This module:
# 1. Exposes full, true-half, mixed, double, and Transformer Engine precision
#    policies plus delayed-scaling configuration and recipe contracts
#
# Design decisions:
# - Clients select established precision names through Trainer or supply one
#   concrete Precision instance; the accelerator connector validates compatibility
# - Optional Transformer Engine imports remain isolated behind its adapter so
#   importing ordinary precision policies never requires the NVIDIA package
#
# Author: Rahul Sawhney

from syntheticmind.plugins.precision.double import DoublePrecision
from syntheticmind.plugins.precision.half import HalfPrecision
from syntheticmind.plugins.precision.mixed import MixedPrecision
from syntheticmind.plugins.precision.precision import Precision
from syntheticmind.plugins.precision.transformer_engine import (
    DelayedScalingConfiguration,
    DelayedScalingRecipe,
    TransformerEnginePrecision
)

__all__: list[str] = [
    "DoublePrecision",
    "DelayedScalingConfiguration",
    "DelayedScalingRecipe",
    "HalfPrecision",
    "MixedPrecision",
    "Precision",
    "TransformerEnginePrecision"
]
