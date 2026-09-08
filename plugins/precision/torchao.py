from typing import Literal, override

import torch
from torchao.float8 import Float8LinearConfig, convert_to_float8_training

from syntheticmind.plugins.precision.half import HalfPrecision
from syntheticmind.primitives.types import PrecisionType

__all__: list[str] = ["TorchAOFloat8Precision"]

type Float8Recipe = Literal["tensorwise", "rowwise", "rowwise_with_gw_hp"]


class TorchAOFloat8Precision(HalfPrecision):
    precision: PrecisionType = "torchao-float8"

    def __init__(self, recipe: Float8Recipe = "rowwise") -> None:
        if recipe not in {"tensorwise", "rowwise", "rowwise_with_gw_hp"}:
            raise ValueError(f"Unsupported TorchAO float8 recipe: {recipe!r}.")
        super().__init__("bf16-true")
        self.precision: PrecisionType = "torchao-float8"
        self._recipe: Float8Recipe = recipe

    @override
    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        converted_module: torch.nn.Module = super().convert_module(module)
        configuration: Float8LinearConfig = Float8LinearConfig.from_recipe_name(
            self._recipe
        )
        convert_to_float8_training(converted_module, config=configuration)
        return converted_module

    @property
    def recipe(self) -> Float8Recipe:
        return self._recipe
