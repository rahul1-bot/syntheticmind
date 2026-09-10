# This module:
# 1. Adapts TorchAO float8 training to the Precision lifecycle
#
# Design decisions:
# - This explicit optional import boundary does not run during core imports
# - Native Linear layers convert once; existing subclasses keep their semantics
# - BF16 parameter storage is distinct from FP8 matrix multiplication
# - Recipe identity participates in checkpoint compatibility before restoration
#
# Author: Rahul Sawhney

from typing import Literal, cast, override

import torch
from torch.distributed.tensor import DTensor
from torchao.float8 import Float8LinearConfig, convert_to_float8_training

from syntheticmind.plugins.precision.half import HalfPrecision
from syntheticmind.primitives.types import CheckpointDict, CheckpointValue, PrecisionType, StateDict

__all__: list[str] = ["TorchAOFloat8Precision"]

type Float8Recipe = Literal["tensorwise", "rowwise", "rowwise_with_gw_hp"]


class TorchAOFloat8Precision(HalfPrecision):
    # Supply through Trainer(plugins=[precision]). For TP/FSDP2 plans, convert
    # native regions through this plugin before sharding in configure_model.
    precision: PrecisionType = "torchao-float8"

    @override
    def __init__(self, recipe: Float8Recipe = "rowwise") -> None:
        if recipe not in {"tensorwise", "rowwise", "rowwise_with_gw_hp"}:
            raise ValueError(f"Unsupported TorchAO float8 recipe: {recipe!r}.")
        super().__init__("bf16-true")
        self.precision: PrecisionType = "torchao-float8"
        self._recipe: Float8Recipe = recipe

    @override
    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        # Reject unsupported hardware and already-sharded native layers before
        # changing parameter dtype. Conversion may replace a root Linear itself.
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise RuntimeError("TorchAOFloat8Precision currently requires NVIDIA CUDA.")
        if torch.cuda.get_device_capability() < (8, 9):
            raise RuntimeError("TorchAO float8 requires NVIDIA compute capability 8.9 or newer.")
        region: torch.nn.Module
        for region in module.modules():
            if type(region) is not torch.nn.Linear:
                continue
            if isinstance(region.weight, DTensor):
                raise ValueError("Convert native Linear layers to float8 before applying a DTensor plan.")
            if region.in_features % 16 or region.out_features % 16:
                raise ValueError("Float8 native Linear dimensions must be divisible by 16.")
        configuration: Float8LinearConfig = Float8LinearConfig.from_recipe_name(self._recipe)
        converted_module: torch.nn.Module = super().convert_module(module)
        return convert_to_float8_training(
            converted_module, config=configuration, module_filter_fn=self._select_linear
        )

    @override
    def state_dict(self) -> StateDict:
        return {"schema_version": 1, "recipe": self._recipe}

    @override
    def load_state_dict(self, state_dict: StateDict) -> None:
        if set(state_dict) != {"schema_version", "recipe"}:
            raise ValueError("Float8 checkpoint state requires schema_version and recipe only.")
        if type(state_dict["schema_version"]) is not int or state_dict["schema_version"] != 1:
            raise ValueError("Unsupported float8 precision state schema.")
        if state_dict["recipe"] != self._recipe:
            raise ValueError("Float8 checkpoint recipe differs from the configured recipe.")

    @override
    def on_load_checkpoint(self, checkpoint: CheckpointDict) -> None:
        state: CheckpointValue | None = checkpoint.get(type(self).__qualname__)
        if state is not None:
            if not isinstance(state, dict):
                raise TypeError("Float8 checkpoint state must be a mapping.")
            self.load_state_dict(cast(StateDict, state))

    @property
    def recipe(self) -> Float8Recipe:
        return self._recipe

    def _select_linear(self, module: torch.nn.Module, path: str) -> bool:
        del path
        return type(module) is torch.nn.Linear
