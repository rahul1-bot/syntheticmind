# This module:
# 1. Implements NVIDIA Transformer Engine precision with delayed-scaling FP8
#    autocast, float16 or bfloat16 weights, fallback autocast, and optional layer
#    replacement
# 2. Re-exports the strict configuration and structural live recipe contracts
#    clients use to customize delayed scaling
#
# Design decisions:
# - The policy requires transformer_engine 2.18.0 or newer and the current
#   autocast(enabled=True, recipe=...) API; unsupported installations fail at
#   configuration rather than degrading silently
# - Weights remain float16 or bfloat16 while eligible compute enters Transformer
#   Engine FP8 autocast with an explicit fallback compute dtype
# - replace_layers=True patches construction during configure_model;
#   replace_layers=None performs recursive post-construction replacement only
#   when the model contains no existing Transformer Engine layer
# - Linear replacement requires both dimensions divisible by sixteen and
#   LayerNorm replacement requires one normalized dimension; unsupported layers
#   warn and remain native rather than being converted incorrectly
# - Outputs return to the current default floating dtype at the plugin boundary
#
# Author: Rahul Sawhney

import warnings
from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from typing import override

import torch
from torch import nn

from syntheticmind.plugins.precision.conversion import _DtypeContext
from syntheticmind.plugins.precision.precision import Precision
from syntheticmind.plugins.precision.transformer_engine_adapter import _TransformerEngineAdapter
from syntheticmind.plugins.precision.transformer_engine_contracts import (
    DelayedScalingConfiguration,
    DelayedScalingRecipe
)
from syntheticmind.primitives.types import Batch, PrecisionType

__all__: list[str] = [
    "DelayedScalingConfiguration",
    "DelayedScalingRecipe",
    "TransformerEnginePrecision"
]


class TransformerEnginePrecision(Precision):
    # Optional FP8 execution policy for CUDA Transformer workloads. Clients may
    # supply DelayedScalingConfiguration or a live DelayedScalingRecipe and
    # choose whether native Linear and LayerNorm layers are replaced.
    precision: PrecisionType = "transformer-engine"

    def __init__(
        self,
        *,
        weights_dtype: torch.dtype,
        recipe: DelayedScalingConfiguration | DelayedScalingRecipe | None = None,
        replace_layers: bool | None = None,
        fallback_compute_dtype: torch.dtype | None = None
    ) -> None:
        # Validates dependency, weight and fallback dtypes, layer-replacement
        # policy, and recipe before any model structure mutates.
        if not _TransformerEngineAdapter.is_available():
            raise ModuleNotFoundError(
                "Transformer Engine precision requires transformer_engine>=2.18.0."
            )
        if weights_dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("Transformer Engine weights require float16 or bfloat16.")
        if replace_layers is not None and type(replace_layers) is not bool:
            raise TypeError("replace_layers must be a bool or None.")
        resolved_compute_dtype: torch.dtype = fallback_compute_dtype or weights_dtype
        if resolved_compute_dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("Transformer Engine fallback compute requires float16 or bfloat16.")
        self._adapter: _TransformerEngineAdapter = _TransformerEngineAdapter()
        if recipe is None:
            resolved_recipe: DelayedScalingRecipe = self._adapter.create_recipe(
                DelayedScalingConfiguration()
            )
        elif isinstance(recipe, DelayedScalingConfiguration):
            resolved_recipe: DelayedScalingRecipe = self._adapter.create_recipe(recipe)
        elif isinstance(recipe, DelayedScalingRecipe):
            resolved_recipe: DelayedScalingRecipe = recipe
        else:
            raise TypeError(
                "recipe must be DelayedScalingConfiguration, DelayedScalingRecipe, or None."
            )
        self.precision: PrecisionType = (
            "transformer-engine-float16"
            if weights_dtype == torch.float16
            else "transformer-engine"
        )
        self.weights_dtype: torch.dtype = weights_dtype
        self.recipe: DelayedScalingRecipe = resolved_recipe
        self.replace_layers: bool | None = replace_layers
        self.fallback_compute_dtype: torch.dtype = resolved_compute_dtype

    @override
    def convert_module(self, module: nn.Module) -> nn.Module:
        # Avoids double conversion when Transformer Engine layers already exist,
        # optionally replaces eligible native layers, then converts weight storage.
        has_transformer_engine_layer: bool = False
        child: nn.Module
        for child in module.modules():
            if "transformer_engine.pytorch" in child.__class__.__module__:
                has_transformer_engine_layer: bool = True
                break
        if not has_transformer_engine_layer and self.replace_layers in {None, True}:
            self._replace_layers(module)
        return module.to(dtype=self.weights_dtype)

    @override
    def tensor_init_context(self) -> AbstractContextManager[None]:
        # Applies the selected true weight dtype to client tensor construction.
        return _DtypeContext(self.weights_dtype)

    @contextmanager
    @override
    def module_init_context(self) -> Iterator[None]:
        # Combines optional constructor replacement with the selected true weight
        # dtype during client configure_model.
        stack: ExitStack = ExitStack()
        with stack:
            if self.replace_layers:
                stack.enter_context(self._adapter.replacement_context())
            stack.enter_context(self.tensor_init_context())
            yield

    @contextmanager
    @override
    def forward_context(self) -> Iterator[None]:
        # Enters weight default dtype, CUDA fallback autocast, and Transformer
        # Engine FP8 autocast as one reversible forward transaction.
        stack: ExitStack = ExitStack()
        with stack:
            stack.enter_context(_DtypeContext(self.weights_dtype))
            stack.enter_context(
                torch.autocast(
                    device_type="cuda",
                    dtype=self.fallback_compute_dtype
                )
            )
            stack.enter_context(self._adapter.autocast(self.recipe))
            yield

    @override
    def convert_input(self, data: Batch) -> Batch:
        # Converts floating client inputs to the selected Transformer Engine
        # weight dtype while preserving non-floating leaves.
        return self._convert_floating_point(data, self.weights_dtype)

    @override
    def convert_output(self, data: Batch) -> Batch:
        # Returns floating outputs to the caller's ambient default dtype.
        return self._convert_floating_point(data, torch.get_default_dtype())

    def _replace_layers(self, module: nn.Module) -> None:
        # Recursively replaces eligible native Linear and one-dimensional
        # LayerNorm children while copying their complete state dictionaries.
        child_name: str
        child: nn.Module
        for child_name, child in module.named_children():
            replacement: nn.Module
            if isinstance(child, nn.Linear):
                if child.in_features % 16 != 0 or child.out_features % 16 != 0:
                    warnings.warn(
                        "Transformer Engine Linear dimensions must be divisible by 16; "
                        f"layer {child_name!r} was not replaced.",
                        UserWarning,
                        stacklevel=3
                    )
                    continue
                replacement: nn.Module = self._adapter.linear_constructor(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None
                )
                replacement.load_state_dict(child.state_dict())
                setattr(module, child_name, replacement)
            elif isinstance(child, nn.LayerNorm):
                normalized_shape: tuple[int, ...] = child.normalized_shape
                if len(normalized_shape) != 1:
                    warnings.warn(
                        "Transformer Engine LayerNorm replacement requires one dimension; "
                        f"layer {child_name!r} was not replaced.",
                        UserWarning,
                        stacklevel=3
                    )
                    continue
                replacement: nn.Module = self._adapter.layer_norm_constructor(
                    normalized_shape[0],
                    eps=child.eps
                )
                replacement.load_state_dict(child.state_dict())
                setattr(module, child_name, replacement)
            else:
                self._replace_layers(child)
