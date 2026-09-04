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
from syntheticmind.utilities.types import Batch, PrecisionType

__all__: list[str] = [
    "DelayedScalingConfiguration",
    "DelayedScalingRecipe",
    "TransformerEnginePrecision"
]


class TransformerEnginePrecision(Precision):
    precision: PrecisionType = "transformer-engine"

    def __init__(
        self,
        *,
        weights_dtype: torch.dtype,
        recipe: DelayedScalingConfiguration | DelayedScalingRecipe | None = None,
        replace_layers: bool | None = None,
        fallback_compute_dtype: torch.dtype | None = None
    ) -> None:
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
        return _DtypeContext(self.weights_dtype)

    @contextmanager
    @override
    def module_init_context(self) -> Iterator[None]:
        stack: ExitStack = ExitStack()
        with stack:
            if self.replace_layers:
                stack.enter_context(self._adapter.replacement_context())
            stack.enter_context(self.tensor_init_context())
            yield

    @contextmanager
    @override
    def forward_context(self) -> Iterator[None]:
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
        return self._convert_floating_point(data, self.weights_dtype)

    @override
    def convert_output(self, data: Batch) -> Batch:
        return self._convert_floating_point(data, torch.get_default_dtype())

    def _replace_layers(self, module: nn.Module) -> None:
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
