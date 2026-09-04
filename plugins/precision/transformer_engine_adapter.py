import importlib
import importlib.util
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType
from typing import cast
from unittest.mock import patch

from packaging.version import InvalidVersion, Version
from torch import nn

from syntheticmind.plugins.precision.transformer_engine_contracts import (
    DelayedScalingConfiguration,
    DelayedScalingRecipe
)

__all__: list[str] = []

type _ModuleConstructor = Callable[..., nn.Module]


class _TransformerEngineAdapter:
    def __init__(self) -> None:
        self._recipe_module: ModuleType = importlib.import_module(
            "transformer_engine.common.recipe"
        )
        self._pytorch_module: ModuleType = importlib.import_module(
            "transformer_engine.pytorch"
        )

    @classmethod
    def is_available(cls: type[_TransformerEngineAdapter]) -> bool:
        del cls
        if "transformer_engine" in sys.modules:
            return True
        try:
            if importlib.util.find_spec("transformer_engine") is None:
                return False
            return Version(version("transformer_engine")) >= Version("2.18.0")
        except (
            ImportError,
            InvalidVersion,
            ModuleNotFoundError,
            PackageNotFoundError,
            ValueError
        ):
            return False

    def create_recipe(
        self,
        configuration: DelayedScalingConfiguration
    ) -> DelayedScalingRecipe:
        format_type: type[Enum] = cast(
            type[Enum],
            getattr(self._recipe_module, "Format")
        )
        fp8_format: Enum = getattr(format_type, configuration.fp8_format)
        constructor: Callable[..., DelayedScalingRecipe] = cast(
            Callable[..., DelayedScalingRecipe],
            getattr(self._recipe_module, "DelayedScaling")
        )
        return constructor(
            margin=configuration.margin,
            fp8_format=fp8_format,
            amax_history_len=configuration.amax_history_len,
            amax_compute_algo=configuration.amax_compute_algo
        )

    def autocast(
        self,
        recipe: DelayedScalingRecipe
    ) -> AbstractContextManager[None]:
        autocast_factory: Callable[..., AbstractContextManager[None]] = cast(
            Callable[..., AbstractContextManager[None]],
            getattr(self._pytorch_module, "autocast")
        )
        return autocast_factory(enabled=True, recipe=recipe)

    def replacement_context(
        self
    ) -> AbstractContextManager[dict[str, _ModuleConstructor]]:
        return cast(
            AbstractContextManager[dict[str, _ModuleConstructor]],
            patch.multiple(
                nn,
                Linear=self.linear_constructor,
                LayerNorm=self.layer_norm_constructor
            )
        )

    @property
    def linear_constructor(self) -> _ModuleConstructor:
        return cast(
            _ModuleConstructor,
            getattr(self._pytorch_module, "Linear")
        )

    @property
    def layer_norm_constructor(self) -> _ModuleConstructor:
        return cast(
            _ModuleConstructor,
            getattr(self._pytorch_module, "LayerNorm")
        )
