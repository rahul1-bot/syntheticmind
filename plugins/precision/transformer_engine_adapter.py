# This module:
# 1. Isolates dynamic imports, version detection, recipe construction, autocast,
#    and layer constructors for the optional NVIDIA Transformer Engine package
#
# Design decisions:
# - Ordinary SyntheticMind import never imports Transformer Engine; availability
#   requires a discoverable package at version 2.18.0 or newer
# - All dynamically acquired values are narrowed at this boundary so the
#   precision policy does not depend on broad optional-package objects
# - Layer replacement uses a reversible patch context during configure_model,
#   while post-construction recursive replacement consumes the same constructor
#   properties directly
#
# Author: Rahul Sawhney

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
    # Optional dependency boundary constructed only after availability succeeds.
    def __init__(self) -> None:
        # Imports the exact recipe and PyTorch namespaces consumed by the
        # precision policy after availability and version validation.
        self._recipe_module: ModuleType = importlib.import_module(
            "transformer_engine.common.recipe"
        )
        self._pytorch_module: ModuleType = importlib.import_module(
            "transformer_engine.pytorch"
        )

    @classmethod
    def is_available(cls: type[_TransformerEngineAdapter]) -> bool:
        # Detects an already imported package or verifies installed metadata and
        # minimum supported version without raising optional-dependency errors.
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
        # Converts strict SyntheticMind configuration into the current NVIDIA
        # Format enum and DelayedScaling constructor contract.
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
        # Creates the Transformer Engine FP8 autocast context with the resolved
        # delayed-scaling recipe.
        autocast_factory: Callable[..., AbstractContextManager[None]] = cast(
            Callable[..., AbstractContextManager[None]],
            getattr(self._pytorch_module, "autocast")
        )
        return autocast_factory(enabled=True, recipe=recipe)

    def replacement_context(
        self
    ) -> AbstractContextManager[dict[str, _ModuleConstructor]]:
        # Temporarily redirects torch.nn Linear and LayerNorm construction to
        # Transformer Engine implementations during configure_model.
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
        # Narrows the optional package's Linear class to the shared constructor
        # contract used by patching and recursive replacement.
        return cast(
            _ModuleConstructor,
            getattr(self._pytorch_module, "Linear")
        )

    @property
    def layer_norm_constructor(self) -> _ModuleConstructor:
        # Narrows the optional package's LayerNorm class to the shared constructor
        # contract used by patching and recursive replacement.
        return cast(
            _ModuleConstructor,
            getattr(self._pytorch_module, "LayerNorm")
        )
