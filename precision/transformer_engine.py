import importlib
import importlib.util
import sys
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, ExitStack, contextmanager
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType, TracebackType
from typing import Literal, Protocol, cast, override

import torch
from packaging.version import InvalidVersion, Version

from syntheticmind.precision.conversion import _DtypeContext, _FloatingPointConverter
from syntheticmind.precision.precision import Precision
from syntheticmind.utilities.types import Batch, PrecisionType

__all__: list[str] = ["TransformerEnginePrecision"]


class _RecipeModule(Protocol):
    DelayedScaling: Callable[..., object]
    Format: object


class DelayedScaling(Protocol):
    margin: int
    interval: int
    fp8_format: object
    amax_history_len: int
    amax_compute_algo: object


class _TransformerEngineModule(Protocol):
    Linear: Callable[..., torch.nn.Module]
    LayerNorm: Callable[..., torch.nn.Module]

    def fp8_autocast(
        self,
        *,
        enabled: bool,
        fp8_recipe: object
    ) -> AbstractContextManager[object]: ...


class _ClassReplacementContext:
    def __init__(
        self,
        replacements: list[tuple[object, str, object]]
    ) -> None:
        self._replacements: list[tuple[object, str, object]] = replacements
        self._originals: list[tuple[object, str, object]] = [
            (namespace, name, getattr(namespace, name))
            for namespace, name, _ in replacements
        ]

    def __enter__(self) -> None:
        namespace: object
        name: str
        replacement: object
        for namespace, name, replacement in self._replacements:
            setattr(namespace, name, replacement)

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None
    ) -> Literal[False]:
        del exception_type, exception, traceback
        namespace: object
        name: str
        original: object
        for namespace, name, original in self._originals:
            setattr(namespace, name, original)
        return False


class TransformerEnginePrecision(Precision):
    precision: PrecisionType = "transformer-engine"

    def __init__(
        self,
        *,
        weights_dtype: torch.dtype,
        recipe: Mapping[str, object] | DelayedScaling | None = None,
        replace_layers: bool | None = None,
        fallback_compute_dtype: torch.dtype | None = None
    ) -> None:
        if not self._dependency_available():
            raise ModuleNotFoundError(
                "Transformer Engine precision requires transformer_engine>=0.11.0."
            )
        recipe_module: _RecipeModule = self._recipe_module()
        resolved_recipe: object
        if recipe is None:
            resolved_recipe = recipe_module.DelayedScaling()
        elif isinstance(recipe, Mapping):
            recipe_configuration: dict[str, object] = dict(recipe)
            fp8_format: object | None = recipe_configuration.get("fp8_format")
            if isinstance(fp8_format, str):
                recipe_configuration["fp8_format"] = getattr(
                    recipe_module.Format,
                    fp8_format
                )
            resolved_recipe = recipe_module.DelayedScaling(**recipe_configuration)
        else:
            resolved_recipe = recipe
        self.weights_dtype: torch.dtype = weights_dtype
        self.recipe: object = resolved_recipe
        self.replace_layers: bool | None = replace_layers
        self.fallback_compute_dtype: torch.dtype = (
            fallback_compute_dtype or weights_dtype
        )
        self._converter: _FloatingPointConverter = _FloatingPointConverter()

    @override
    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        has_transformer_engine_layer: bool = any(
            "transformer_engine.pytorch" in child.__class__.__module__
            for child in module.modules()
        )
        if not has_transformer_engine_layer and self.replace_layers in {None, True}:
            self._replace_layers(module)
        return module.to(dtype=self.weights_dtype)

    @override
    def tensor_init_context(self) -> AbstractContextManager[None]:
        return _DtypeContext(self.weights_dtype)

    @override
    @contextmanager
    def module_init_context(self) -> Iterator[None]:
        with ExitStack() as stack:
            if self.replace_layers:
                transformer_engine: _TransformerEngineModule = self._pytorch_module()
                stack.enter_context(
                    _ClassReplacementContext(
                        [
                            (torch.nn, "Linear", transformer_engine.Linear),
                            (torch.nn, "LayerNorm", transformer_engine.LayerNorm)
                        ]
                    )
                )
            stack.enter_context(self.tensor_init_context())
            yield

    @override
    @contextmanager
    def forward_context(self) -> Iterator[None]:
        transformer_engine: _TransformerEngineModule = self._pytorch_module()
        with ExitStack() as stack:
            stack.enter_context(_DtypeContext(self.weights_dtype))
            stack.enter_context(
                torch.autocast(
                    device_type="cuda",
                    dtype=self.fallback_compute_dtype
                )
            )
            stack.enter_context(
                transformer_engine.fp8_autocast(
                    enabled=True,
                    fp8_recipe=self.recipe
                )
            )
            yield

    @override
    def convert_input(self, data: Batch) -> Batch:
        return self._converter.convert(data, self.weights_dtype)

    @override
    def convert_output(self, data: Batch) -> Batch:
        return self._converter.convert(data, torch.get_default_dtype())

    def _replace_layers(self, module: torch.nn.Module) -> None:
        transformer_engine: _TransformerEngineModule = self._pytorch_module()
        name: str
        child: torch.nn.Module
        for name, child in module.named_children():
            replacement: torch.nn.Module
            if isinstance(child, torch.nn.Linear):
                if child.in_features % 8 != 0 or child.out_features % 16 != 0:
                    warnings.warn(
                        f"Transformer Engine requires Linear dimensions divisible by 8 and 16; "
                        f"layer {name!r} has shape ({child.in_features}, {child.out_features}) and was not replaced.",
                        UserWarning,
                        stacklevel=3
                    )
                    continue
                replacement = transformer_engine.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None
                )
                replacement.load_state_dict(child.state_dict())
                setattr(module, name, replacement)
            elif isinstance(child, torch.nn.LayerNorm):
                normalized_shape: tuple[int, ...] = child.normalized_shape
                replacement = transformer_engine.LayerNorm(
                    normalized_shape[0],
                    eps=child.eps
                )
                replacement.load_state_dict(child.state_dict())
                setattr(module, name, replacement)
            else:
                self._replace_layers(child)

    @staticmethod
    def _dependency_available() -> bool:
        if "transformer_engine" in sys.modules:
            return True
        try:
            if importlib.util.find_spec("transformer_engine") is None:
                return False
            return Version(version("transformer_engine")) >= Version("0.11.0")
        except (
            ImportError,
            InvalidVersion,
            ModuleNotFoundError,
            PackageNotFoundError,
            ValueError
        ):
            return False

    @staticmethod
    def _recipe_module() -> _RecipeModule:
        module: ModuleType = importlib.import_module(
            "transformer_engine.common.recipe"
        )
        return cast(_RecipeModule, module)

    @staticmethod
    def _pytorch_module() -> _TransformerEngineModule:
        module: ModuleType = importlib.import_module("transformer_engine.pytorch")
        return cast(_TransformerEngineModule, module)
