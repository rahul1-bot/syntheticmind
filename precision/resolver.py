import warnings
from typing import cast

import torch

from syntheticmind.precision.double import DoublePrecision
from syntheticmind.precision.half import HalfPrecision
from syntheticmind.precision.mixed import MixedPrecision
from syntheticmind.precision.precision import Precision
from syntheticmind.precision.transformer_engine import TransformerEnginePrecision
from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.types import PrecisionInput, PrecisionType

__all__: list[str] = ["PrecisionResolver"]


class PrecisionResolver:
    @classmethod
    def resolve(
        cls,
        precision: PrecisionInput | None,
        device: str,
        plugins: Precision | list[Precision] | None = None,
        strategy_precision_plugin: Precision | None = None
    ) -> Precision:
        plugin: Precision | None = cls._resolve_plugin(plugins)
        if plugin is not None and strategy_precision_plugin is not None:
            raise MisconfigurationError(
                "A precision plugin was configured through both Trainer(plugins=...) and the strategy. "
                "Configure it in exactly one place."
            )
        configured_plugin: Precision | None = plugin or strategy_precision_plugin
        if configured_plugin is not None:
            if precision is not None:
                raise MisconfigurationError(
                    f"Received precision={precision!r} together with a precision plugin. "
                    "Configure either precision or the plugin, not both."
                )
            return configured_plugin
        normalized_precision: PrecisionType = cls._normalize(precision)
        if normalized_precision == "16-mixed" and device == "cpu":
            warnings.warn(
                "Trainer(accelerator='cpu', precision='16-mixed') is unsupported because CPU AMP "
                "does not support float16. Using precision='bf16-mixed' instead.",
                UserWarning,
                stacklevel=3
            )
            normalized_precision: PrecisionType = "bf16-mixed"
        match normalized_precision:
            case "16-true" | "bf16-true":
                return HalfPrecision(normalized_precision)
            case "32-true":
                return Precision()
            case "64-true":
                return DoublePrecision()
            case "16-mixed" | "bf16-mixed":
                return MixedPrecision(normalized_precision, device)
            case "transformer-engine":
                return TransformerEnginePrecision(weights_dtype=torch.bfloat16)
            case "transformer-engine-float16":
                return TransformerEnginePrecision(weights_dtype=torch.float16)

    @classmethod
    def _resolve_plugin(
        cls,
        plugins: Precision | list[Precision] | None
    ) -> Precision | None:
        del cls
        if plugins is None:
            return None
        if isinstance(plugins, Precision):
            return plugins
        if not isinstance(plugins, list):
            raise TypeError(
                f"plugins must be a Precision instance, a list of Precision instances, or None, "
                f"got {type(plugins).__name__}."
            )
        if len(plugins) > 1:
            raise MisconfigurationError(
                f"Trainer accepts at most one precision plugin, got {len(plugins)}."
            )
        if not plugins:
            return None
        plugin: object = plugins[0]
        if not isinstance(plugin, Precision):
            raise TypeError(
                f"plugins must contain Precision instances, got {type(plugin).__name__}."
            )
        return plugin

    @classmethod
    def _normalize(cls, precision: PrecisionInput | None) -> PrecisionType:
        del cls
        if precision is None:
            return "32-true"
        if isinstance(precision, bool):
            raise ValueError(
                f"Precision {precision!r} is invalid. Use 64, 32, 16, 'bf16', or an explicit precision mode."
            )
        if not isinstance(precision, (int, str)):
            raise TypeError(
                f"precision must be an int, a string, or None, got {type(precision).__name__}."
            )
        aliases: dict[int | str, PrecisionType] = {
            64: "64-true",
            32: "32-true",
            16: "16-mixed",
            "64": "64-true",
            "32": "32-true",
            "16": "16-mixed",
            "bf16": "bf16-mixed"
        }
        if precision in aliases:
            if precision in {16, "16", "bf16"}:
                warnings.warn(
                    f"precision={precision!r} is supported for compatibility but discouraged. "
                    f"Use precision={aliases[precision]!r} instead.",
                    UserWarning,
                    stacklevel=3
                )
            return aliases[precision]
        supported: tuple[str, ...] = (
            "16-true",
            "16-mixed",
            "bf16-true",
            "bf16-mixed",
            "32-true",
            "64-true",
            "transformer-engine",
            "transformer-engine-float16"
        )
        if precision not in supported:
            raise ValueError(
                f"Precision {precision!r} is invalid. Allowed precision values: "
                f"{supported + (64, 32, 16, '64', '32', '16', 'bf16')}."
            )
        return cast(PrecisionType, precision)
