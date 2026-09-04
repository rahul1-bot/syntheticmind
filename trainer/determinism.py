import os
import warnings
from typing import Literal

import torch

from syntheticmind.primitives.exceptions import MisconfigurationError

__all__: list[str] = []

type _DeterminismMode = bool | Literal["warn"] | None


class _Determinism:
    def __init__(
        self,
        mode: _DeterminismMode,
        benchmark: bool | None
    ) -> None:
        if mode is not None and type(mode) is not bool and mode != "warn":
            raise MisconfigurationError(
                "`deterministic` must be True, False, 'warn', or None."
            )
        if benchmark is not None and type(benchmark) is not bool:
            raise TypeError(
                f"`benchmark` must be a bool or None, got {benchmark!r}."
            )
        resolved_benchmark: bool | None = benchmark
        if mode:
            if resolved_benchmark is None:
                resolved_benchmark: bool | None = False
            elif resolved_benchmark:
                warnings.warn(
                    "`deterministic` and `benchmark=True` were both requested. "
                    "cuDNN benchmark selection can prevent deterministic execution.",
                    stacklevel=3
                )
        self._mode: _DeterminismMode = mode
        self._benchmark: bool | None = resolved_benchmark
        self._apply()

    @property
    def mode(self) -> _DeterminismMode:
        return self._mode

    @property
    def algorithms_enabled(self) -> bool:
        return torch.are_deterministic_algorithms_enabled()

    @property
    def warn_only_enabled(self) -> bool:
        return torch.is_deterministic_algorithms_warn_only_enabled()

    @property
    def benchmark_enabled(self) -> bool:
        return torch.backends.cudnn.benchmark

    @property
    def cudnn_deterministic_enabled(self) -> bool:
        return torch.backends.cudnn.deterministic

    def restore(
        self,
        algorithms_enabled: bool,
        warn_only_enabled: bool,
        benchmark_enabled: bool,
        cudnn_deterministic_enabled: bool
    ) -> None:
        torch.use_deterministic_algorithms(
            algorithms_enabled,
            warn_only=warn_only_enabled
        )
        torch.backends.cudnn.benchmark: bool = benchmark_enabled
        torch.backends.cudnn.deterministic: bool = cudnn_deterministic_enabled

    def _apply(self) -> None:
        if self._benchmark is not None:
            torch.backends.cudnn.benchmark: bool = self._benchmark
        if self._mode == "warn":
            torch.use_deterministic_algorithms(True, warn_only=True)
        elif type(self._mode) is bool:
            torch.use_deterministic_algorithms(self._mode)
        if self._mode:
            os.environ.update({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
