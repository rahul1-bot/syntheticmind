# This module:
# 1. Defines the Trainer-owned deterministic execution policy over PyTorch
#    algorithms, warning mode, cuDNN benchmarking and deterministic flags, and
#    CUDA BLAS workspace configuration
# 2. Exposes process-state inspection and restoration used by launchers without
#    transferring policy ownership to process transport
#
# Design decisions:
# - Deterministic execution and seeded random streams remain separate: rng owns
#   stochastic trajectory, while this class constrains algorithm selection
# - ``"warn"`` enables deterministic algorithms but reports unavailable kernels
#   instead of failing; bool values retain PyTorch's strict enable/disable meaning
# - Deterministic mode defaults cuDNN benchmark to false unless the client
#   explicitly requests true, in which case the conflict remains visible through
#   a warning rather than silent override
# - CUBLAS_WORKSPACE_CONFIG is exported for CUDA deterministic algorithms before
#   launched workers begin execution
# - Launchers query and restore flags through this owner so no second component
#   mutates torch deterministic state independently
#
# Author: Rahul Sawhney

import os
import warnings
from typing import Literal

import torch

from syntheticmind.primitives.exceptions import MisconfigurationError

__all__: list[str] = []

type _DeterminismMode = bool | Literal["warn"] | None


class _Determinism:
    # Stateful Trainer policy retaining the requested mode and resolved benchmark
    # choice while exposing live process-global settings for launcher transfer.
    def __init__(
        self,
        mode: _DeterminismMode,
        benchmark: bool | None
    ) -> None:
        # Validates mode and benchmark, resolves their interaction, records policy,
        # and applies it before model or data execution begins.
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
        # Returns the client-requested deterministic algorithm policy.
        return self._mode

    @property
    def algorithms_enabled(self) -> bool:
        # Reads live PyTorch algorithm state for launcher transfer.
        return torch.are_deterministic_algorithms_enabled()

    @property
    def warn_only_enabled(self) -> bool:
        # Reads whether nondeterministic operations warn instead of failing.
        return torch.is_deterministic_algorithms_warn_only_enabled()

    @property
    def benchmark_enabled(self) -> bool:
        # Reads live cuDNN benchmark selection for launcher transfer.
        return torch.backends.cudnn.benchmark

    @property
    def cudnn_deterministic_enabled(self) -> bool:
        # Reads live cuDNN deterministic-kernel selection for launcher transfer.
        return torch.backends.cudnn.deterministic

    def restore(
        self,
        algorithms_enabled: bool,
        warn_only_enabled: bool,
        benchmark_enabled: bool,
        cudnn_deterministic_enabled: bool
    ) -> None:
        # Restores algorithm, warning, benchmark, and cuDNN deterministic state
        # captured from another process owner.
        torch.use_deterministic_algorithms(
            algorithms_enabled,
            warn_only=warn_only_enabled
        )
        torch.backends.cudnn.benchmark: bool = benchmark_enabled
        torch.backends.cudnn.deterministic: bool = cudnn_deterministic_enabled

    def _apply(self) -> None:
        # Applies only explicitly resolved settings and exports the CUDA BLAS
        # workspace policy whenever deterministic execution is requested.
        if self._benchmark is not None:
            torch.backends.cudnn.benchmark: bool = self._benchmark
        if self._mode == "warn":
            torch.use_deterministic_algorithms(True, warn_only=True)
        elif type(self._mode) is bool:
            torch.use_deterministic_algorithms(self._mode)
        if self._mode:
            os.environ.update({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
