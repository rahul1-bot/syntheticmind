# This module:
# 1. Implements float16 and bfloat16 mixed precision through autocast, optional
#    gradient scaling, unscaling, optimizer stepping, clipping, and resumable
#    scaler state
# 2. Defines the autocast-cache wrapper that keeps nested no-grad and inference
#    contexts from retaining stale cast tensors across evaluation boundaries
#
# Design decisions:
# - Float16 mixed precision owns a GradScaler by default, while bfloat16 mixed
#   precision rejects a scaler because its exponent range does not require one
# - Forward temporarily wraps torch.no_grad and torch.inference_mode so nested
#   evaluation contexts clear the autocast cache on exit; the original callables
#   restore in finally even when client forward fails
# - Float16 closures run before scaler.step, gradients unscale exactly once per
#   optimizer, and on_before_optimizer_step plus clipping observe unscaled values
# - Optimizers that declare internal AMP scaling cannot also use framework
#   clipping because external unscale would duplicate their private operation
# - A skipped automatic step still executes the shared post-closure hooks and
#   clipping boundary, while scaler update occurs only after a real scaler step
# - Scaler state participates in checkpoints and the per-optimizer unscale set is
#   transient runtime state cleared after every step and teardown
#
# Author: Rahul Sawhney

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from functools import partial
from types import TracebackType
from typing import Literal, cast, override

import torch

from syntheticmind.core.module import Module
from syntheticmind.plugins.precision.precision import Precision
from syntheticmind.primitives.exceptions import MisconfigurationError
from syntheticmind.primitives.types import (
    ClipAlgorithm,
    OptimizerClosure,
    OptimizerClosureResult,
    OptimizerStepOutput,
    OptimizerStepValue,
    PrecisionType,
    StateDict
)

__all__: list[str] = ["MixedPrecision"]


class _AutocastCacheContext:
    # Structural context wrapper that delegates no-grad or inference behavior and
    # clears torch's autocast cache only when the enclosed transition requires it.
    def __init__(
        self,
        context_manager: AbstractContextManager[None],
        clear_cache: bool
    ) -> None:
        # Wraps one concrete no-gradient context and records whether its exit
        # must invalidate cached autocast conversions.
        self._context_manager: AbstractContextManager[None] = context_manager
        self._clear_cache: bool = clear_cache

    def __enter__(self) -> None:
        # Delegates entry without replacing the enclosed context's return value.
        return self._context_manager.__enter__()

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None
    ) -> bool | None:
        # Preserves the wrapped context's exception policy and clears cached casts
        # only after that context has completed its own exit behavior.
        output: bool | None = self._context_manager.__exit__(
            exception_type,
            exception,
            traceback
        )
        if self._clear_cache:
            torch.clear_autocast_cache()
        return output


class MixedPrecision(Precision):
    # Autocast policy for float16 or bfloat16 compute. Clients select
    # ``"16-mixed"`` or ``"bf16-mixed"`` through Trainer; a custom GradScaler
    # may be supplied only for float16 execution.
    precision: PrecisionType = "16-mixed"

    def __init__(
        self,
        precision: Literal["16-mixed", "bf16-mixed"],
        device: str,
        scaler: torch.GradScaler | None = None
    ) -> None:
        # Validates the precision/scaler pair, constructs the default float16
        # scaler for the selected device, and starts with no unscaled optimizer.
        if precision not in {"16-mixed", "bf16-mixed"}:
            raise ValueError(
                f"MixedPrecision requires '16-mixed' or 'bf16-mixed', got {precision!r}."
            )
        if scaler is not None and precision == "bf16-mixed":
            raise MisconfigurationError(
                f"precision='bf16-mixed' does not use a GradScaler, got {scaler!r}."
            )
        self.precision: PrecisionType = precision
        self.device: str = device
        self.scaler: torch.GradScaler | None = (
            torch.GradScaler(device=device)
            if scaler is None and precision == "16-mixed"
            else scaler
        )
        self._unscaled_optimizers: set[int] = set()

    def autocast_context_manager(self) -> torch.autocast:
        # Public autocast context for client operations that must match this
        # plugin outside the standard step hooks.
        dtype: torch.dtype = (
            torch.float16 if self.precision == "16-mixed" else torch.bfloat16
        )
        return torch.autocast(
            self.device,
            dtype=dtype,
            cache_enabled=False
        )

    @override
    @contextmanager
    def forward_context(self) -> Iterator[None]:
        # Runs forward under the selected autocast dtype and installs reversible
        # cache-clearing wrappers around nested no-grad and inference contexts.
        dtype: torch.dtype = (
            torch.float16 if self.precision == "16-mixed" else torch.bfloat16
        )
        original_no_grad: Callable[[], AbstractContextManager[None]] = cast(
            Callable[[], AbstractContextManager[None]],
            torch.no_grad
        )
        original_inference_mode: Callable[
            [bool], AbstractContextManager[None]
        ] = cast(
            Callable[[bool], AbstractContextManager[None]],
            torch.inference_mode
        )
        no_grad_factory: Callable[[], _AutocastCacheContext] = partial(
            self._no_grad_context,
            original_no_grad
        )
        inference_mode_factory: Callable[[bool], _AutocastCacheContext] = partial(
            self._inference_mode_context,
            original_inference_mode
        )
        try:
            setattr(torch, "no_grad", no_grad_factory)
            setattr(torch, "inference_mode", inference_mode_factory)
            with torch.autocast(self.device, dtype=dtype):
                yield
        finally:
            setattr(torch, "no_grad", original_no_grad)
            setattr(torch, "inference_mode", original_inference_mode)

    @override
    def pre_backward(self, tensor: torch.Tensor, module: Module) -> torch.Tensor:
        # Scales float16 loss before the shared pre-backward hook sequence;
        # bfloat16 passes the original loss unchanged.
        scaler: torch.GradScaler | None = self.scaler
        scaled_tensor: torch.Tensor = scaler.scale(tensor) if scaler is not None else tensor
        return super().pre_backward(scaled_tensor, module)

    @override
    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        model: Module,
        closure: OptimizerClosure,
        **kwargs: OptimizerStepValue
    ) -> OptimizerStepOutput:
        # Executes the closure, handles automatic skip semantics, unscales once,
        # dispatches hooks and clipping, then performs scaler-controlled optimizer
        # mutation and scale update.
        scaler: torch.GradScaler | None = self.scaler
        if scaler is None:
            return super().optimizer_step(optimizer, model, closure, **kwargs)
        if isinstance(optimizer, torch.optim.LBFGS):
            raise MisconfigurationError(
                "precision='16-mixed' is incompatible with the LBFGS optimizer."
            )
        closure_result: OptimizerClosureResult = closure()
        should_skip_step: bool = closure_result is None and model.automatic_optimization
        if should_skip_step:
            self._after_closure(model, optimizer)
            return closure_result
        try:
            if not self._optimizer_handles_unscaling(optimizer):
                self.unscale_gradients(optimizer)
            self._after_closure(model, optimizer)
            step_output: OptimizerStepOutput = cast(
                OptimizerStepOutput,
                scaler.step(optimizer, **kwargs)
            )
            scaler.update()
            return step_output
        finally:
            self._unscaled_optimizers.discard(id(optimizer))

    @override
    def clip_gradients(
        self,
        optimizer: torch.optim.Optimizer,
        clip_val: int | float = 0.0,
        gradient_clip_algorithm: ClipAlgorithm = "norm"
    ) -> None:
        # Avoids unscale work when no gradient exists, then unscales exactly once
        # before delegating to ordinary norm or value clipping.
        if clip_val <= 0:
            return
        parameters: list[torch.nn.Parameter] = list(self.main_params(optimizer))
        has_gradients: bool = False
        parameter: torch.nn.Parameter
        for parameter in parameters:
            if parameter.grad is not None:
                has_gradients: bool = True
                break
        if not has_gradients:
            return
        self.unscale_gradients(optimizer)
        super().clip_gradients(
            optimizer,
            clip_val=clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm
        )

    @override
    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        # Applies scaler.unscale_ once per optimizer per step and rejects
        # optimizers that own private AMP unscaling when external inspection was
        # requested.
        scaler: torch.GradScaler | None = self.scaler
        if scaler is None or id(optimizer) in self._unscaled_optimizers:
            return
        if self._optimizer_handles_unscaling(optimizer):
            raise RuntimeError(
                f"Gradient clipping is unavailable for {type(optimizer).__name__} because "
                "the optimizer performs AMP unscaling internally."
            )
        scaler.unscale_(optimizer)
        self._unscaled_optimizers.add(id(optimizer))

    @override
    def state_dict(self) -> StateDict:
        # Persists GradScaler trajectory for exact float16 continuation;
        # bfloat16 contributes an empty state.
        scaler: torch.GradScaler | None = self.scaler
        return cast(StateDict, scaler.state_dict()) if scaler is not None else {}

    @override
    def load_state_dict(self, state_dict: StateDict) -> None:
        # Restores scaler growth and overflow state when scaling is active.
        scaler: torch.GradScaler | None = self.scaler
        if scaler is not None:
            scaler.load_state_dict(state_dict)

    @override
    def teardown(self) -> None:
        # Clears transient unscale identities between Trainer lifecycles.
        self._unscaled_optimizers: set[int] = set()

    def _no_grad_context(
        self,
        context_factory: Callable[[], AbstractContextManager[None]]
    ) -> _AutocastCacheContext:
        # Wraps torch.no_grad and always clears cached casts when it exits.
        return _AutocastCacheContext(context_factory(), clear_cache=True)

    def _inference_mode_context(
        self,
        context_factory: Callable[[bool], AbstractContextManager[None]],
        mode: bool = True
    ) -> _AutocastCacheContext:
        # Wraps torch.inference_mode and clears cached casts only when inference
        # mode was enabled rather than explicitly disabled.
        return _AutocastCacheContext(context_factory(mode), clear_cache=mode)

    def _optimizer_handles_unscaling(
        self,
        optimizer: torch.optim.Optimizer
    ) -> bool:
        # Reads PyTorch's optimizer capability marker without assuming a concrete
        # optimizer subclass.
        return bool(getattr(optimizer, "_step_supports_amp_scaling", False))
