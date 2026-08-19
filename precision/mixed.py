from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from types import TracebackType
from typing import Literal, cast, override

import torch

from syntheticmind.core.module import Module
from syntheticmind.precision.precision import Precision
from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.types import ClipAlgorithm, PrecisionType, StateDict

__all__: list[str] = ["MixedPrecision"]


type _ContextFactory = Callable[..., AbstractContextManager[object]]


class _AutocastClearCacheOnExit:
    def __init__(
        self,
        context_manager: AbstractContextManager[object],
        clear_cache: bool
    ) -> None:
        self._context_manager: AbstractContextManager[object] = context_manager
        self._clear_cache: bool = clear_cache

    def __enter__(self) -> object:
        return self._context_manager.__enter__()

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None
    ) -> bool | None:
        output: bool | None = self._context_manager.__exit__(
            exception_type,
            exception,
            traceback
        )
        if self._clear_cache:
            torch.clear_autocast_cache()
        return output

    def __call__(
        self,
        function: Callable[..., object]
    ) -> Callable[..., object]:
        decorator: Callable[
            [Callable[..., object]],
            Callable[..., object]
        ] = cast(
            Callable[[Callable[..., object]], Callable[..., object]],
            self._context_manager
        )
        return decorator(function)


class MixedPrecision(Precision):
    precision: PrecisionType = "16-mixed"

    def __init__(
        self,
        precision: Literal["16-mixed", "bf16-mixed"],
        device: str,
        scaler: torch.GradScaler | None = None
    ) -> None:
        if precision not in {"16-mixed", "bf16-mixed"}:
            raise ValueError(
                f"MixedPrecision requires '16-mixed' or 'bf16-mixed', got {precision!r}."
            )
        if scaler is not None and precision == "bf16-mixed":
            raise MisconfigurationError(
                f"precision='bf16-mixed' does not use a GradScaler, got {scaler!r}."
            )
        self.precision = precision
        self.device: str = device
        self.scaler: torch.GradScaler | None = (
            torch.GradScaler(device=device)
            if scaler is None and precision == "16-mixed"
            else scaler
        )
        self._unscaled_optimizers: set[int] = set()

    def autocast_context_manager(self) -> torch.autocast:
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
        dtype: torch.dtype = (
            torch.float16 if self.precision == "16-mixed" else torch.bfloat16
        )
        original_no_grad: _ContextFactory = cast(_ContextFactory, torch.no_grad)
        original_inference_mode: _ContextFactory = cast(
            _ContextFactory,
            torch.inference_mode
        )

        def wrap_context_factory(
            context_factory: _ContextFactory,
            clear_cache: Callable[..., bool]
        ) -> _ContextFactory:
            def wrapped_factory(
                *args: object,
                **kwargs: object
            ) -> AbstractContextManager[object]:
                return _AutocastClearCacheOnExit(
                    context_factory(*args, **kwargs),
                    clear_cache(*args, **kwargs)
                )

            return wrapped_factory

        def always_clear_cache(
            *args: object,
            **kwargs: object
        ) -> bool:
            del args, kwargs
            return True

        def clear_cache_for_inference_mode(
            *args: object,
            **kwargs: object
        ) -> bool:
            return bool(args[0] if args else kwargs.get("mode", True))

        try:
            setattr(
                torch,
                "no_grad",
                wrap_context_factory(
                    original_no_grad,
                    always_clear_cache
                )
            )
            setattr(
                torch,
                "inference_mode",
                wrap_context_factory(
                    original_inference_mode,
                    clear_cache_for_inference_mode
                )
            )
            with torch.autocast(self.device, dtype=dtype):
                yield
        finally:
            setattr(torch, "no_grad", original_no_grad)
            setattr(torch, "inference_mode", original_inference_mode)

    @override
    def pre_backward(self, tensor: torch.Tensor, module: Module) -> torch.Tensor:
        scaler: torch.GradScaler | None = self.scaler
        scaled_tensor: torch.Tensor = scaler.scale(tensor) if scaler is not None else tensor
        return super().pre_backward(scaled_tensor, module)

    @override
    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        model: Module,
        closure: Callable[[], object],
        **kwargs: object
    ) -> object:
        scaler: torch.GradScaler | None = self.scaler
        if scaler is None:
            return super().optimizer_step(optimizer, model, closure, **kwargs)
        if isinstance(optimizer, torch.optim.LBFGS):
            raise MisconfigurationError(
                "precision='16-mixed' is incompatible with the LBFGS optimizer."
            )
        closure_result: object = closure()
        should_skip_step: bool = closure_result is None and model.automatic_optimization
        if should_skip_step:
            self._after_closure(model, optimizer)
            return closure_result
        try:
            if not self._optimizer_handles_unscaling(optimizer):
                self.unscale_gradients(optimizer)
            self._after_closure(model, optimizer)
            scaler_step: Callable[..., object] = scaler.step
            step_output: object = scaler_step(optimizer, **kwargs)
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
        if clip_val <= 0:
            return
        parameters: list[torch.nn.Parameter] = list(self.main_params(optimizer))
        has_gradients: bool = any(
            parameter.grad is not None for parameter in parameters
        )
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
        scaler: torch.GradScaler | None = self.scaler
        return cast(StateDict, scaler.state_dict()) if scaler is not None else {}

    @override
    def load_state_dict(self, state_dict: StateDict) -> None:
        scaler: torch.GradScaler | None = self.scaler
        if scaler is not None:
            scaler.load_state_dict(state_dict)

    @override
    def teardown(self) -> None:
        self._unscaled_optimizers: set[int] = set()

    def _optimizer_handles_unscaling(
        self,
        optimizer: torch.optim.Optimizer
    ) -> bool:
        return bool(getattr(optimizer, "_step_supports_amp_scaling", False))
