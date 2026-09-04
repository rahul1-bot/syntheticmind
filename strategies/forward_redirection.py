from collections.abc import Callable
from typing import Literal, override

from torch import nn
from torch.nn.parallel import DistributedDataParallel

from syntheticmind.core.module import Module
from syntheticmind.utilities.types import ModelOutput, StepArgument

__all__: list[str] = []

type StepName = Literal[
    "predict_step",
    "test_step",
    "training_step",
    "validation_step"
]


class _ForwardRedirection:
    def __init__(self) -> None:
        self._active: bool = False
        self._module: Module | None = None
        self._original_forward: Callable[..., ModelOutput] | None = None
        self._step: Callable[..., ModelOutput] | None = None
        self._wrapper: nn.Module | None = None

    def execute(
        self,
        wrapper: nn.Module,
        module: Module,
        step_name: StepName,
        *args: StepArgument,
        **kwargs: StepArgument
    ) -> ModelOutput:
        step: Callable[..., ModelOutput] = getattr(module, step_name)
        if not callable(step):
            raise TypeError(f"Module step {step_name!r} is not callable.")
        if wrapper is module:
            return step(*args, **kwargs)
        if self._active:
            raise RuntimeError("A wrapped module step is already active.")
        original_forward: Callable[..., ModelOutput] = module.forward
        self._active: bool = True
        self._module: Module | None = module
        self._original_forward: Callable[..., ModelOutput] | None = original_forward
        self._step: Callable[..., ModelOutput] | None = step
        self._wrapper: nn.Module | None = wrapper
        module.forward: Callable[..., ModelOutput] = self._redirected_forward
        try:
            output: ModelOutput = wrapper(*args, **kwargs)
            self._after_outer_forward(wrapper, module)
            return output
        finally:
            module.forward: Callable[..., ModelOutput] = original_forward
            self._active: bool = False
            self._module: Module | None = None
            self._original_forward: Callable[..., ModelOutput] | None = None
            self._step: Callable[..., ModelOutput] | None = None
            self._wrapper: nn.Module | None = None

    def _redirected_forward(
        self,
        *args: StepArgument,
        **kwargs: StepArgument
    ) -> ModelOutput:
        module: Module | None = self._module
        original_forward: Callable[..., ModelOutput] | None = self._original_forward
        step: Callable[..., ModelOutput] | None = self._step
        wrapper: nn.Module | None = self._wrapper
        if module is None or original_forward is None or step is None or wrapper is None:
            raise RuntimeError("Wrapped module step redirection is not active.")
        module.forward: Callable[..., ModelOutput] = original_forward
        output: ModelOutput = step(*args, **kwargs)
        self._after_inner_forward(wrapper, module)
        return output

    def _after_inner_forward(self, wrapper: nn.Module, module: Module) -> None:
        del wrapper, module

    def _after_outer_forward(self, wrapper: nn.Module, module: Module) -> None:
        del wrapper, module


class _DDPForwardRedirection(_ForwardRedirection):
    @override
    def _after_inner_forward(self, wrapper: nn.Module, module: Module) -> None:
        if isinstance(wrapper, DistributedDataParallel) and not module.automatic_optimization:
            wrapper.require_backward_grad_sync: bool = False

    @override
    def _after_outer_forward(self, wrapper: nn.Module, module: Module) -> None:
        if isinstance(wrapper, DistributedDataParallel) and not module.automatic_optimization:
            wrapper.require_backward_grad_sync: bool = True
