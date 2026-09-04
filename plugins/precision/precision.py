# This module:
# 1. Defines Precision, the concrete full-precision policy and shared numerical
#    lifecycle for model conversion, step contexts, backward, optimizer stepping,
#    clipping, input conversion, and checkpoint state
# 2. Owns the hook order surrounding backward and optimizer closures for every
#    precision implementation
#
# Design decisions:
# - Precision is a stateful regular base with working float32 behavior rather
#   than an abstract interface; specialized policies override only numerical
#   operations that differ
# - Trainer connects model, optimizers, and schedulers through the precision
#   boundary before execution so a policy may replace them without changing
#   outward orchestration
# - Backward order is fixed here: pre-backward callbacks and model hook,
#   Module.backward through HookDispatcher, loss detachment, then post-backward
#   callbacks and model hook
# - Optimizer closures run before on_before_optimizer_step and clipping; mixed
#   precision may unscale first, while the ordinary policy clips native gradients
# - Manual optimization reuses precision backward and managed optimizer stepping,
#   but automatic clipping is suppressed so the client retains explicit control
# - Floating conversion preserves integer, boolean, byte, string, and custom
#   batch leaves while recursively converting tensors inside supported containers
#
# Author: Rahul Sawhney

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from functools import partial
from typing import cast, override

import torch

from syntheticmind.core.module import Module
from syntheticmind.core.hooks import CheckpointHooks
from syntheticmind.primitives.types import (
    BackwardInputs,
    Batch,
    CheckpointDict,
    ClipAlgorithm,
    OptimizerClosure,
    OptimizerClosureResult,
    OptimizerStateValue,
    OptimizerStepOutput,
    OptimizerStepValue,
    PrecisionType,
    StateDict
)

__all__: list[str] = ["Precision"]


class Precision(CheckpointHooks):
    # Full float32 numerical policy and extension base. Clients usually select a
    # precision string through Trainer; custom policies subclass Precision only
    # when they own a genuinely different conversion, context, backward, step, or
    # checkpoint lifecycle.
    precision: PrecisionType = "32-true"
    scaler: torch.GradScaler | None = None

    def connect[SchedulerT](
        self,
        model: torch.nn.Module,
        optimizers: list[torch.optim.Optimizer],
        lr_schedulers: list[SchedulerT]
    ) -> tuple[torch.nn.Module, list[torch.optim.Optimizer], list[SchedulerT]]:
        # Connects or replaces the model, optimizers, and schedulers before
        # Strategy setup. Full precision preserves each identity unchanged.
        return model, optimizers, lr_schedulers

    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        # Converts model parameter storage before placement; float32 policy is an
        # identity because client construction dtype remains authoritative.
        return module

    def tensor_init_context(self) -> AbstractContextManager[None]:
        # Context for free tensor construction during Strategy initialization.
        return nullcontext()

    def module_init_context(self) -> AbstractContextManager[None]:
        # Context for client configure_model construction.
        return nullcontext()

    def forward_context(self) -> AbstractContextManager[None]:
        # Numerical context shared by train, validation, test, and prediction
        # steps unless a specialized policy overrides stage behavior.
        return nullcontext()

    def train_step_context(self) -> AbstractContextManager[None]:
        # Training-step context; defaults to the common forward context.
        return self.forward_context()

    def val_step_context(self) -> AbstractContextManager[None]:
        # Validation-step context; defaults to the common forward context.
        return self.forward_context()

    def test_step_context(self) -> AbstractContextManager[None]:
        # Test-step context; defaults to the common forward context.
        return self.forward_context()

    def predict_step_context(self) -> AbstractContextManager[None]:
        # Prediction-step context; defaults to the common forward context.
        return self.forward_context()

    def convert_input(self, data: Batch) -> Batch:
        # Converts a device-resident batch before forward. Full precision keeps
        # the client's dtypes unchanged.
        return data

    def convert_output(self, data: Batch) -> Batch:
        # Converts returned values after forward. Full precision is an identity.
        return data

    def pre_backward(self, tensor: torch.Tensor, module: Module) -> torch.Tensor:
        # Dispatches callback and model pre-backward hooks with the exact tensor
        # the selected precision policy will differentiate.
        module.trainer._hook_dispatcher.call_callbacks(
            module,
            "on_before_backward",
            tensor
        )
        module.trainer._hook_dispatcher.call_module(
            module,
            "on_before_backward",
            module.on_before_backward,
            tensor
        )
        return tensor

    def backward(
        self,
        tensor: torch.Tensor,
        model: Module,
        optimizer: torch.optim.Optimizer | None,
        gradient: torch.Tensor | None = None,
        retain_graph: bool | None = None,
        create_graph: bool = False,
        inputs: BackwardInputs = None
    ) -> None:
        # Invokes the model's overridable backward operation through
        # HookDispatcher so profiling and active hook context remain coherent.
        del optimizer
        model.trainer._hook_dispatcher.call_module(
            model,
            "backward",
            model.backward,
            tensor,
            gradient,
            retain_graph,
            create_graph,
            inputs
        )

    def post_backward(self, tensor: torch.Tensor, module: Module) -> torch.Tensor:
        # Detaches the reporting loss and dispatches post-backward hooks while
        # populated gradients remain untouched by clipping or optimizer state.
        detached_tensor: torch.Tensor = tensor.detach()
        module.trainer._hook_dispatcher.call_callbacks(module, "on_after_backward")
        module.trainer._hook_dispatcher.call_module(
            module,
            "on_after_backward",
            module.on_after_backward
        )
        return detached_tensor

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        model: Module,
        closure: OptimizerClosure,
        **kwargs: OptimizerStepValue
    ) -> OptimizerStepOutput:
        # Wraps the closure so pre-step hooks and automatic clipping execute after
        # gradient production and before native optimizer mutation.
        wrapped_closure: OptimizerClosure = partial(
            self._wrap_closure,
            model,
            optimizer,
            closure
        )
        return cast(
            OptimizerStepOutput,
            optimizer.step(closure=wrapped_closure, **kwargs)
        )

    def clip_gradients(
        self,
        optimizer: torch.optim.Optimizer,
        clip_val: int | float = 0.0,
        gradient_clip_algorithm: ClipAlgorithm = "norm"
    ) -> None:
        # Applies positive norm or value clipping to the optimizer's main
        # parameters; zero disables clipping without traversing parameter groups.
        if clip_val <= 0:
            return
        match gradient_clip_algorithm:
            case "norm":
                self.clip_grad_by_norm(optimizer, clip_val)
            case "value":
                self.clip_grad_by_value(optimizer, clip_val)

    def clip_grad_by_value(
        self,
        optimizer: torch.optim.Optimizer,
        clip_val: int | float
    ) -> None:
        # Clamps each populated gradient element to the configured magnitude.
        parameters: list[torch.nn.Parameter] = list(self.main_params(optimizer))
        if parameters:
            torch.nn.utils.clip_grad_value_(parameters, clip_val)

    def clip_grad_by_norm(
        self,
        optimizer: torch.optim.Optimizer,
        clip_val: int | float
    ) -> None:
        # Rescales populated gradients to the configured global norm ceiling.
        parameters: list[torch.nn.Parameter] = list(self.main_params(optimizer))
        if parameters:
            torch.nn.utils.clip_grad_norm_(parameters, clip_val)

    def main_params(
        self,
        optimizer: torch.optim.Optimizer
    ) -> Iterator[torch.nn.Parameter]:
        # Yields native Parameter objects from every optimizer group and rejects
        # malformed custom optimizer group structures before clipping.
        parameter_group: dict[str, OptimizerStateValue]
        parameter_groups: list[dict[str, OptimizerStateValue]] = cast(
            list[dict[str, OptimizerStateValue]],
            optimizer.param_groups
        )
        for parameter_group in parameter_groups:
            parameters: OptimizerStateValue = parameter_group["params"]
            if not isinstance(parameters, list):
                raise TypeError(
                    "Optimizer parameter groups must contain a list, got "
                    f"{type(parameters).__name__}."
                )
            parameter: OptimizerStateValue
            for parameter in parameters:
                if not isinstance(parameter, torch.nn.Parameter):
                    raise TypeError(
                        "Optimizer parameter groups must contain Parameters, got "
                        f"{type(parameter).__name__}."
                    )
                yield parameter

    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        # Full precision has no scaling state. Mixed policies override this
        # operation before hooks or clipping inspect gradients.
        del optimizer

    def state_dict(self) -> StateDict:
        # Serializes numerical continuation state; full precision contributes none.
        return {}

    def load_state_dict(self, state_dict: StateDict) -> None:
        # Restores numerical continuation state; full precision accepts no state.
        del state_dict

    @override
    def on_save_checkpoint(self, checkpoint: CheckpointDict) -> None:
        # Precision checkpoint hook reserved for policies with payload-level state.
        del checkpoint

    @override
    def on_load_checkpoint(self, checkpoint: CheckpointDict) -> None:
        # Precision load hook reserved for policies requiring pre-state setup.
        del checkpoint

    def teardown(self) -> None:
        # Releases precision-owned runtime resources; full precision owns none.
        pass

    def _wrap_closure(
        self,
        model: Module,
        optimizer: torch.optim.Optimizer,
        closure: OptimizerClosure
    ) -> OptimizerClosureResult:
        # Executes model forward and backward through the loop closure, then runs
        # the shared pre-step boundary before returning its result to optimizer.
        closure_result: OptimizerClosureResult = closure()
        self._after_closure(model, optimizer)
        return closure_result

    def _clip_gradients(
        self,
        model: Module,
        optimizer: torch.optim.Optimizer,
        clip_val: int | float | None = None,
        gradient_clip_algorithm: ClipAlgorithm | None = None
    ) -> None:
        # Dispatches the model's clipping extension only under automatic
        # optimization; manual clients choose their clipping calls explicitly.
        if not model.automatic_optimization:
            return
        model.trainer._hook_dispatcher.call_module(
            model,
            "configure_gradient_clipping",
            model.configure_gradient_clipping,
            optimizer,
            clip_val,
            gradient_clip_algorithm
        )

    def _after_closure(
        self,
        model: Module,
        optimizer: torch.optim.Optimizer
    ) -> None:
        # Dispatches pre-step hooks while gradients are visible, then applies the
        # model-owned automatic clipping extension.
        model.trainer._hook_dispatcher.call_callbacks(
            model,
            "on_before_optimizer_step",
            optimizer
        )
        model.trainer._hook_dispatcher.call_module(
            model,
            "on_before_optimizer_step",
            model.on_before_optimizer_step,
            optimizer
        )
        self._clip_gradients(
            model,
            optimizer,
            model.trainer.gradient_clip_val,
            model.trainer.gradient_clip_algorithm
        )

    def _convert_floating_point(self, data: Batch, dtype: torch.dtype) -> Batch:
        # Recursively converts only floating tensor leaves and reconstructs dict,
        # list, tuple, and named-tuple topology unchanged.
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype) if data.is_floating_point() else data
        if isinstance(data, dict):
            converted_mapping: dict[str, Batch] = {}
            key: str
            value: Batch
            for key, value in data.items():
                converted_mapping[key]: Batch = self._convert_floating_point(
                    value,
                    dtype
                )
            return converted_mapping
        if isinstance(data, tuple):
            converted_items: list[Batch] = []
            item: Batch
            for item in data:
                converted_items.append(self._convert_floating_point(item, dtype))
            if hasattr(data, "_fields"):
                namedtuple_factory: Callable[[list[Batch]], Batch] = cast(
                    Callable[[list[Batch]], Batch],
                    getattr(type(data), "_make")
                )
                return namedtuple_factory(converted_items)
            return tuple(converted_items)
        if isinstance(data, list):
            converted_items: list[Batch] = []
            item: Batch
            for item in data:
                converted_items.append(self._convert_floating_point(item, dtype))
            return converted_items
        return data
