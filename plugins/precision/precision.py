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
    precision: PrecisionType = "32-true"
    scaler: torch.GradScaler | None = None

    def connect[SchedulerT](
        self,
        model: torch.nn.Module,
        optimizers: list[torch.optim.Optimizer],
        lr_schedulers: list[SchedulerT]
    ) -> tuple[torch.nn.Module, list[torch.optim.Optimizer], list[SchedulerT]]:
        return model, optimizers, lr_schedulers

    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        return module

    def tensor_init_context(self) -> AbstractContextManager[None]:
        return nullcontext()

    def module_init_context(self) -> AbstractContextManager[None]:
        return nullcontext()

    def forward_context(self) -> AbstractContextManager[None]:
        return nullcontext()

    def train_step_context(self) -> AbstractContextManager[None]:
        return self.forward_context()

    def val_step_context(self) -> AbstractContextManager[None]:
        return self.forward_context()

    def test_step_context(self) -> AbstractContextManager[None]:
        return self.forward_context()

    def predict_step_context(self) -> AbstractContextManager[None]:
        return self.forward_context()

    def convert_input(self, data: Batch) -> Batch:
        return data

    def convert_output(self, data: Batch) -> Batch:
        return data

    def pre_backward(self, tensor: torch.Tensor, module: Module) -> torch.Tensor:
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
        parameters: list[torch.nn.Parameter] = list(self.main_params(optimizer))
        if parameters:
            torch.nn.utils.clip_grad_value_(parameters, clip_val)

    def clip_grad_by_norm(
        self,
        optimizer: torch.optim.Optimizer,
        clip_val: int | float
    ) -> None:
        parameters: list[torch.nn.Parameter] = list(self.main_params(optimizer))
        if parameters:
            torch.nn.utils.clip_grad_norm_(parameters, clip_val)

    def main_params(
        self,
        optimizer: torch.optim.Optimizer
    ) -> Iterator[torch.nn.Parameter]:
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
        del optimizer

    def state_dict(self) -> StateDict:
        return {}

    def load_state_dict(self, state_dict: StateDict) -> None:
        del state_dict

    @override
    def on_save_checkpoint(self, checkpoint: CheckpointDict) -> None:
        del checkpoint

    @override
    def on_load_checkpoint(self, checkpoint: CheckpointDict) -> None:
        del checkpoint

    def teardown(self) -> None:
        pass

    def _wrap_closure(
        self,
        model: Module,
        optimizer: torch.optim.Optimizer,
        closure: OptimizerClosure
    ) -> OptimizerClosureResult:
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
