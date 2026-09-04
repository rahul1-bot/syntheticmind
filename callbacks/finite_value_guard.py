from typing import override

import syntheticmind
import torch

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module

__all__: list[str] = ["FiniteValueGuard"]


class FiniteValueGuard(Callback):
    def __init__(self, check_gradients: bool = False) -> None:
        super().__init__()
        if type(check_gradients) is not bool:
            raise TypeError("`check_gradients` must be a bool.")
        self.check_gradients: bool = check_gradients

    @override
    def on_before_backward(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        loss: torch.Tensor
    ) -> None:
        if bool(torch.isfinite(loss.detach()).all()):
            return
        raise FloatingPointError(
            "Non-finite loss detected before backward at "
            f"epoch={module.current_epoch}, global_step={trainer.global_step}."
        )

    @override
    def on_after_backward(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        if not self.check_gradients:
            return
        name: str
        parameter: torch.nn.Parameter
        for name, parameter in module.named_parameters():
            gradient: torch.Tensor | None = parameter.grad
            if gradient is None:
                continue
            gradient_values: torch.Tensor = (
                gradient.coalesce().values()
                if gradient.is_sparse
                else gradient
            )
            if bool(torch.isfinite(gradient_values.detach()).all()):
                continue
            raise FloatingPointError(
                f"Non-finite gradient detected for parameter {name!r} before "
                f"optimizer step at epoch={module.current_epoch}, "
                f"global_step={trainer.global_step}."
            )

    @override
    def __repr__(self) -> str:
        return f"FiniteValueGuard(check_gradients={self.check_gradients})"
