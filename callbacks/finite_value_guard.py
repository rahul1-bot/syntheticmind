# This module:
# 1. Defines FiniteValueGuard, the optimization-boundary callback that rejects
#    non-finite losses before backward and optionally rejects non-finite
#    gradients before optimizer stepping
#
# Design decisions:
# - Loss validation happens before backward, preventing a poisoned scalar from
#   contaminating gradient buffers and optimizer state
# - Gradient validation happens immediately after backward, while unscaled or
#   precision-managed gradients remain visible and before clipping or stepping
# - Sparse gradients are coalesced and inspected through their stored values,
#   avoiding dense materialization of potentially enormous tensors
# - Failures include epoch, global step, and parameter identity, giving client
#   experiments a precise numerical fault boundary instead of a later corrupted
#   checkpoint or opaque optimizer failure
#
# Author: Rahul Sawhney

from typing import override

import syntheticmind
import torch

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module

__all__: list[str] = ["FiniteValueGuard"]


class FiniteValueGuard(Callback):
    # Fail-fast numerical guard for client training runs. The default checks
    # only the loss; gradient scanning is opt-in because it adds a full gradient
    # traversal to every backward pass.
    def __init__(self, check_gradients: bool = False) -> None:
        # Args:
        #     check_gradients: Whether every populated dense or sparse gradient
        #         is checked after backward. Default: ``False``.
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
        # Rejects NaN or infinity before the precision plugin performs backward.
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
        # Locates the first non-finite parameter gradient after backward. None
        # gradients are valid for unused or currently disabled parameters.
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
        # Makes the cost-bearing gradient policy explicit in diagnostics.
        return f"FiniteValueGuard(check_gradients={self.check_gradients})"
