# This module:
# 1. Implements automatic training-step output normalization, zeroing, loss
#    scaling for gradient accumulation, and backward invocation
#
# Design decisions:
# - training_step may return a scalar tensor, a mapping containing ``"loss"``, or
#   None; distributed execution rejects None because rank-divergent skipping
#   would break collective ordering
# - The loss divides by accumulation factor before backward so the summed
#   microbatch gradient matches a mean over the accumulation window
# - The detached normalized loss replaces any client mapping loss for callbacks
#   and logging, preventing batch outputs from retaining the autograd graph
# - Zeroing occurs only when the epoch loop supplies an operation for the current
#   accumulation boundary
#
# Author: Rahul Sawhney

import warnings
from collections.abc import Callable, Mapping

import torch

from syntheticmind.strategies.strategy import Strategy
from syntheticmind.primitives.exceptions import MisconfigurationError
from syntheticmind.primitives.types import (
    OptimizerStepOutput,
    StepOutput,
    TrainingStepOutput
)

__all__: list[str] = []


class _AutomaticOptimization:
    # One automatic closure execution with normalized reporting output.
    def __init__(self, accumulation_factor: int) -> None:
        # Binds the positive loss-scaling factor and starts with no batch output.
        if accumulation_factor < 1:
            raise ValueError("accumulation_factor must be positive.")
        self._accumulation_factor: int = accumulation_factor
        self._outputs: StepOutput = {}

    @torch.enable_grad()
    def run(
        self,
        strategy: Strategy,
        step_operation: Callable[[], TrainingStepOutput],
        backward_operation: Callable[[torch.Tensor], None],
        zero_grad_operation: Callable[[], None] | None
    ) -> OptimizerStepOutput:
        # Executes training_step, validates and scales its loss, optionally zeros
        # gradients, invokes precision-aware backward, and returns detached loss.
        raw_outputs: TrainingStepOutput = step_operation()
        if raw_outputs is None:
            warnings.warn(
                "training_step returned None; this batch will be skipped.",
                UserWarning,
                stacklevel=3
            )
            if strategy.is_distributed:
                raise RuntimeError("Distributed training_step cannot return None.")
            self._outputs: StepOutput = {}
            if zero_grad_operation is not None:
                zero_grad_operation()
            return None
        raw_loss: torch.Tensor | float | None
        normalized_outputs: StepOutput
        if isinstance(raw_outputs, torch.Tensor):
            raw_loss: torch.Tensor | float | None = raw_outputs
            normalized_outputs: StepOutput = {}
        elif isinstance(raw_outputs, Mapping):
            raw_loss: torch.Tensor | float | None = raw_outputs.get("loss")
            normalized_outputs: StepOutput = dict(raw_outputs)
        else:
            raise MisconfigurationError(
                "Automatic training_step must return a Tensor, mapping, or None."
            )
        if not isinstance(raw_loss, torch.Tensor):
            raise MisconfigurationError("Automatic optimization requires a tensor loss.")
        loss: torch.Tensor = raw_loss / self._accumulation_factor
        normalized_outputs["loss"]: torch.Tensor | float = loss.detach().clone()
        self._outputs: StepOutput = normalized_outputs
        if zero_grad_operation is not None:
            zero_grad_operation()
        backward_operation(loss)
        return loss.detach().clone()

    @property
    def outputs(self) -> StepOutput:
        # Defensive copy of normalized step output for batch-end hooks.
        return dict(self._outputs)
