# This module:
# 1. Normalizes reporting output from a manual-optimization training_step while
#    leaving backward, zeroing, optimizer order, and stepping with client code
#
# Design decisions:
# - A tensor becomes a detached ``"loss"`` mapping for callback consistency
# - Mapping outputs retain client keys but detach a tensor loss so returned batch
#   state cannot preserve an autograd graph
# - None represents a valid reporting-empty manual step because optimization may
#   have completed entirely through Module methods
#
# Author: Rahul Sawhney

from collections.abc import Callable, Mapping

import torch

from syntheticmind.primitives.exceptions import MisconfigurationError
from syntheticmind.primitives.types import StepOutput, TrainingStepOutput

__all__: list[str] = []


class _ManualOptimization:
    # Stateless normalization of one client-owned manual training step.
    @torch.enable_grad()
    def run(
        self,
        step_operation: Callable[[], TrainingStepOutput]
    ) -> StepOutput:
        # Executes the client step with gradients enabled and converts its return
        # to the common detached batch-output mapping.
        raw_outputs: TrainingStepOutput = step_operation()
        if raw_outputs is None:
            return {}
        if isinstance(raw_outputs, torch.Tensor):
            return {"loss": raw_outputs.detach()}
        if isinstance(raw_outputs, Mapping):
            outputs: StepOutput = dict(raw_outputs)
            loss: torch.Tensor | float | None = outputs.get("loss")
            if isinstance(loss, torch.Tensor):
                outputs["loss"]: torch.Tensor | float = loss.detach()
            return outputs
        raise MisconfigurationError(
            "Manual training_step must return a Tensor, mapping, or None."
        )
