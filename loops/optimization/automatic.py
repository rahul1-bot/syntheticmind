import warnings
from collections.abc import Callable, Mapping

import torch

from syntheticmind.strategies.strategy import Strategy
from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.types import (
    OptimizerStepOutput,
    StepOutput,
    TrainingStepOutput
)

__all__: list[str] = []


class _AutomaticOptimization:
    def __init__(self, accumulation_factor: int) -> None:
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
        return dict(self._outputs)
