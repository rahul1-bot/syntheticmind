from collections.abc import Callable, Mapping

import torch

from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.types import StepOutput, TrainingStepOutput

__all__: list[str] = []


class _ManualOptimization:
    @torch.enable_grad()
    def run(
        self,
        step_operation: Callable[[], TrainingStepOutput]
    ) -> StepOutput:
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
