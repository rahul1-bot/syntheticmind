from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.utilities.model_summary import (
    ModelSummary as Summary,
    SummaryData,
    _format_summary_table,
    _write_summary,
    summarize
)

if TYPE_CHECKING:
    from syntheticmind.core.trainer import Trainer

__all__: list[str] = ["ModelSummary"]


class ModelSummary(Callback):
    def __init__(self, max_depth: int = 1, **summarize_kwargs: Any) -> None:
        if not isinstance(max_depth, int) or max_depth < -1:
            raise ValueError(f"`max_depth` can be -1, 0 or > 0, got {max_depth}.")
        self._max_depth: int = max_depth
        self._summarize_kwargs: dict[str, Any] = summarize_kwargs

    @override
    def on_fit_start(self, trainer: Trainer, module: Module) -> None:
        if self._max_depth == 0:
            return
        model_summary: Summary = self._summary(trainer, module)
        if trainer.is_global_zero:
            self.summarize(
                model_summary._get_summary_data(),
                model_summary.total_parameters,
                model_summary.trainable_parameters,
                model_summary.model_size,
                model_summary.total_training_modes,
                total_flops=model_summary.total_flops,
                **self._summarize_kwargs
            )

    def _summary(self, trainer: Trainer, module: Module) -> Summary:
        del trainer
        return summarize(module, max_depth=self._max_depth)

    @staticmethod
    def summarize(
        summary_data: SummaryData,
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        total_training_modes: dict[str, int],
        total_flops: int,
        **summarize_kwargs: Any
    ) -> None:
        del summarize_kwargs
        table: str = _format_summary_table(
            total_parameters,
            trainable_parameters,
            model_size,
            total_training_modes,
            total_flops,
            *summary_data
        )
        _write_summary(table)
