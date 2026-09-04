from __future__ import annotations

import sys
from typing import TextIO, override

import syntheticmind

from syntheticmind.callbacks.callback import Callback
from syntheticmind.callbacks.model_summary.report import (
    SummaryData,
    _ModelSummaryReport
)
from syntheticmind.core.module import Module

__all__: list[str] = ["ModelSummary"]


class ModelSummary(Callback):
    def __init__(
        self,
        max_depth: int = 1,
        **summarize_kwargs: str
    ) -> None:
        if type(max_depth) is not int or max_depth < -1:
            raise ValueError(f"`max_depth` can be -1, 0 or greater, got {max_depth}.")
        self._max_depth: int = max_depth
        self._summarize_kwargs: dict[str, str] = summarize_kwargs

    @override
    def on_fit_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        if self._max_depth == 0:
            return
        report: _ModelSummaryReport = self._summary(trainer, module)
        if not trainer.is_global_zero:
            return
        self.summarize(
            report.summary_data(),
            report.total_parameters,
            report.trainable_parameters,
            report.model_size,
            report.total_training_modes,
            report.total_flops,
            **self._summarize_kwargs
        )

    def _summary(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> _ModelSummaryReport:
        del trainer
        return _ModelSummaryReport(module, max_depth=self._max_depth)

    @staticmethod
    def summarize(
        summary_data: SummaryData,
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        total_training_modes: dict[str, int],
        total_flops: int,
        **summarize_kwargs: str
    ) -> None:
        del summarize_kwargs
        table: str = ModelSummary._format_summary_table(
            total_parameters,
            trainable_parameters,
            model_size,
            total_training_modes,
            total_flops,
            summary_data
        )
        ModelSummary._write_summary(table)

    @staticmethod
    def _format_summary_table(
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        total_training_modes: dict[str, int],
        total_flops: int,
        columns: SummaryData
    ) -> str:
        if not columns:
            raise ValueError("At least one summary column is required.")
        row_count: int = len(columns[0][1])
        column_widths: list[int] = [
            max(
                len(name),
                max((len(str(value)) for value in values), default=0)
            )
            for name, values in columns
        ]
        header: str = " | ".join(
            f"{name:<{width}}"
            for (name, _), width in zip(columns, column_widths, strict=True)
        )
        total_width: int = sum(column_widths) + 3 * (len(columns) + 1)
        lines: list[str] = [header, "─" * total_width]
        row_index: int
        for row_index in range(row_count):
            lines.append(
                " | ".join(
                    f"{values[row_index]:<{width}}"
                    for (_, values), width in zip(
                        columns,
                        column_widths,
                        strict=True
                    )
                )
            )
        lines.extend(
            [
                "─" * total_width,
                f"{_ModelSummaryReport.human_readable_count(trainable_parameters):<10}Trainable params",
                f"{_ModelSummaryReport.human_readable_count(total_parameters - trainable_parameters):<10}Non-trainable params",
                f"{_ModelSummaryReport.human_readable_count(total_parameters):<10}Total params",
                f"{_ModelSummaryReport.formatted_model_size(model_size):<10}Total estimated model params size (MB)",
                f"{total_training_modes['train']:<10}Modules in train mode",
                f"{total_training_modes['eval']:<10}Modules in eval mode",
                f"{_ModelSummaryReport.human_readable_count(total_flops):<10}Total FLOPs"
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _write_summary(summary: str, stream: TextIO | None = None) -> None:
        destination: TextIO = sys.stderr if stream is None else stream
        try:
            destination.write(f"{summary}\n")
            destination.flush()
        except (OSError, ValueError):
            return
