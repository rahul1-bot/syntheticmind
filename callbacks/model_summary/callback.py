# This module:
# 1. Defines ModelSummary, the fit-start callback that constructs and presents a
#    bounded structural report for the attached client model
# 2. Provides the plain-text summary renderer and the summarize extension point
#    shared by alternate presentation adapters
#
# Design decisions:
# - Every rank builds the report because example forwarding may exercise
#   strategy-aware transfer and precision paths, but only global zero presents
#   terminal output
# - max_depth=0 disables report construction entirely, max_depth=1 reports direct
#   children, and -1 permits the complete module hierarchy
# - Report construction, measurement, and state restoration live in
#   _ModelSummaryReport; this callback owns lifecycle placement and presentation
# - summarize receives normalized columns and aggregate values, giving clients a
#   renderer extension point without exposing report mutation internals
# - Plain output targets stderr and tolerates closed or unavailable streams so a
#   presentation failure cannot terminate training
#
# Author: Rahul Sawhney

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
    # Default plain-terminal model inspection callback. Client renderers can
    # subclass it and override summarize while reusing the canonical report.
    def __init__(
        self,
        max_depth: int = 1,
        **summarize_kwargs: str
    ) -> None:
        # Args:
        #     max_depth: Maximum reported module nesting depth; ``-1`` means all
        #         descendants and ``0`` disables the callback. Default: ``1``.
        #     **summarize_kwargs: String presentation options forwarded to
        #         summarize implementations.
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
        # Builds the report at fit start and presents it only on global zero.
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
        # Creates the canonical measurement report; subclasses may replace this
        # only when they preserve its state-restoration guarantees.
        del trainer
        return _ModelSummaryReport(module, max_depth=self._max_depth)

    def summarize(
        self,
        summary_data: SummaryData,
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        total_training_modes: dict[str, int],
        total_flops: int,
        **summarize_kwargs: str
    ) -> None:
        # Formats and writes the plain summary. Alternate renderers override this
        # method while accepting the same normalized report payload.
        del summarize_kwargs
        table: str = self._format_summary_table(
            total_parameters,
            trainable_parameters,
            model_size,
            total_training_modes,
            total_flops,
            summary_data
        )
        self._write_summary(table)

    def _format_summary_table(
        self,
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        total_training_modes: dict[str, int],
        total_flops: int,
        columns: SummaryData
    ) -> str:
        # Produces an aligned table plus aggregate parameter, memory, mode, and
        # FLOP totals from canonical report columns.
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
        trainable_count: str = _ModelSummaryReport.human_readable_count(
            trainable_parameters
        )
        non_trainable_count: str = _ModelSummaryReport.human_readable_count(
            total_parameters - trainable_parameters
        )
        total_count: str = _ModelSummaryReport.human_readable_count(
            total_parameters
        )
        formatted_size: str = _ModelSummaryReport.formatted_model_size(model_size)
        flop_count: str = _ModelSummaryReport.human_readable_count(total_flops)
        lines.extend(
            [
                "─" * total_width,
                f"{trainable_count:<10}Trainable params",
                f"{non_trainable_count:<10}Non-trainable params",
                f"{total_count:<10}Total params",
                f"{formatted_size:<10}Total estimated model params size (MB)",
                f"{total_training_modes['train']:<10}Modules in train mode",
                f"{total_training_modes['eval']:<10}Modules in eval mode",
                f"{flop_count:<10}Total FLOPs"
            ]
        )
        return "\n".join(lines)

    def _write_summary(
        self,
        summary: str,
        stream: TextIO | None = None
    ) -> None:
        # Writes one complete summary atomically enough for terminal use and
        # suppresses only stream-level failures.
        destination: TextIO = sys.stderr if stream is None else stream
        try:
            destination.write(f"{summary}\n")
            destination.flush()
        except (OSError, ValueError):
            return
