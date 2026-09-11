# This module:
# 1. Defines RichModelSummary, the optional Rich terminal renderer for the
#    canonical ModelSummary report
# 2. Converts normalized report columns and aggregate values into a Rich table
#    and compact summary grid
#
# Design decisions:
# - Rich remains an optional import boundary; requesting this renderer without
#   the dependency raises one actionable installation error at import time
# - Measurement stays entirely in _ModelSummaryReport. This adapter changes only
#   presentation, so plain and Rich clients receive identical model facts
# - Optional shape and per-device columns are discovered from report column
#   names rather than coupled to one execution strategy
# - Client summarize_kwargs may choose header_style while the structural column
#   order remains canonical
#
# Author: Rahul Sawhney

from typing import override

try:
    from rich import get_console
    from rich.console import Console
    from rich.table import Table
except ModuleNotFoundError as exception:
    raise ModuleNotFoundError(
        "RichModelSummary requires the optional 'rich' package. Install it with "
        "`python -m pip install rich --break-system-packages`."
    ) from exception

from syntheticmind.callbacks.model_summary.callback import ModelSummary
from syntheticmind.callbacks.model_summary.report import (
    SummaryData,
    _ModelSummaryReport
)

__all__: list[str] = ["RichModelSummary"]


class RichModelSummary(ModelSummary):
    # Rich presentation adapter selected explicitly by clients that installed
    # the optional dependency.
    @override
    def __init__(
        self,
        max_depth: int = 1,
        **summarize_kwargs: str
    ) -> None:
        # Preserves ModelSummary depth and presentation-option validation.
        super().__init__(max_depth=max_depth, **summarize_kwargs)

    @override
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
        # Renders canonical layer rows and aggregate model accounting without
        # recomputing or mutating the underlying report.
        column_names: set[str] = {name for name, _ in summary_data}
        header_style: str = summarize_kwargs.get("header_style", "bold magenta")
        table: Table = Table(header_style=header_style)
        table.add_column(" ", style="dim")
        table.add_column("Name", justify="left", no_wrap=True)
        table.add_column("Type")
        table.add_column("Params", justify="right")
        if "Params per Device" in column_names:
            table.add_column("Params per Device", justify="right")
        table.add_column("Mode")
        table.add_column("FLOPs", justify="right")
        column_name: str
        for column_name in ("In sizes", "Out sizes"):
            if column_name in column_names:
                table.add_column(column_name, justify="right", style="white")
        rows: list[tuple[str, ...]] = list(
            zip(*(values for _, values in summary_data), strict=True)
        )
        row: tuple[str, ...]
        for row in rows:
            table.add_row(*row)
        console: Console = get_console()
        console.print(table)
        parameter_values: list[str] = [
            f"{_ModelSummaryReport.human_readable_count(count):<10}"
            for count in (
                trainable_parameters,
                total_parameters - trainable_parameters,
                total_parameters
            )
        ]
        parameter_values.append(
            f"{_ModelSummaryReport.formatted_model_size(model_size):<10}"
        )
        grid: Table = Table.grid(expand=True)
        grid.add_column()
        grid.add_column()
        grid.add_row("[bold]Trainable params[/]", parameter_values[0])
        grid.add_row("[bold]Non-trainable params[/]", parameter_values[1])
        grid.add_row("[bold]Total params[/]", parameter_values[2])
        grid.add_row(
            "[bold]Estimated model params size (MB)[/]",
            parameter_values[3]
        )
        grid.add_row(
            "[bold]Modules in train mode[/]",
            str(total_training_modes["train"])
        )
        grid.add_row(
            "[bold]Modules in eval mode[/]",
            str(total_training_modes["eval"])
        )
        grid.add_row(
            "[bold]Total FLOPs[/]",
            _ModelSummaryReport.human_readable_count(total_flops)
        )
        console.print(grid)
