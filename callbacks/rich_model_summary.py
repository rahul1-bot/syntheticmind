from __future__ import annotations

import importlib.util
from typing import Any, override

from syntheticmind.callbacks.model_summary import ModelSummary
from syntheticmind.utilities.model_summary import (
    SummaryData,
    get_formatted_model_size,
    get_human_readable_count
)

__all__: list[str] = ["RichModelSummary"]


class RichModelSummary(ModelSummary):
    def __init__(self, max_depth: int = 1, **summarize_kwargs: Any) -> None:
        if importlib.util.find_spec("rich") is None:
            raise ModuleNotFoundError(
                "RichModelSummary requires the optional 'rich' package. "
                "Install it with `python -m pip install rich --break-system-packages`."
            )
        super().__init__(max_depth=max_depth, **summarize_kwargs)

    @staticmethod
    @override
    def summarize(
        summary_data: SummaryData,
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        total_training_modes: dict[str, int],
        total_flops: int,
        **summarize_kwargs: Any
    ) -> None:
        from rich import get_console
        from rich.table import Table

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
        for column_name in ("In sizes", "Out sizes"):
            if column_name in column_names:
                table.add_column(column_name, justify="right", style="white")
        rows: list[tuple[str, ...]] = list(
            zip(*(values for _, values in summary_data), strict=True)
        )
        for row in rows:
            table.add_row(*row)
        console = get_console()
        console.print(table)
        values: list[str] = [
            f"{get_human_readable_count(count):<10}"
            for count in (
                trainable_parameters,
                total_parameters - trainable_parameters,
                total_parameters
            )
        ]
        values.append(f"{get_formatted_model_size(model_size):<10}")
        grid: Table = Table.grid(expand=True)
        grid.add_column()
        grid.add_column()
        grid.add_row("[bold]Trainable params[/]", values[0])
        grid.add_row("[bold]Non-trainable params[/]", values[1])
        grid.add_row("[bold]Total params[/]", values[2])
        grid.add_row("[bold]Estimated model params size (MB)[/]", values[3])
        grid.add_row("[bold]Modules in train mode[/]", str(total_training_modes["train"]))
        grid.add_row("[bold]Modules in eval mode[/]", str(total_training_modes["eval"]))
        grid.add_row("[bold]Total FLOPs[/]", get_human_readable_count(total_flops))
        console.print(grid)
