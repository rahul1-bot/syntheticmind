from __future__ import annotations

import math
import random
import sys
import warnings
from collections import OrderedDict
from contextlib import AbstractContextManager, nullcontext
from functools import cache
from typing import TYPE_CHECKING, Any, TextIO, cast

import numpy as np
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from torch.utils.hooks import RemovableHandle

from syntheticmind.core.module import Module
from syntheticmind.utilities.model_helpers import ModuleMode
from syntheticmind.utilities.types import Batch

if TYPE_CHECKING:
    from syntheticmind.trainer.trainer import Trainer

__all__: list[str] = [
    "LayerSummary",
    "ModelSummary",
    "get_formatted_model_size",
    "get_human_readable_count",
    "parse_batch_shape",
    "summarize"
]

_parameter_count_units: tuple[str, ...] = (" ", "K", "M", "B", "T")
_unknown_size: str = "?"
_leftover_parameters_name: str = "other params"
_not_applicable: str = "n/a"

type BatchShape = str | list[object]
type SummaryData = list[tuple[str, list[str]]]


@cache
def _warn_uninitialized_parameter() -> None:
    warnings.warn(
        "The total number of parameters may be inaccurate because the model "
        "contains an UninitializedParameter. Set example_input_array to "
        "materialize lazy parameters before summarization.",
        UserWarning,
        stacklevel=4
    )


def _parameter_numel(parameter: torch.Tensor) -> int:
    from torch.nn.parameter import UninitializedParameter

    if isinstance(parameter, UninitializedParameter):
        try:
            parameter.shape
        except RuntimeError:
            _warn_uninitialized_parameter()
            return 0
    return parameter.numel()


class LayerSummary:
    def __init__(self, module: nn.Module) -> None:
        self._module: nn.Module = module
        self._hook_handle: RemovableHandle | None = None
        self._in_size: BatchShape | None = None
        self._out_size: BatchShape | None = None
        self._hook_handle = self._register_hook()

    def __del__(self) -> None:
        self.detach_hook()

    def _register_hook(self) -> RemovableHandle | None:
        if isinstance(self._module, torch.jit.ScriptModule):
            return None

        def hook(
            module: nn.Module,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            output: object
        ) -> None:
            del module
            inputs: object = (*args, *kwargs.values())
            if isinstance(inputs, tuple) and len(inputs) == 1:
                inputs = inputs[0]
            self._in_size = parse_batch_shape(inputs)
            self._out_size = parse_batch_shape(output)
            if self._hook_handle is not None:
                self._hook_handle.remove()

        return self._module.register_forward_hook(hook, with_kwargs=True)

    def detach_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()

    @property
    def in_size(self) -> BatchShape:
        return self._in_size or _unknown_size

    @property
    def out_size(self) -> BatchShape:
        return self._out_size or _unknown_size

    @property
    def layer_type(self) -> str:
        return type(self._module).__name__

    @property
    def num_parameters(self) -> int:
        return sum(_parameter_numel(parameter) for parameter in self._module.parameters())

    @property
    def training(self) -> bool:
        return self._module.training


class ModelSummary:
    def __init__(self, model: Module, max_depth: int = 1) -> None:
        if not isinstance(max_depth, int) or max_depth < -1:
            raise ValueError(f"`max_depth` can be -1, 0 or > 0, got {max_depth}.")
        self._model: Module = model
        self._max_depth: int = max_depth
        self._flop_counter: FlopCounterMode = FlopCounterMode(
            display=False,
            depth=max_depth + 1
        )
        self._layer_summary: dict[str, LayerSummary] = self.summarize()
        precision_to_bits: dict[str, int] = {
            "64": 64,
            "32": 32,
            "16": 16,
            "bf16": 16,
            "16-true": 16,
            "bf16-true": 16,
            "32-true": 32,
            "64-true": 64
        }
        trainer: Trainer | None = model._trainer
        precision: str = str(trainer.precision) if trainer is not None else "32-true"
        if trainer is not None and precision not in precision_to_bits:
            warnings.warn(
                f"Precision {precision} is not supported by the model summary size estimate. "
                "Using 32 bits per parameter.",
                UserWarning,
                stacklevel=2
            )
        precision_bits: int = precision_to_bits.get(precision, 32)
        self._precision_megabytes: float = precision_bits / 8.0 * 1e-6

    @property
    def named_modules(self) -> list[tuple[str, nn.Module]]:
        if self._max_depth == 0:
            return []
        if self._max_depth == 1:
            return cast(list[tuple[str, nn.Module]], list(self._model.named_children()))
        return cast(list[tuple[str, nn.Module]], list(self._model.named_modules())[1:])

    @property
    def layer_names(self) -> list[str]:
        return list(self._layer_summary)

    @property
    def layer_types(self) -> list[str]:
        return [layer.layer_type for layer in self._layer_summary.values()]

    @property
    def in_sizes(self) -> list[BatchShape]:
        return [layer.in_size for layer in self._layer_summary.values()]

    @property
    def out_sizes(self) -> list[BatchShape]:
        return [layer.out_size for layer in self._layer_summary.values()]

    @property
    def param_nums(self) -> list[int]:
        return [layer.num_parameters for layer in self._layer_summary.values()]

    @property
    def training_modes(self) -> list[bool]:
        return [layer.training for layer in self._layer_summary.values()]

    @property
    def total_training_modes(self) -> dict[str, int]:
        modes: list[bool] = [module.training for module in self._model.modules()][1:]
        return {"train": modes.count(True), "eval": modes.count(False)}

    @property
    def total_parameters(self) -> int:
        return sum(_parameter_numel(parameter) for parameter in self._model.parameters())

    @property
    def trainable_parameters(self) -> int:
        return sum(
            _parameter_numel(parameter)
            for parameter in self._model.parameters()
            if parameter.requires_grad
        )

    @property
    def total_layer_params(self) -> int:
        return sum(self.param_nums)

    @property
    def model_size(self) -> float:
        return self.total_parameters * self._precision_megabytes

    @property
    def total_flops(self) -> int:
        return self._flop_counter.get_total_flops()

    @property
    def flop_counts(self) -> dict[str, dict[Any, int]]:
        flop_counts: dict[str, dict[Any, int]] = self._flop_counter.get_flop_counts()
        model_name: str = type(self._model).__name__
        return {
            name: flop_counts.get(f"{model_name}.{name}", {})
            for name in self.layer_names
        }

    def summarize(self) -> dict[str, LayerSummary]:
        summary: OrderedDict[str, LayerSummary] = OrderedDict(
            (name, LayerSummary(module))
            for name, module in self.named_modules
        )
        try:
            if self._model.example_input_array is not None:
                self._forward_example_input()
        finally:
            for layer in summary.values():
                layer.detach_hook()
        if self._max_depth >= 1:
            for name in [
                name
                for name in summary
                if name.count(".") >= self._max_depth
            ]:
                del summary[name]
        return dict(summary)

    def _forward_example_input(self) -> None:
        example_input: Batch = cast(Batch, self._model.example_input_array)
        trainer: Trainer | None = self._model._trainer
        module_mode: ModuleMode = ModuleMode()
        module_mode.capture(self._model)
        python_rng_state: object = random.getstate()
        numpy_rng_state: object = np.random.get_state()
        cpu_rng_state: torch.Tensor = torch.random.get_rng_state()
        device_rng_state: torch.Tensor | None = None
        device_type: str = self._model.device.type
        if device_type == "cuda" and torch.cuda.is_available():
            device_rng_state = torch.cuda.get_rng_state(self._model.device)
        elif device_type == "mps" and torch.backends.mps.is_available():
            device_rng_state = torch.mps.get_rng_state()
        try:
            if trainer is not None:
                example_input = trainer.precision_plugin.convert_input(example_input)
            with self._model._hook_context("on_before_batch_transfer"):
                example_input = self._model.on_before_batch_transfer(example_input, 0)
            with self._model._hook_context("transfer_batch_to_device"):
                example_input = self._model.transfer_batch_to_device(
                    example_input,
                    self._model.device,
                    0
                )
            with self._model._hook_context("on_after_batch_transfer"):
                example_input = self._model.on_after_batch_transfer(example_input, 0)
            self._model.eval()
            forward_context: AbstractContextManager[None] = (
                trainer.precision_plugin.forward_context()
                if trainer is not None
                else nullcontext()
            )
            with torch.no_grad(), forward_context, self._flop_counter:
                if isinstance(example_input, (list, tuple)):
                    self._model(*example_input)
                elif isinstance(example_input, dict):
                    self._model(**example_input)
                else:
                    self._model(example_input)
        finally:
            module_mode.restore(self._model)
            random.setstate(python_rng_state)
            np.random.set_state(cast(Any, numpy_rng_state))
            torch.random.set_rng_state(cpu_rng_state)
            if device_type == "cuda" and device_rng_state is not None:
                torch.cuda.set_rng_state(device_rng_state, self._model.device)
            elif device_type == "mps" and device_rng_state is not None:
                torch.mps.set_rng_state(device_rng_state)

    def _get_summary_data(self) -> SummaryData:
        arrays: SummaryData = [
            (" ", [str(index) for index in range(len(self._layer_summary))]),
            ("Name", self.layer_names),
            ("Type", self.layer_types),
            ("Params", [get_human_readable_count(count) for count in self.param_nums]),
            ("Mode", ["train" if mode else "eval" for mode in self.training_modes]),
            (
                "FLOPs",
                [
                    get_human_readable_count(sum(layer_flops.values()))
                    for layer_flops in self.flop_counts.values()
                ]
            )
        ]
        if self._model.example_input_array is not None:
            arrays.append(("In sizes", [str(size) for size in self.in_sizes]))
            arrays.append(("Out sizes", [str(size) for size in self.out_sizes]))
        leftover_parameters: int = self.total_parameters - self.total_layer_params
        if leftover_parameters > 0:
            self._add_leftover_params_to_summary(arrays, leftover_parameters)
        return arrays

    def _add_leftover_params_to_summary(
        self,
        arrays: SummaryData,
        total_leftover_params: int
    ) -> None:
        columns: dict[str, list[str]] = dict(arrays)
        columns[" "].append(" ")
        columns["Name"].append(_leftover_parameters_name)
        columns["Type"].append(_not_applicable)
        columns["Params"].append(get_human_readable_count(total_leftover_params))
        columns["Mode"].append(_not_applicable)
        columns["FLOPs"].append(_not_applicable)
        if "In sizes" in columns:
            columns["In sizes"].append(_not_applicable)
        if "Out sizes" in columns:
            columns["Out sizes"].append(_not_applicable)

    def __str__(self) -> str:
        return _format_summary_table(
            self.total_parameters,
            self.trainable_parameters,
            self.model_size,
            self.total_training_modes,
            self.total_flops,
            *self._get_summary_data()
        )

    def __repr__(self) -> str:
        return str(self)


def parse_batch_shape(batch: object) -> BatchShape:
    shape: object | None = getattr(batch, "shape", None)
    if shape is not None:
        try:
            return list(cast(Any, shape))
        except TypeError:
            return _unknown_size
    if isinstance(batch, (list, tuple)):
        return [parse_batch_shape(element) for element in batch]
    return _unknown_size


def _format_summary_table(
    total_parameters: int,
    trainable_parameters: int,
    model_size: float,
    total_training_modes: dict[str, int],
    total_flops: int,
    *cols: tuple[str, list[str]]
) -> str:
    if not cols:
        raise ValueError("At least one summary column is required.")
    row_count: int = len(cols[0][1])
    column_widths: list[int] = [
        max(
            len(name),
            max((len(str(value)) for value in values), default=0)
        )
        for name, values in cols
    ]
    header: str = " | ".join(
        f"{name:<{width}}"
        for (name, _), width in zip(cols, column_widths, strict=True)
    )
    total_width: int = sum(column_widths) + 3 * (len(cols) + 1)
    lines: list[str] = [header, "─" * total_width]
    for row_index in range(row_count):
        lines.append(
            " | ".join(
                f"{values[row_index]:<{width}}"
                for (_, values), width in zip(cols, column_widths, strict=True)
            )
        )
    lines.extend(
        [
            "─" * total_width,
            f"{get_human_readable_count(trainable_parameters):<10}Trainable params",
            f"{get_human_readable_count(total_parameters - trainable_parameters):<10}Non-trainable params",
            f"{get_human_readable_count(total_parameters):<10}Total params",
            f"{get_formatted_model_size(model_size):<10}Total estimated model params size (MB)",
            f"{total_training_modes['train']:<10}Modules in train mode",
            f"{total_training_modes['eval']:<10}Modules in eval mode",
            f"{get_human_readable_count(total_flops):<10}Total FLOPs"
        ]
    )
    return "\n".join(lines)


def get_formatted_model_size(total_model_size: float) -> str:
    return f"{total_model_size:,.3f}"


def get_human_readable_count(number: int) -> str:
    if number < 0:
        raise ValueError(f"number must be non-negative, got {number}")
    labels: tuple[str, ...] = _parameter_count_units
    digit_count: int = int(math.floor(math.log10(number)) + 1 if number > 0 else 1)
    group_count: int = min(int(math.ceil(digit_count / 3)), len(labels))
    shift: int = -3 * (group_count - 1)
    shifted_number: float = number * 10 ** shift
    index: int = group_count - 1
    if index < 1 or shifted_number >= 100:
        return f"{int(shifted_number):,d} {labels[index]}"
    return f"{shifted_number:,.1f} {labels[index]}"


def summarize(module: Module, max_depth: int = 1) -> ModelSummary:
    return ModelSummary(module, max_depth=max_depth)


def _write_summary(summary: str, stream: TextIO | None = None) -> None:
    destination: TextIO = sys.stderr if stream is None else stream
    try:
        destination.write(f"{summary}\n")
        destination.flush()
    except (OSError, ValueError):
        return
