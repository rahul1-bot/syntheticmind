import math
import warnings
from collections.abc import Hashable, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import ClassVar, cast

import syntheticmind
import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter
from torch.utils.flop_counter import FlopCounterMode
from torch.utils.hooks import RemovableHandle

from syntheticmind.core.module import Module
from syntheticmind.randomness import _Randomness, _RandomStateSnapshot
from syntheticmind.types import Batch, ModelOutput

__all__: list[str] = []

type SummaryShape = str | list[int | SummaryShape]
type SummaryData = list[tuple[str, list[str]]]
type _FlopCounts = dict[str, dict[Hashable, int]]


class _LayerSummary:
    _unknown_size: ClassVar[str] = "?"

    def __init__(self, module: nn.Module) -> None:
        self._module: nn.Module = module
        self._hook_handle: RemovableHandle | None = None
        self._input_size: SummaryShape | None = None
        self._output_size: SummaryShape | None = None

    def attach_hook(self) -> None:
        if isinstance(self._module, torch.jit.ScriptModule):
            return
        self._hook_handle: RemovableHandle | None = (
            self._module.register_forward_hook(
                self._capture_shapes,
                with_kwargs=True
            )
        )

    def detach_hook(self) -> None:
        if self._hook_handle is None:
            return
        self._hook_handle.remove()
        self._hook_handle: RemovableHandle | None = None

    def _capture_shapes(
        self,
        module: nn.Module,
        args: tuple[Batch, ...],
        kwargs: dict[str, Batch],
        output: ModelOutput
    ) -> None:
        del module
        input_values: tuple[Batch, ...] = (*args, *kwargs.values())
        inputs: Batch = (
            input_values[0]
            if len(input_values) == 1
            else input_values
        )
        self._input_size: SummaryShape | None = self._batch_shape(inputs)
        self._output_size: SummaryShape | None = self._batch_shape(output)
        self.detach_hook()

    def _batch_shape(self, value: Batch | ModelOutput) -> SummaryShape:
        shape: Sequence[int] | None = cast(
            Sequence[int] | None,
            getattr(value, "shape", None)
        )
        if shape is not None:
            try:
                return [int(dimension) for dimension in shape]
            except (TypeError, ValueError):
                return self._unknown_size
        if isinstance(value, list | tuple):
            return [self._batch_shape(element) for element in value]
        return self._unknown_size

    def _parameter_numel(self, parameter: torch.Tensor) -> int:
        if isinstance(parameter, UninitializedParameter):
            try:
                parameter.shape
            except RuntimeError:
                warnings.warn(
                    "The parameter total is incomplete because the model contains an "
                    "uninitialized parameter. Supply example_input_array to materialize "
                    "lazy parameters before summarization.",
                    UserWarning,
                    stacklevel=4
                )
                return 0
        return parameter.numel()

    @property
    def input_size(self) -> SummaryShape:
        return (
            self._unknown_size
            if self._input_size is None
            else self._input_size
        )

    @property
    def output_size(self) -> SummaryShape:
        return (
            self._unknown_size
            if self._output_size is None
            else self._output_size
        )

    @property
    def layer_type(self) -> str:
        return type(self._module).__name__

    @property
    def num_parameters(self) -> int:
        return sum(
            self._parameter_numel(parameter)
            for parameter in self._module.parameters()
        )

    @property
    def trainable_parameters(self) -> int:
        return sum(
            self._parameter_numel(parameter)
            for parameter in self._module.parameters()
            if parameter.requires_grad
        )

    @property
    def parameter_bytes(self) -> int:
        total_bytes: int = 0
        parameter: torch.nn.Parameter
        for parameter in self._module.parameters():
            parameter_count: int = self._parameter_numel(parameter)
            if parameter_count > 0:
                total_bytes: int = (
                    total_bytes + parameter_count * parameter.element_size()
                )
        return total_bytes

    @property
    def training(self) -> bool:
        return self._module.training


class _ModelSummaryReport:
    _parameter_count_units: ClassVar[tuple[str, ...]] = (" ", "K", "M", "B", "T")
    _leftover_parameters_name: ClassVar[str] = "other params"
    _not_applicable: ClassVar[str] = "n/a"

    def __init__(self, model: Module, max_depth: int = 1) -> None:
        if type(max_depth) is not int or max_depth < -1:
            raise ValueError(f"`max_depth` can be -1, 0 or greater, got {max_depth}.")
        self._model: Module = model
        self._max_depth: int = max_depth
        self._flop_counter: FlopCounterMode = FlopCounterMode(
            display=False,
            depth=max_depth + 1
        )
        self._root_summary: _LayerSummary = _LayerSummary(model)
        self._layers: dict[str, _LayerSummary] = self._summarize_layers()

    @property
    def layer_names(self) -> list[str]:
        return list(self._layers)

    @property
    def layer_types(self) -> list[str]:
        return [layer.layer_type for layer in self._layers.values()]

    @property
    def input_sizes(self) -> list[SummaryShape]:
        return [layer.input_size for layer in self._layers.values()]

    @property
    def output_sizes(self) -> list[SummaryShape]:
        return [layer.output_size for layer in self._layers.values()]

    @property
    def parameter_counts(self) -> list[int]:
        return [layer.num_parameters for layer in self._layers.values()]

    @property
    def training_modes(self) -> list[bool]:
        return [layer.training for layer in self._layers.values()]

    @property
    def total_training_modes(self) -> dict[str, int]:
        modes: list[bool] = [module.training for module in self._model.modules()][1:]
        return {"train": modes.count(True), "eval": modes.count(False)}

    @property
    def total_parameters(self) -> int:
        return self._root_summary.num_parameters

    @property
    def trainable_parameters(self) -> int:
        return self._root_summary.trainable_parameters

    @property
    def model_size(self) -> float:
        return self._root_summary.parameter_bytes * 1e-6

    @property
    def total_flops(self) -> int:
        return self._flop_counter.get_total_flops()

    def summary_data(self) -> SummaryData:
        arrays: SummaryData = [
            (" ", [str(index) for index in range(len(self._layers))]),
            ("Name", self.layer_names),
            ("Type", self.layer_types),
            (
                "Params",
                [self.human_readable_count(count) for count in self.parameter_counts]
            ),
            ("Mode", ["train" if mode else "eval" for mode in self.training_modes]),
            (
                "FLOPs",
                [self.human_readable_count(count) for count in self._layer_flops()]
            )
        ]
        if self._model.example_input_array is not None:
            arrays.append(("In sizes", [str(size) for size in self.input_sizes]))
            arrays.append(("Out sizes", [str(size) for size in self.output_sizes]))
        leftover_parameters: int = self.total_parameters - sum(self.parameter_counts)
        if leftover_parameters > 0:
            self._add_leftover_parameters(arrays, leftover_parameters)
        return arrays

    @classmethod
    def human_readable_count(cls, number: int) -> str:
        if number < 0:
            raise ValueError(f"number must be non-negative, got {number}")
        digit_count: int = int(
            math.floor(math.log10(number)) + 1
            if number > 0
            else 1
        )
        group_count: int = min(
            int(math.ceil(digit_count / 3)),
            len(cls._parameter_count_units)
        )
        shift: int = -3 * (group_count - 1)
        shifted_number: float = number * 10 ** shift
        unit_index: int = group_count - 1
        if unit_index < 1 or shifted_number >= 100:
            return f"{int(shifted_number):,d} {cls._parameter_count_units[unit_index]}"
        return f"{shifted_number:,.1f} {cls._parameter_count_units[unit_index]}"

    @classmethod
    def formatted_model_size(cls, model_size: float) -> str:
        del cls
        return f"{model_size:,.3f}"

    def _named_modules(self) -> list[tuple[str, nn.Module]]:
        if self._max_depth == 0:
            return []
        if self._max_depth == 1:
            return list(self._model.named_children())
        return list(self._model.named_modules())[1:]

    def _summarize_layers(self) -> dict[str, _LayerSummary]:
        summaries: dict[str, _LayerSummary] = {}
        try:
            name: str
            module: nn.Module
            for name, module in self._named_modules():
                layer: _LayerSummary = _LayerSummary(module)
                summaries[name] = layer
                layer.attach_hook()
            if self._model.example_input_array is not None:
                self._forward_example_input()
        finally:
            for layer in summaries.values():
                layer.detach_hook()
        if self._max_depth >= 1:
            for name in [
                candidate
                for candidate in summaries
                if candidate.count(".") >= self._max_depth
            ]:
                del summaries[name]
        return summaries

    def _forward_example_input(self) -> None:
        example_input: Batch = cast(Batch, self._model.example_input_array)
        trainer: syntheticmind.Trainer | None = self._model._trainer
        training_modes: dict[str, bool] = self._model._capture_training_modes()
        device_type: str = self._model.device.type
        randomness: _Randomness = (
            trainer._randomness
            if trainer is not None
            else _Randomness(
                [device_type if device_type in {"cpu", "cuda", "mps"} else "cpu"]
            )
        )
        random_state: _RandomStateSnapshot = randomness.capture()
        try:
            if trainer is not None:
                example_input: Batch = trainer.precision_plugin.convert_input(
                    example_input
                )
                example_input: Batch = trainer._transfer_batch(
                    self._model,
                    example_input,
                    self._model.device,
                    0
                )
            else:
                with self._model._hook_context("on_before_batch_transfer"):
                    example_input: Batch = self._model.on_before_batch_transfer(
                        example_input,
                        0
                    )
                with self._model._hook_context("transfer_batch_to_device"):
                    example_input: Batch = self._model.transfer_batch_to_device(
                        example_input,
                        self._model.device,
                        0
                    )
                with self._model._hook_context("on_after_batch_transfer"):
                    example_input: Batch = self._model.on_after_batch_transfer(
                        example_input,
                        0
                    )
            self._model.eval()
            forward_context: AbstractContextManager[None] = (
                trainer.precision_plugin.forward_context()
                if trainer is not None
                else nullcontext()
            )
            with torch.no_grad(), forward_context, self._flop_counter:
                if isinstance(example_input, list | tuple):
                    self._model(*example_input)
                elif isinstance(example_input, dict):
                    self._model(**example_input)
                else:
                    self._model(example_input)
        finally:
            self._model._restore_training_modes(training_modes)
            randomness.restore(random_state)

    def _layer_flops(self) -> list[int]:
        flop_counts: _FlopCounts = cast(
            _FlopCounts,
            self._flop_counter.get_flop_counts()
        )
        model_name: str = type(self._model).__name__
        return [
            sum(flop_counts.get(f"{model_name}.{name}", {}).values())
            for name in self.layer_names
        ]

    def _add_leftover_parameters(
        self,
        arrays: SummaryData,
        leftover_parameters: int
    ) -> None:
        columns: dict[str, list[str]] = dict(arrays)
        columns[" "].append(" ")
        columns["Name"].append(self._leftover_parameters_name)
        columns["Type"].append(self._not_applicable)
        columns["Params"].append(self.human_readable_count(leftover_parameters))
        columns["Mode"].append(self._not_applicable)
        columns["FLOPs"].append(self._not_applicable)
        if "In sizes" in columns:
            columns["In sizes"].append(self._not_applicable)
        if "Out sizes" in columns:
            columns["Out sizes"].append(self._not_applicable)
