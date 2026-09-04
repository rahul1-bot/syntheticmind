from types import TracebackType
from typing import Self

import torch
from torch import nn
from torch.profiler import record_function
from torch.utils.hooks import RemovableHandle

from syntheticmind.primitives.types import Batch, ModelOutput

__all__: list[str] = []


class _ModuleRecordFunctions:
    def __init__(self, model: nn.Module) -> None:
        self._model: nn.Module = model
        self._handles: list[RemovableHandle] = []
        self._recordings: dict[int, list[record_function]] = {}
        self._module_names: dict[int, str] = {
            id(module): name
            for name, module in model.named_modules()
        }

    def _start_forward(
        self,
        module: nn.Module,
        inputs: tuple[Batch, ...]
    ) -> None:
        del inputs
        name: str = self._module_names.get(id(module), "")
        qualified_name: str = (
            f"{type(module).__module__}.{type(module).__qualname__}"
        )
        recording: record_function = record_function(
            f"[syntheticmind][module]{qualified_name}: {name or '<root>'}"
        )
        recording.__enter__()
        self._recordings.setdefault(id(module), []).append(recording)

    def _stop_forward(
        self,
        module: nn.Module,
        inputs: tuple[Batch, ...],
        output: ModelOutput
    ) -> None:
        del inputs, output
        recordings: list[record_function] | None = self._recordings.get(
            id(module)
        )
        if not recordings:
            return
        recording: record_function = recordings.pop()
        recording.__exit__(None, None, None)

    def __enter__(self) -> Self:
        module: nn.Module
        for module in self._model.modules():
            if isinstance(module, torch.jit.ScriptModule):
                continue
            self._handles.append(
                module.register_forward_pre_hook(self._start_forward)
            )
            self._handles.append(
                module.register_forward_hook(
                    self._stop_forward,
                    always_call=True
                )
            )
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None
    ) -> None:
        recordings: list[record_function]
        for recordings in self._recordings.values():
            while recordings:
                recording: record_function = recordings.pop()
                recording.__exit__(exception_type, exception, traceback)
        self._recordings.clear()
        handle: RemovableHandle
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
