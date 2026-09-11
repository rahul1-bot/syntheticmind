# This module:
# 1. Defines the reversible module instrumentation used by PyTorchProfiler to
#    annotate eager forward execution with qualified record_function ranges
# 2. Owns hook handles and nested range state for exact success and failure
#    cleanup
#
# Design decisions:
# - Every eager module receives paired pre-forward and always-call post-forward
#   hooks, so ranges close even when client forward raises
# - Active recordings are stored as stacks by module identity because shared or
#   recursive modules may enter more than once before their matching exit
# - Labels include the Python-qualified type and model path, distinguishing equal
#   layer classes at different positions in operator traces
# - ScriptModule instrumentation is skipped because eager hook registration does
#   not provide a reliable equivalent contract for scripted execution
# - Context exit closes every surviving range before removing every handle,
#   making partial forward failure and profiler teardown reversible
#
# Author: Rahul Sawhney

from types import TracebackType
from typing import Self

import torch
from torch import nn
from torch.profiler import record_function
from torch.utils.hooks import RemovableHandle

from syntheticmind.primitives.types import Batch, ModelOutput

__all__: list[str] = []


class _ModuleRecordFunctions:
    # Private instrumentation transaction created and consumed by
    # PyTorchProfiler; client code enables it through record_module_names.
    def __init__(self, model: nn.Module) -> None:
        # Captures stable module identities and paths before hooks are installed.
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
        # Opens one module-qualified trace range and pushes it for the matching
        # always-call forward hook.
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
        # Closes the most recent range for this module after successful or failed
        # eager forward execution.
        del inputs, output
        recordings: list[record_function] | None = self._recordings.get(
            id(module)
        )
        if not recordings:
            return
        recording: record_function = recordings.pop()
        recording.__exit__(None, None, None)

    def __enter__(self) -> Self:
        # Installs paired hooks across the current eager model hierarchy.
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
        # Closes leaked ranges with the causal exception context and removes all
        # hooks exactly once.
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
