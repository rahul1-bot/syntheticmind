from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import torch

from syntheticmind.core.datamodule import DataModule
from syntheticmind.core.module import Module
from syntheticmind.core.trainer import Trainer
from syntheticmind.utilities.filesystem import open_binary
from syntheticmind.utilities.model_helpers import ModuleMode, is_overridden
from syntheticmind.utilities.types import Batch

__all__: list[str] = ["ModelTorchScriptExporter"]


class ModelExportInputTransfer:
    def __init__(self, model: Module) -> None:
        self._model: Module = model

    def prepare(self, value: object) -> object:
        batch: Batch = cast(Batch, value)
        batch: Batch = self._on_before_batch_transfer(batch)
        batch: Batch = self._transfer_batch_to_device(batch)
        batch: Batch = self._on_after_batch_transfer(batch)
        return batch

    def _on_before_batch_transfer(self, batch: Batch) -> Batch:
        trainer: Trainer | None = self._model._trainer
        datamodule: DataModule | None = trainer.datamodule if trainer is not None else None
        if datamodule is not None and is_overridden(
            "on_before_batch_transfer",
            datamodule,
            DataModule
        ):
            return trainer._call_datamodule_hook(
                datamodule,
                "on_before_batch_transfer",
                datamodule.on_before_batch_transfer,
                batch,
                0
            )
        if trainer is not None:
            return trainer._call_module_hook(
                self._model,
                "on_before_batch_transfer",
                self._model.on_before_batch_transfer,
                batch,
                0
            )
        with self._model._hook_context("on_before_batch_transfer"):
            return self._model.on_before_batch_transfer(batch, 0)

    def _transfer_batch_to_device(self, batch: Batch) -> Batch:
        trainer: Trainer | None = self._model._trainer
        datamodule: DataModule | None = trainer.datamodule if trainer is not None else None
        device: torch.device = self._model.device
        if datamodule is not None and is_overridden(
            "transfer_batch_to_device",
            datamodule,
            DataModule
        ):
            return trainer._call_datamodule_hook(
                datamodule,
                "transfer_batch_to_device",
                datamodule.transfer_batch_to_device,
                batch,
                device,
                0
            )
        if trainer is not None:
            return trainer._call_module_hook(
                self._model,
                "transfer_batch_to_device",
                self._model.transfer_batch_to_device,
                batch,
                device,
                0
            )
        with self._model._hook_context("transfer_batch_to_device"):
            return self._model.transfer_batch_to_device(batch, device, 0)

    def _on_after_batch_transfer(self, batch: Batch) -> Batch:
        trainer: Trainer | None = self._model._trainer
        datamodule: DataModule | None = trainer.datamodule if trainer is not None else None
        if datamodule is not None and is_overridden(
            "on_after_batch_transfer",
            datamodule,
            DataModule
        ):
            return trainer._call_datamodule_hook(
                datamodule,
                "on_after_batch_transfer",
                datamodule.on_after_batch_transfer,
                batch,
                0
            )
        if trainer is not None:
            return trainer._call_module_hook(
                self._model,
                "on_after_batch_transfer",
                self._model.on_after_batch_transfer,
                batch,
                0
            )
        with self._model._hook_context("on_after_batch_transfer"):
            return self._model.on_after_batch_transfer(batch, 0)


class ModelTorchScriptExporter:
    def __init__(self, model: Module) -> None:
        self._model: Module = model
        self._input_transfer: ModelExportInputTransfer = ModelExportInputTransfer(model)

    def export(
        self,
        file_path: str | Path | None,
        method: str | None,
        example_inputs: object | None,
        options: dict[str, object]
    ) -> torch.jit.ScriptModule:
        module_mode: ModuleMode = ModuleMode()
        module_mode.capture(self._model)
        export_options: dict[str, object] = dict(options)
        try:
            self._model.eval()
            if method == "script":
                exported_module: torch.jit.ScriptModule = self._script(export_options)
            elif method == "trace":
                exported_module: torch.jit.ScriptModule = self._trace(
                    example_inputs,
                    export_options
                )
            else:
                raise ValueError(
                    "The 'method' parameter only supports 'script' or 'trace', "
                    f"but value given was: {method}"
                )
        finally:
            module_mode.restore(self._model)
        if file_path is not None:
            with open_binary(file_path, "wb") as output_file:
                torch.jit.save(exported_module, output_file)
        return exported_module

    def _script(self, options: dict[str, object]) -> torch.jit.ScriptModule:
        with self._scripting_context():
            scripted: object = torch.jit.script(self._model, **options)
        if not isinstance(scripted, torch.jit.ScriptModule):
            raise TypeError(
                "Scripting a Module must return torch.jit.ScriptModule, got "
                f"{type(scripted).__name__}."
            )
        return scripted

    def _trace(
        self,
        example_inputs: object | None,
        options: dict[str, object]
    ) -> torch.jit.ScriptModule:
        resolved_inputs: object | None = example_inputs
        if resolved_inputs is None:
            resolved_inputs: object | None = self._model.example_input_array
        if resolved_inputs is None:
            raise ValueError(
                "Choosing method=`trace` requires either `example_inputs` "
                "or `model.example_input_array` to be defined."
            )
        check_inputs: object | None = options.get("check_inputs")
        if check_inputs is not None:
            options["check_inputs"] = self._input_transfer.prepare(check_inputs)
        prepared_inputs: object = self._input_transfer.prepare(resolved_inputs)
        with self._scripting_context():
            if isinstance(prepared_inputs, dict):
                if "example_kwarg_inputs" in options:
                    raise TypeError(
                        "Pass keyword tracing inputs through `example_inputs` or "
                        "`example_kwarg_inputs`, not both."
                    )
                traced: object = torch.jit.trace(
                    func=self._model,
                    example_inputs=None,
                    example_kwarg_inputs=prepared_inputs,
                    **options
                )
            else:
                traced: object = torch.jit.trace(
                    func=self._model,
                    example_inputs=prepared_inputs,
                    **options
                )
        if not isinstance(traced, torch.jit.ScriptModule):
            raise TypeError(
                "Tracing a Module must return torch.jit.ScriptModule, got "
                f"{type(traced).__name__}."
            )
        return traced

    @contextmanager
    def _scripting_context(self) -> Iterator[None]:
        previous_state: bool = Module._jit_is_scripting
        Module._jit_is_scripting: bool = True
        try:
            yield
        finally:
            Module._jit_is_scripting: bool = previous_state
