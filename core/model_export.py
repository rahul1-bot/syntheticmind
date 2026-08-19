from collections.abc import Iterator
from contextlib import contextmanager
from importlib.util import find_spec
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import torch

from syntheticmind.core.datamodule import DataModule
from syntheticmind.core.module import Module
from syntheticmind.core.trainer import Trainer
from syntheticmind.utilities.filesystem import (
    basename,
    copy_local_file,
    dirname,
    is_local_path,
    join_path,
    normalize_path,
    open_binary
)
from syntheticmind.utilities.model_helpers import ModuleMode, is_overridden
from syntheticmind.utilities.types import Batch

__all__: list[str] = ["ModelONNXExporter", "ModelTorchScriptExporter"]


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


class ModelONNXArtifactDestination:
    @contextmanager
    def resolve(
        self,
        file_path: str | Path | BytesIO | None
    ) -> Iterator[str | Path | BytesIO | None]:
        if file_path is None or isinstance(file_path, BytesIO):
            yield file_path
            return
        if is_local_path(file_path):
            local_path: Path = Path(normalize_path(file_path))
            local_path.parent.mkdir(parents=True, exist_ok=True)
            yield local_path
            return
        remote_path: str = normalize_path(file_path)
        artifact_name: str = basename(remote_path)
        if not artifact_name:
            raise ValueError("ONNX file path must include an artifact name.")
        with TemporaryDirectory(prefix="syntheticmind-onnx-") as directory_name:
            local_directory: Path = Path(directory_name)
            local_path: Path = local_directory / artifact_name
            yield local_path
            artifact: Path
            for artifact in local_directory.iterdir():
                if artifact.is_file():
                    destination: str = join_path(
                        dirname(remote_path),
                        artifact.name
                    )
                    copy_local_file(artifact, destination)


class ModelONNXExporter:
    def __init__(self, model: Module) -> None:
        self._model: Module = model
        self._input_transfer: ModelExportInputTransfer = ModelExportInputTransfer(model)
        self._destination: ModelONNXArtifactDestination = ModelONNXArtifactDestination()

    def export(
        self,
        file_path: str | Path | BytesIO | None,
        input_sample: object | None,
        options: dict[str, object]
    ) -> torch.onnx.ONNXProgram | None:
        export_options: dict[str, object] = dict(options)
        self._validate_dependencies(export_options)
        self._validate_destination(file_path, export_options)
        resolved_sample: object | None = input_sample
        if resolved_sample is None:
            resolved_sample: object | None = self._model.example_input_array
        if resolved_sample is None:
            raise ValueError(
                "Could not export to ONNX since neither `input_sample` nor "
                "`model.example_input_array` attribute is set."
            )
        module_mode: ModuleMode = ModuleMode()
        module_mode.capture(self._model)
        try:
            self._set_export_mode(export_options)
            positional_inputs: tuple[object, ...] = self._prepare_inputs(
                resolved_sample,
                export_options
            )
            with self._destination.resolve(file_path) as destination:
                exported_program: object = torch.onnx.export(
                    self._model,
                    positional_inputs,
                    destination,
                    **export_options
                )
        finally:
            module_mode.restore(self._model)
        if exported_program is not None and not isinstance(
            exported_program,
            torch.onnx.ONNXProgram
        ):
            raise TypeError(
                "ONNX export must return torch.onnx.ONNXProgram or None, got "
                f"{type(exported_program).__name__}."
            )
        return exported_program

    def _prepare_inputs(
        self,
        input_sample: object,
        options: dict[str, object]
    ) -> tuple[object, ...]:
        prepared_sample: object = self._input_transfer.prepare(input_sample)
        keyword_inputs: object | None = options.get("kwargs")
        if isinstance(prepared_sample, dict):
            if keyword_inputs is not None:
                if not isinstance(keyword_inputs, dict):
                    raise TypeError(
                        "The ONNX exporter `kwargs` option must be a dictionary or None."
                    )
                if keyword_inputs:
                    raise TypeError(
                        "Pass keyword ONNX inputs through `input_sample` or the "
                        "exporter `kwargs` option, not both."
                    )
            options["kwargs"] = prepared_sample
            return ()
        if keyword_inputs is not None:
            if not isinstance(keyword_inputs, dict):
                raise TypeError(
                    "The ONNX exporter `kwargs` option must be a dictionary or None."
                )
            options["kwargs"] = self._input_transfer.prepare(keyword_inputs)
        if isinstance(prepared_sample, tuple):
            return prepared_sample
        return (prepared_sample,)

    def _set_export_mode(self, options: dict[str, object]) -> None:
        training_mode: object = options.get(
            "training",
            torch.onnx.TrainingMode.EVAL
        )
        if training_mode == torch.onnx.TrainingMode.PRESERVE:
            return
        if training_mode == torch.onnx.TrainingMode.TRAINING:
            self._model.train()
            return
        self._model.eval()

    def _validate_dependencies(self, options: dict[str, object]) -> None:
        model_name: str = type(self._model).__name__
        if find_spec("onnx") is None:
            raise ModuleNotFoundError(
                f"`{model_name}.to_onnx()` requires `onnx` to be installed."
            )
        uses_dynamo: bool = self._uses_dynamo(options)
        if uses_dynamo and find_spec("onnxscript") is None:
            raise ModuleNotFoundError(
                f"`{model_name}.to_onnx(dynamo=True)` requires `onnxscript` "
                "to be installed."
            )
        if uses_dynamo and options.get("verify") is True and find_spec(
            "onnxruntime"
        ) is None:
            raise ModuleNotFoundError(
                f"`{model_name}.to_onnx(verify=True)` requires `onnxruntime` "
                "to be installed."
            )

    def _validate_destination(
        self,
        file_path: str | Path | BytesIO | None,
        options: dict[str, object]
    ) -> None:
        if not self._uses_dynamo(options) and file_path is None:
            raise ValueError(
                "ONNX export with `dynamo=False` requires `file_path` because "
                "the legacy exporter does not return an in-memory program."
            )

    def _uses_dynamo(self, options: dict[str, object]) -> bool:
        if "dynamo" in options:
            return options["dynamo"] is True
        keyword_defaults: object = getattr(
            torch.onnx.export,
            "__kwdefaults__",
            None
        )
        return (
            isinstance(keyword_defaults, dict)
            and keyword_defaults.get("dynamo") is True
        )
