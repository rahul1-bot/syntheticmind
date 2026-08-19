import copy
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib import import_module
from importlib.util import find_spec
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Literal, cast

import torch

from syntheticmind.core.model_export import ModelExportInputTransfer
from syntheticmind.core.module import Module
from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.filesystem import (
    basename,
    copy_local_file,
    is_local_path,
    normalize_path
)
from syntheticmind.utilities.model_helpers import ModuleMode

__all__: list[str] = ["ModelTensorRTCompiler"]


class TensorRTArtifactDestination:
    @contextmanager
    def resolve(
        self,
        file_path: str | Path | BytesIO
    ) -> Iterator[str | Path | BytesIO]:
        if isinstance(file_path, BytesIO):
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
            raise ValueError("TensorRT file path must include an artifact name.")
        with TemporaryDirectory(prefix="syntheticmind-tensorrt-") as directory_name:
            local_path: Path = Path(directory_name) / artifact_name
            yield local_path
            if not local_path.is_file():
                raise RuntimeError(
                    "Torch-TensorRT did not create the requested artifact."
                )
            copy_local_file(local_path, remote_path)


class TensorRTInputPreparation:
    def __init__(self, model: Module) -> None:
        self._model: Module = model
        self._transfer: ModelExportInputTransfer = ModelExportInputTransfer(model)

    def resolve(self, input_sample: object | None) -> object:
        resolved_sample: object | None = input_sample
        if resolved_sample is None:
            resolved_sample: object | None = self._model.example_input_array
        if resolved_sample is None:
            raise ValueError(
                "Could not export to TensorRT since neither `input_sample` nor "
                "`model.example_input_array` attribute is set."
            )
        return resolved_sample

    def prepare(
        self,
        input_sample: object,
        ir: Literal["default", "dynamo", "ts"]
    ) -> tuple[tuple[object, ...], dict[str, object] | None]:
        copied_sample: object = copy.deepcopy(input_sample)
        prepared_sample: object = self._transfer.prepare(copied_sample)
        positional_inputs: tuple[object, ...]
        keyword_inputs: dict[str, object] | None = None
        if isinstance(prepared_sample, dict):
            positional_inputs: tuple[object, ...] = ()
            keyword_inputs: dict[str, object] | None = prepared_sample
        elif isinstance(prepared_sample, tuple):
            if prepared_sample and isinstance(prepared_sample[-1], dict):
                positional_inputs: tuple[object, ...] = prepared_sample[:-1]
                keyword_inputs: dict[str, object] | None = prepared_sample[-1]
            else:
                positional_inputs: tuple[object, ...] = prepared_sample
        elif isinstance(prepared_sample, list):
            positional_inputs: tuple[object, ...] = tuple(prepared_sample)
        else:
            positional_inputs: tuple[object, ...] = (prepared_sample,)
        if ir == "ts" and keyword_inputs is not None:
            raise ValueError(
                "TensorRT TorchScript IR does not support keyword input samples."
            )
        return positional_inputs, keyword_inputs


class TensorRTRuntime:
    def __init__(self, model: Module) -> None:
        self._model: Module = model

    def load_package(self) -> ModuleType:
        if find_spec("torch_tensorrt") is None:
            raise ModuleNotFoundError(
                f"`{type(self._model).__name__}.to_tensorrt` requires "
                "`torch_tensorrt` to be installed."
            )
        loaded_package: object = import_module("torch_tensorrt")
        if not isinstance(loaded_package, ModuleType):
            raise TypeError(
                "Importing `torch_tensorrt` must return a Python module."
            )
        return loaded_package

    def target_device(
        self,
        original_device: torch.device,
        default_device: str | torch.device
    ) -> torch.device:
        if original_device.type == "cuda":
            return original_device
        target_device: torch.device = torch.device(default_device)
        if not torch.cuda.is_available() or target_device.type != "cuda":
            raise MisconfigurationError(
                "TensorRT only supports CUDA devices. The current device is "
                f"{original_device}. Please set `default_device` to a CUDA device."
            )
        return target_device

    def validate_configuration(
        self,
        file_path: str | Path | BytesIO | None,
        ir: str,
        output_format: str,
        compile_kwargs: dict[str, object]
    ) -> None:
        if ir not in {"default", "dynamo", "ts"}:
            raise ValueError(
                "`ir` must be 'default', 'dynamo', or 'ts', got "
                f"{ir!r}."
            )
        if output_format not in {"exported_program", "torchscript"}:
            raise ValueError(
                "`output_format` must be 'exported_program' or 'torchscript', "
                f"got {output_format!r}."
            )
        if file_path is not None and ir == "ts" and output_format != "torchscript":
            raise ValueError(
                "TensorRT with IR mode 'ts' only supports output format "
                f"'torchscript'. The current output format is {output_format}."
            )
        duplicate_inputs: list[str] = [
            name
            for name in ("inputs", "arg_inputs", "kwarg_inputs")
            if name in compile_kwargs
        ]
        if duplicate_inputs:
            joined_inputs: str = ", ".join(duplicate_inputs)
            raise TypeError(
                "Pass TensorRT compile inputs through `input_sample`, not "
                f"through compile options: {joined_inputs}."
            )

    def operation(
        self,
        package: ModuleType,
        name: str
    ) -> Callable[..., object]:
        candidate: object = getattr(package, name, None)
        if not callable(candidate):
            raise RuntimeError(
                f"Installed `torch_tensorrt` does not expose callable `{name}`."
            )
        return cast(Callable[..., object], candidate)


class ModelTensorRTCompiler:
    def __init__(self, model: Module) -> None:
        self._model: Module = model
        self._inputs: TensorRTInputPreparation = TensorRTInputPreparation(model)
        self._runtime: TensorRTRuntime = TensorRTRuntime(model)
        self._destination: TensorRTArtifactDestination = TensorRTArtifactDestination()

    def compile(
        self,
        file_path: str | Path | BytesIO | None,
        input_sample: object | None,
        ir: Literal["default", "dynamo", "ts"],
        output_format: Literal["exported_program", "torchscript"],
        retrace: bool,
        default_device: str | torch.device,
        compile_kwargs: dict[str, object]
    ) -> torch.jit.ScriptModule | torch.fx.GraphModule:
        package: ModuleType = self._runtime.load_package()
        self._runtime.validate_configuration(
            file_path,
            ir,
            output_format,
            compile_kwargs
        )
        resolved_sample: object = self._inputs.resolve(input_sample)
        original_device: torch.device = self._model.device
        target_device: torch.device = self._runtime.target_device(
            original_device,
            default_device
        )
        module_mode: ModuleMode = ModuleMode()
        module_mode.capture(self._model)
        moved_model: bool = False
        try:
            if target_device != original_device:
                self._model.to(target_device)
                moved_model: bool = True
            positional_inputs: tuple[object, ...]
            keyword_inputs: dict[str, object] | None
            positional_inputs, keyword_inputs = self._inputs.prepare(
                resolved_sample,
                ir
            )
            self._model.eval()
            compiled_module: object = self._compile_module(
                package,
                ir,
                positional_inputs,
                keyword_inputs,
                compile_kwargs
            )
        finally:
            module_mode.restore(self._model)
            if moved_model:
                self._model.to(original_device)
        if file_path is not None:
            self._save_module(
                package,
                compiled_module,
                file_path,
                ir,
                output_format,
                retrace,
                positional_inputs,
                keyword_inputs
            )
        return cast(
            torch.jit.ScriptModule | torch.fx.GraphModule,
            compiled_module
        )

    def _compile_module(
        self,
        package: ModuleType,
        ir: Literal["default", "dynamo", "ts"],
        positional_inputs: tuple[object, ...],
        keyword_inputs: dict[str, object] | None,
        compile_kwargs: dict[str, object]
    ) -> object:
        compile_function: Callable[..., object] = self._runtime.operation(
            package,
            "compile"
        )
        options: dict[str, object] = dict(compile_kwargs)
        if keyword_inputs is not None:
            options["kwarg_inputs"] = keyword_inputs
        if ir == "ts":
            with self._scripting_context():
                return compile_function(
                    module=self._model,
                    ir=ir,
                    inputs=positional_inputs,
                    **options
                )
        return compile_function(
            module=self._model,
            ir=ir,
            inputs=positional_inputs,
            **options
        )

    def _save_module(
        self,
        package: ModuleType,
        compiled_module: object,
        file_path: str | Path | BytesIO,
        ir: Literal["default", "dynamo", "ts"],
        output_format: Literal["exported_program", "torchscript"],
        retrace: bool,
        positional_inputs: tuple[object, ...],
        keyword_inputs: dict[str, object] | None
    ) -> None:
        with self._destination.resolve(file_path) as destination:
            if ir == "ts":
                if not isinstance(
                    compiled_module,
                    torch.jit.ScriptModule | torch.jit.ScriptFunction
                ):
                    raise TypeError(
                        "TensorRT TorchScript compilation must return a "
                        "ScriptModule or ScriptFunction."
                    )
                torch.jit.save(compiled_module, destination)
                return
            save_function: Callable[..., object] = self._runtime.operation(
                package,
                "save"
            )
            save_options: dict[str, object] = {
                "inputs": positional_inputs,
                "output_format": output_format,
                "retrace": retrace
            }
            if keyword_inputs is not None:
                save_options["kwarg_inputs"] = keyword_inputs
            save_function(compiled_module, destination, **save_options)

    @contextmanager
    def _scripting_context(self) -> Iterator[None]:
        previous_state: bool = Module._jit_is_scripting
        Module._jit_is_scripting: bool = True
        try:
            yield
        finally:
            Module._jit_is_scripting: bool = previous_state
