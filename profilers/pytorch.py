from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import cast, override

import torch
from torch import nn
from torch.autograd.profiler_util import EventList
from torch.profiler import ProfilerAction, ProfilerActivity, record_function
from torch.utils.hooks import RemovableHandle

from syntheticmind.profilers.profiler import Profiler
from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.filesystem import join_path, local_artifact_path

__all__: list[str] = ["PyTorchProfiler", "RegisterRecordFunction"]

type _TableValue = int | str | bool | None


class RegisterRecordFunction:
    def __init__(self, model: nn.Module) -> None:
        self._model: nn.Module = model
        self._handles: list[RemovableHandle] = []
        self._recordings: dict[int, list[record_function]] = {}
        self._module_names: dict[int, str] = {
            id(module): name
            for name, module in model.named_modules()
        }

    def _start_recording_forward(
        self,
        module: nn.Module,
        inputs: tuple[object, ...]
    ) -> None:
        del inputs
        name: str = self._module_names.get(id(module), "")
        qualified_name: str = f"{type(module).__module__}.{type(module).__qualname__}"
        recording: record_function = record_function(
            f"[syntheticmind][module]{qualified_name}: {name or '<root>'}"
        )
        recording.__enter__()
        self._recordings.setdefault(id(module), []).append(recording)

    def _stop_recording_forward(
        self,
        module: nn.Module,
        inputs: tuple[object, ...],
        output: object
    ) -> None:
        del inputs, output
        recordings: list[record_function] | None = self._recordings.get(id(module))
        if not recordings:
            return
        recording: record_function = recordings.pop()
        recording.__exit__(None, None, None)

    def __enter__(self) -> RegisterRecordFunction:
        for module in self._model.modules():
            if isinstance(module, torch.jit.ScriptModule):
                continue
            self._handles.append(
                module.register_forward_pre_hook(self._start_recording_forward)
            )
            self._handles.append(
                module.register_forward_hook(
                    self._stop_recording_forward,
                    always_call=True
                )
            )
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None
    ) -> None:
        del exception_type, exception, traceback
        for recordings in self._recordings.values():
            while recordings:
                recordings.pop().__exit__(None, None, None)
        self._recordings.clear()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class PyTorchProfiler(Profiler):
    STEP_FUNCTIONS: set[str] = {
        "training_step",
        "validation_step",
        "test_step",
        "predict_step"
    }
    AVAILABLE_SORT_KEYS: set[str] = {
        "cpu_time",
        "cuda_time",
        "cpu_time_total",
        "cuda_time_total",
        "self_cpu_time_total",
        "self_cuda_time_total",
        "cpu_memory_usage",
        "cuda_memory_usage",
        "self_cpu_memory_usage",
        "self_cuda_memory_usage",
        "count"
    }

    def __init__(
        self,
        dirpath: str | Path | None = None,
        filename: str | None = None,
        group_by_input_shapes: bool = False,
        emit_nvtx: bool = False,
        export_to_chrome: bool = True,
        row_limit: int = 20,
        sort_by_key: str | None = None,
        record_module_names: bool = True,
        table_kwargs: dict[str, _TableValue] | None = None,
        **profiler_kwargs: object
    ) -> None:
        super().__init__(dirpath=dirpath, filename=filename)
        self._group_by_input_shapes: bool = bool(
            group_by_input_shapes and profiler_kwargs.get("record_shapes", False)
        )
        self._emit_nvtx: bool = emit_nvtx
        self._export_to_chrome: bool = export_to_chrome
        self._row_limit: int = row_limit
        self._record_module_names: bool = record_module_names
        self._profiler_kwargs: dict[str, object] = dict(profiler_kwargs)
        self._sort_by_key: str = sort_by_key or self._default_sort_by_key()
        self._table_kwargs: dict[str, _TableValue] = (
            {} if table_kwargs is None else dict(table_kwargs)
        )
        self._export_to_flame_graph: bool = bool(
            self._profiler_kwargs.get("export_to_flame_graph", False)
        )
        self._metric: str = str(
            self._profiler_kwargs.get("metric", "self_cpu_time_total")
        )
        if self._export_to_flame_graph:
            self._profiler_kwargs["with_stack"] = True
        self.profiler: torch.profiler.profile | None = None
        self.function_events: EventList | None = None
        self._register: RegisterRecordFunction | None = None
        self._nvtx_context: AbstractContextManager[object] | None = None
        self._recording_map: dict[str, list[record_function]] = {}
        self._trace_index: int = 0
        self._last_step_action: str | None = None
        if self._sort_by_key not in self.AVAILABLE_SORT_KEYS:
            raise MisconfigurationError(
                f"Found sort_by_key {self._sort_by_key!r}. "
                f"Expected one of {sorted(self.AVAILABLE_SORT_KEYS)}."
            )
        schedule: object = self._profiler_kwargs.get("schedule")
        if schedule is not None:
            if not callable(schedule):
                raise MisconfigurationError(f"Schedule must be callable, got {schedule!r}.")
            action: object = schedule(0)
            if not isinstance(action, ProfilerAction):
                raise MisconfigurationError(
                    "Schedule must return torch.profiler.ProfilerAction, got "
                    f"{type(action).__name__}."
                )
        for key in self._table_kwargs:
            if key in {"sort_by", "row_limit"}:
                raise KeyError(
                    f"Found invalid table_kwargs key: {key}. "
                    "This value is already controlled by the profiler."
                )
        valid_table_keys: set[str] = set(inspect.signature(EventList.table).parameters) - {
            "self",
            "sort_by",
            "row_limit"
        }
        for key in self._table_kwargs:
            if key not in valid_table_keys:
                raise KeyError(
                    f"Found invalid table_kwargs key: {key}. "
                    f"Expected one of {sorted(valid_table_keys)}."
                )

    def _default_activities(self) -> list[ProfilerActivity]:
        activities: list[ProfilerActivity] = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        return activities

    def _default_sort_by_key(self) -> str:
        activities: object = self._profiler_kwargs.get("activities", ())
        uses_cuda: bool = bool(self._profiler_kwargs.get("use_cuda", False))
        if isinstance(activities, (list, tuple, set)):
            uses_cuda = uses_cuda or ProfilerActivity.CUDA in activities
        if not activities:
            uses_cuda = uses_cuda or torch.cuda.is_available()
        return f"{'cuda' if uses_cuda else 'cpu'}_time_total"

    def _trace_ready(self, profiler: torch.profiler.profile) -> None:
        if "on_trace_ready" in self._profiler_kwargs:
            callback: object = self._profiler_kwargs["on_trace_ready"]
            if callable(callback):
                cast(Callable[[torch.profiler.profile], object], callback)(profiler)
            return
        if not self._export_to_chrome or self.dirpath is None:
            return
        path: str = join_path(
            self.dirpath,
            self._prepare_filename(
                action_name=f"{self._last_step_action or 'trace'}-{self._trace_index}",
                extension=".json"
            )
        )
        self._trace_index: int = self._trace_index + 1
        with local_artifact_path(path) as local_path:
            profiler.export_chrome_trace(str(local_path))

    def _export_stacks(self, profiler: torch.profiler.profile) -> None:
        if not self._export_to_flame_graph:
            return
        if self.dirpath is None:
            warnings.warn(
                "The PyTorchProfiler cannot export stacks because dirpath is None.",
                RuntimeWarning,
                stacklevel=3
            )
            return
        path: str = join_path(
            self.dirpath,
            self._prepare_filename(
                action_name=self._last_step_action or "stacks",
                extension=".stack"
            )
        )
        with local_artifact_path(path) as local_path:
            profiler.export_stacks(str(local_path), metric=self._metric)

    def _create_profiler(self) -> None:
        if self.profiler is not None or self._nvtx_context is not None:
            return
        if self._emit_nvtx:
            if not torch.cuda.is_available():
                raise RuntimeError("emit_nvtx=True requires an available CUDA runtime.")
            self._nvtx_context = torch.autograd.profiler.emit_nvtx()
            self._nvtx_context.__enter__()
            return
        kwargs: dict[str, object] = dict(self._profiler_kwargs)
        kwargs.setdefault("activities", self._default_activities())
        if "schedule" in kwargs:
            kwargs.setdefault("on_trace_ready", self._trace_ready)
        valid_parameters: set[str] = set(
            inspect.signature(torch.profiler.profile).parameters
        )
        kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in valid_parameters
        }
        profiler_factory: Callable[..., torch.profiler.profile] = cast(
            Callable[..., torch.profiler.profile],
            torch.profiler.profile
        )
        self.profiler = profiler_factory(**kwargs)
        entered: torch.profiler.profile | None = self.profiler.__enter__()
        if entered is not None:
            self.profiler = entered
        if self._module is not None and self._record_module_names:
            self._register = RegisterRecordFunction(self._module)
            self._register.__enter__()

    @override
    def _attach_module(self, module: nn.Module) -> None:
        if module is self._module:
            return
        if self._register is not None:
            self._register.__exit__(None, None, None)
            self._register = None
        super()._attach_module(module)
        if self.profiler is not None and self._record_module_names:
            self._register = RegisterRecordFunction(module)
            self._register.__enter__()

    @override
    def start(self, action_name: str) -> None:
        self._create_profiler()
        recording: record_function = record_function(
            f"[syntheticmind][profile]{action_name}"
        )
        recording.__enter__()
        self._recording_map.setdefault(action_name, []).append(recording)

    @override
    def stop(self, action_name: str) -> None:
        recordings: list[record_function] | None = self._recording_map.get(action_name)
        if recordings:
            recordings.pop().__exit__(None, None, None)
            if not recordings:
                del self._recording_map[action_name]
        action_leaf: str = action_name.rsplit(".", 1)[-1]
        if self.profiler is not None and action_leaf in self.STEP_FUNCTIONS:
            self._last_step_action = action_name
            self.profiler.step()

    def _close_profiler(self) -> None:
        for action_name in list(self._recording_map):
            while action_name in self._recording_map:
                self.stop(action_name)
        if self._register is not None:
            self._register.__exit__(None, None, None)
            self._register = None
        if self.profiler is not None:
            active_profiler: torch.profiler.profile = self.profiler
            active_profiler.__exit__(None, None, None)
            self.function_events = active_profiler.events()
            if self._export_to_chrome and "schedule" not in self._profiler_kwargs:
                self._trace_ready(active_profiler)
            self._export_stacks(active_profiler)
            self.profiler = None
        if self._nvtx_context is not None:
            self._nvtx_context.__exit__(None, None, None)
            self._nvtx_context = None

    @override
    def summary(self) -> str:
        if not self._profiler_kwargs.get("enabled", True) or self._emit_nvtx:
            return ""
        self._close_profiler()
        if not self.function_events:
            return ""
        averaged_events: EventList = self.function_events.key_averages(
            group_by_input_shapes=self._group_by_input_shapes
        )
        table_formatter: Callable[..., str] = cast(
            Callable[..., str],
            averaged_events.table
        )
        table: str = table_formatter(
            sort_by=self._sort_by_key,
            row_limit=self._row_limit,
            **self._table_kwargs
        )
        return self._stats_to_str({"records": table})

    @override
    def setup(
        self,
        stage: str,
        local_rank: int | None = None,
        log_dir: str | None = None
    ) -> None:
        super().setup(stage=stage, local_rank=local_rank, log_dir=log_dir)
        self.function_events: EventList | None = None
        self._trace_index: int = 0
        self._last_step_action: str | None = None

    @override
    def teardown(self, stage: str | None) -> None:
        self._close_profiler()
        super().teardown(stage=stage)
