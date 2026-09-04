import inspect
import math
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import ClassVar, cast, override
import torch
from torch import nn
from torch.autograd.profiler_util import EventList
from torch.profiler import ExecutionTraceObserver, ProfilerAction, ProfilerActivity, record_function
from syntheticmind.primitives.exceptions import MisconfigurationError
from syntheticmind.profilers.profiler import Profiler
from syntheticmind.profilers.pytorch.module_recording import _ModuleRecordFunctions
from syntheticmind.profilers.pytorch.schedule import _ProfilerSchedule, _ProfilerScheduleFunction
from syntheticmind.storage.filesystem import StoragePath
from syntheticmind.primitives.types import TrainerStage

__all__: list[str] = ["PyTorchProfiler"]
type _ProfilerActivityFilter = dict[ProfilerActivity, list[str]]
type _ProfilerActivities = Iterable[ProfilerActivity | _ProfilerActivityFilter]
type _ProfilerOption = (
    bool | int | float | _ProfilerActivities | _ProfilerScheduleFunction
    | Callable[[torch.profiler.profile], None] | Callable[[], str]
    | torch.profiler._ExperimentalConfig | ExecutionTraceObserver | None
)
type _TableValue = int | str | bool | None

class PyTorchProfiler(Profiler):
    _step_functions: ClassVar[frozenset[str]] = frozenset({
        "training_step", "validation_step", "test_step", "predict_step"
    })
    _available_sort_keys: ClassVar[frozenset[str]] = frozenset({
        "cpu_time", "cuda_time", "cpu_time_total", "cuda_time_total",
        "self_cpu_time_total", "self_cuda_time_total", "cpu_memory_usage",
        "cuda_memory_usage", "self_cpu_memory_usage",
        "self_cuda_memory_usage", "count"
    })
    _boolean_options: ClassVar[frozenset[str]] = frozenset({
        "record_shapes", "profile_memory", "with_stack", "with_flops",
        "with_modules", "acc_events"
    })

    @override
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
        export_to_flame_graph: bool = False,
        metric: str = "self_cpu_time_total",
        **profiler_kwargs: _ProfilerOption
    ) -> None:
        super().__init__(dirpath=dirpath, filename=filename)
        self._validate_public_configuration(
            group_by_input_shapes,
            emit_nvtx,
            export_to_chrome,
            row_limit,
            sort_by_key,
            record_module_names,
            export_to_flame_graph,
            metric
        )
        options: dict[str, _ProfilerOption] = self._validate_profiler_options(
            dict(profiler_kwargs)
        )
        if group_by_input_shapes:
            options["record_shapes"] = True
        if export_to_flame_graph:
            options["with_stack"] = True
        if emit_nvtx:
            unsupported_nvtx_options: set[str] = set(options) - {"record_shapes"}
            if unsupported_nvtx_options:
                raise MisconfigurationError(
                    "NVTX profiling only accepts record_shapes; unsupported options: "
                    f"{sorted(unsupported_nvtx_options)}."
                )
        self._group_by_input_shapes: bool = group_by_input_shapes
        self._emit_nvtx: bool = emit_nvtx
        self._export_to_chrome: bool = export_to_chrome
        self._row_limit: int = row_limit
        self._record_module_names: bool = record_module_names
        self._profiler_options: dict[str, _ProfilerOption] = options
        self._export_to_flame_graph: bool = export_to_flame_graph
        self._metric: str = metric
        self._sort_by_key: str = sort_by_key or self._default_sort_key(options)
        if table_kwargs is not None and not isinstance(table_kwargs, dict):
            raise TypeError("`table_kwargs` must be a dictionary or None.")
        self._table_kwargs: dict[str, _TableValue] = (
            {} if table_kwargs is None else dict(table_kwargs)
        )
        self._validate_table_configuration()
        schedule_value: _ProfilerOption = options.get("schedule")
        self._schedule: _ProfilerSchedule | None = (
            _ProfilerSchedule(cast(_ProfilerScheduleFunction, schedule_value))
            if schedule_value is not None
            else None
        )
        self.profiler: torch.profiler.profile | None = None
        self.function_events: EventList | None = None
        self._module_recordings: _ModuleRecordFunctions | None = None
        self._nvtx_context: torch.autograd.profiler.emit_nvtx | None = None
        self._recording_map: dict[str, list[record_function]] = {}
        self._trace_index: int = 0
        self._last_step_action: str | None = None

    @override
    def start(self, action_name: str) -> None:
        try:
            self._create_profiler()
            recording: record_function = record_function(
                f"[syntheticmind][profile]{action_name}"
            )
            recording.__enter__()
            self._recording_map.setdefault(action_name, []).append(recording)
        except BaseException as exception:
            try:
                self._close_profiler()
            except BaseException as cleanup_exception:
                exception.add_note(
                    "PyTorch profiler cleanup after start failure raised "
                    f"{type(cleanup_exception).__name__}: {cleanup_exception}"
                )
            raise

    @override
    def stop(self, action_name: str) -> None:
        recordings: list[record_function] | None = self._recording_map.get(
            action_name
        )
        if not recordings:
            raise ValueError(
                f"Attempted to stop action {action_name!r} before it was started."
            )
        recording: record_function = recordings.pop()
        recording.__exit__(None, None, None)
        if recordings:
            return
        del self._recording_map[action_name]
        action_leaf: str = action_name.rsplit(".", 1)[-1]
        if self.profiler is None or action_leaf not in self._step_functions:
            return
        self._last_step_action: str | None = action_name
        if self._schedule is not None:
            self._schedule.prepare_step(action_name)
            self.profiler.step_num: int = self._schedule.step_number
        self.profiler.step()

    @override
    def setup(
        self,
        stage: TrainerStage,
        local_rank: int | None = None,
        log_dir: str | Path | None = None
    ) -> None:
        if self.profiler is not None or self._nvtx_context is not None:
            raise RuntimeError("The PyTorch profiler is already active.")
        super().setup(stage, local_rank, log_dir)
        self.function_events: EventList | None = None
        self._recording_map: dict[str, list[record_function]] = {}
        self._trace_index: int = 0
        self._last_step_action: str | None = None
        if self._schedule is not None:
            self._schedule.setup(stage)

    @override
    def teardown(self, stage: TrainerStage | None) -> None:
        try:
            self._close_profiler()
        finally:
            super().teardown(stage)

    @override
    def summary(self) -> str:
        if self._emit_nvtx:
            self._close_profiler()
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
        return self._format_statistics({"records": table})

    @override
    def _attach_module(self, module: nn.Module) -> None:
        if module is self._module:
            return
        if self._module_recordings is not None:
            self._module_recordings.__exit__(None, None, None)
            self._module_recordings: _ModuleRecordFunctions | None = None
        super()._attach_module(module)
        if self.profiler is not None and self._record_module_names:
            self._module_recordings: _ModuleRecordFunctions | None = (
                _ModuleRecordFunctions(module)
            )
            self._module_recordings.__enter__()

    def _create_profiler(self) -> None:
        if self.profiler is not None or self._nvtx_context is not None:
            return
        if self._emit_nvtx:
            record_shapes: bool = cast(
                bool,
                self._profiler_options.get("record_shapes", False)
            )
            nvtx_context: torch.autograd.profiler.emit_nvtx = (
                torch.autograd.profiler.emit_nvtx(record_shapes=record_shapes)
            )
            nvtx_context.__enter__()
            self._nvtx_context: torch.autograd.profiler.emit_nvtx | None = (
                nvtx_context
            )
            return
        options: dict[str, _ProfilerOption] = dict(self._profiler_options)
        if self._schedule is not None:
            options["schedule"] = self._schedule.next_action
            if "on_trace_ready" not in options:
                options["on_trace_ready"] = self._trace_ready
        active_profiler: torch.profiler.profile = torch.profiler.profile(**options)
        entered_profiler: torch.profiler.profile | None = active_profiler.__enter__()
        self.profiler: torch.profiler.profile | None = (
            active_profiler if entered_profiler is None else entered_profiler
        )
        if self._module is not None and self._record_module_names:
            self._module_recordings: _ModuleRecordFunctions | None = (
                _ModuleRecordFunctions(self._module)
            )
            self._module_recordings.__enter__()

    def _close_profiler(self) -> None:
        failures: list[BaseException] = []
        self._close_recording_ranges(failures)
        if self._module_recordings is not None:
            module_recordings: _ModuleRecordFunctions = self._module_recordings
            self._module_recordings: _ModuleRecordFunctions | None = None
            try:
                module_recordings.__exit__(None, None, None)
            except BaseException as exception:
                self._record_cleanup_failure(
                    failures,
                    "module recording cleanup",
                    exception
                )
        if self.profiler is not None:
            active_profiler: torch.profiler.profile = self.profiler
            self.profiler: torch.profiler.profile | None = None
            try:
                active_profiler.__exit__(None, None, None)
                self.function_events: EventList | None = active_profiler.events()
                if self._schedule is None and "on_trace_ready" not in self._profiler_options:
                    self._trace_ready(active_profiler)
            except BaseException as exception:
                self._record_cleanup_failure(
                    failures,
                    "profiler session cleanup",
                    exception
                )
        if self._nvtx_context is not None:
            nvtx_context: torch.autograd.profiler.emit_nvtx = self._nvtx_context
            self._nvtx_context: torch.autograd.profiler.emit_nvtx | None = None
            try:
                nvtx_context.__exit__(None, None, None)
            except BaseException as exception:
                self._record_cleanup_failure(
                    failures,
                    "NVTX session cleanup",
                    exception
                )
        if failures:
            primary_failure: BaseException = failures[0]
            cleanup_failure: BaseException
            for cleanup_failure in failures[1:]:
                primary_failure.add_note(
                    f"Additional profiler cleanup failure: {cleanup_failure}"
                )
            raise primary_failure

    def _close_recording_ranges(self, failures: list[BaseException]) -> None:
        action_name: str
        recordings: list[record_function]
        for action_name, recordings in self._recording_map.items():
            while recordings:
                recording: record_function = recordings.pop()
                try:
                    recording.__exit__(None, None, None)
                except BaseException as exception:
                    self._record_cleanup_failure(
                        failures,
                        f"recording cleanup for {action_name!r}",
                        exception
                    )
        self._recording_map: dict[str, list[record_function]] = {}

    def _trace_ready(self, profiler: torch.profiler.profile) -> None:
        if not self._export_to_chrome and not self._export_to_flame_graph:
            return
        if self.dirpath is None:
            warnings.warn(
                "PyTorchProfiler cannot export traces because its directory is unresolved.",
                RuntimeWarning,
                stacklevel=3
            )
            return
        action_name: str = self._last_step_action or "trace"
        trace_index: int = self._trace_index
        self._trace_index: int = self._trace_index + 1
        if self._export_to_chrome:
            trace_path: str = StoragePath(self.dirpath).join(
                self._prepare_filename(f"{action_name}-{trace_index}", ".json")
            )
            with StoragePath(trace_path).local_artifact() as local_trace_path:
                profiler.export_chrome_trace(str(local_trace_path))
        if self._export_to_flame_graph:
            if not any(bool(event.stack) for event in profiler.events()):
                raise RuntimeError(
                    "The active PyTorch profiler captured no stack frames for "
                    "flame-graph export."
                )
            stack_path: str = StoragePath(self.dirpath).join(
                self._prepare_filename(f"{action_name}-{trace_index}", ".stack")
            )
            with StoragePath(stack_path).local_artifact() as local_stack_path:
                profiler.export_stacks(
                    str(local_stack_path),
                    metric=self._metric
                )

    def _validate_public_configuration(
        self,
        group_by_input_shapes: bool,
        emit_nvtx: bool,
        export_to_chrome: bool,
        row_limit: int,
        sort_by_key: str | None,
        record_module_names: bool,
        export_to_flame_graph: bool,
        metric: str
    ) -> None:
        boolean_values: dict[str, bool] = {
            "group_by_input_shapes": group_by_input_shapes,
            "emit_nvtx": emit_nvtx,
            "export_to_chrome": export_to_chrome,
            "record_module_names": record_module_names,
            "export_to_flame_graph": export_to_flame_graph
        }
        name: str
        value: bool
        for name, value in boolean_values.items():
            if type(value) is not bool:
                raise TypeError(f"`{name}` must be a bool, got {value!r}.")
        if type(row_limit) is not int:
            raise TypeError(f"`row_limit` must be an int, got {row_limit!r}.")
        if row_limit < -1:
            raise ValueError(f"`row_limit` must be -1 or non-negative, got {row_limit}.")
        if sort_by_key is not None and not isinstance(sort_by_key, str):
            raise TypeError("`sort_by_key` must be a string or None.")
        if metric not in {"self_cpu_time_total", "self_cuda_time_total"}:
            raise ValueError("`metric` must be 'self_cpu_time_total' or 'self_cuda_time_total'.")

    def _validate_profiler_options(
        self,
        options: dict[str, _ProfilerOption]
    ) -> dict[str, _ProfilerOption]:
        valid_option_names: set[str] = set(
            inspect.signature(torch.profiler.profile).parameters
        )
        unknown_options: set[str] = set(options) - valid_option_names
        if unknown_options:
            raise MisconfigurationError(
                "Unsupported torch.profiler.profile options: "
                f"{sorted(unknown_options)}."
            )
        option_name: str
        for option_name in self._boolean_options:
            if option_name in options and type(options[option_name]) is not bool:
                raise TypeError(f"Profiler option `{option_name}` must be a bool.")
        use_cuda: _ProfilerOption = options.get("use_cuda")
        if use_cuda is not None and type(use_cuda) is not bool:
            raise TypeError("Profiler option `use_cuda` must be a bool or None.")
        schedule: _ProfilerOption = options.get("schedule")
        if schedule is not None and not callable(schedule):
            raise TypeError("Profiler option `schedule` must be callable or None.")
        handler: _ProfilerOption = options.get("on_trace_ready")
        if handler is not None and not callable(handler):
            raise TypeError("Profiler option `on_trace_ready` must be callable or None.")
        identifier: _ProfilerOption = options.get("custom_trace_id_callback")
        if identifier is not None and not callable(identifier):
            raise TypeError(
                "Profiler option `custom_trace_id_callback` must be callable or None."
            )
        timeout: _ProfilerOption = options.get("post_processing_timeout_s")
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError(
                "Profiler option `post_processing_timeout_s` must be finite and positive."
            )
        self._normalize_activities(options)
        return options

    def _normalize_activities(
        self,
        options: dict[str, _ProfilerOption]
    ) -> None:
        activity_value: _ProfilerOption = options.get("activities")
        if activity_value is None:
            return
        if isinstance(activity_value, str | bytes | dict) or not isinstance(
            activity_value,
            Iterable
        ):
            raise TypeError("Profiler option `activities` must be an iterable of activities.")
        activities: list[ProfilerActivity | _ProfilerActivityFilter] = list(
            cast(_ProfilerActivities, activity_value)
        )
        activity: ProfilerActivity | _ProfilerActivityFilter
        for activity in activities:
            if isinstance(activity, ProfilerActivity):
                continue
            if not isinstance(activity, dict):
                raise TypeError(
                    "Profiler activities must contain ProfilerActivity values or filters."
                )
            activity_kind: ProfilerActivity
            filters: list[str]
            for activity_kind, filters in activity.items():
                if not isinstance(activity_kind, ProfilerActivity):
                    raise TypeError(
                        "Profiler activity filter keys must be ProfilerActivity values."
                    )
                if not isinstance(filters, list) or not all(
                    isinstance(name, str) for name in filters
                ):
                    raise TypeError("Profiler activity filter values must be lists of strings.")
        options["activities"] = activities

    def _validate_table_configuration(self) -> None:
        if self._sort_by_key not in self._available_sort_keys:
            raise MisconfigurationError(
                f"Invalid profiler sort key {self._sort_by_key!r}; expected one of "
                f"{sorted(self._available_sort_keys)}."
            )
        valid_table_keys: set[str] = set(
            inspect.signature(EventList.table).parameters
        ) - {"self", "sort_by", "row_limit"}
        key: str
        for key in self._table_kwargs:
            if key in {"sort_by", "row_limit"}:
                raise KeyError(
                    f"Profiler table option {key!r} is controlled by PyTorchProfiler."
                )
            if key not in valid_table_keys:
                raise KeyError(
                    f"Unsupported profiler table option {key!r}; expected one of "
                    f"{sorted(valid_table_keys)}."
                )

    def _default_sort_key(
        self,
        options: dict[str, _ProfilerOption]
    ) -> str:
        use_cuda: _ProfilerOption = options.get("use_cuda")
        if use_cuda is True:
            return "cuda_time_total"
        activities_value: _ProfilerOption = options.get("activities")
        if isinstance(activities_value, Iterable):
            activity: ProfilerActivity | _ProfilerActivityFilter
            for activity in cast(_ProfilerActivities, activities_value):
                if activity == ProfilerActivity.CUDA:
                    return "cuda_time_total"
                if isinstance(activity, dict) and ProfilerActivity.CUDA in activity:
                    return "cuda_time_total"
        supported_activities: set[ProfilerActivity] = torch.profiler.supported_activities()
        if activities_value is None and ProfilerActivity.CUDA in supported_activities:
            return "cuda_time_total"
        return "cpu_time_total"

    def _record_cleanup_failure(
        self,
        failures: list[BaseException],
        operation: str,
        exception: BaseException
    ) -> None:
        exception.add_note(f"PyTorch profiler failed during {operation}.")
        failures.append(exception)
