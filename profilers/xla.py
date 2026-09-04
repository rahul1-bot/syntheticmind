from __future__ import annotations

import importlib
import importlib.util
from typing import Protocol, cast, override

from syntheticmind.profilers.profiler import Profiler

__all__: list[str] = ["XLAProfiler"]


class _XLARecording(Protocol):
    def __enter__(self) -> object:
        ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None
    ) -> object:
        ...


class _XLAProfilerModule(Protocol):
    def start_server(self, port: int) -> object:
        ...

    def StepTrace(self, action_name: str, step_num: int) -> _XLARecording:
        ...

    def Trace(self, action_name: str) -> _XLARecording:
        ...


class XLAProfiler(Profiler):
    _step_functions: frozenset[str] = frozenset(
        {"validation_step", "test_step", "predict_step"}
    )
    _record_functions: frozenset[str] = frozenset(
        {
            "training_step",
            "backward",
            "validation_step",
            "test_step",
            "predict_step"
        }
    )

    def __init__(self, port: int = 9012) -> None:
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError(f"`port` must be an integer from 1 through 65535, got {port!r}.")
        if importlib.util.find_spec("torch_xla") is None:
            raise ModuleNotFoundError("XLAProfiler requires the optional torch_xla package.")
        super().__init__(dirpath=None, filename=None)
        self.port: int = port
        self._recording_map: dict[str, _XLARecording] = {}
        self._step_recording_map: dict[str, int] = {}
        self._server: object | None = None

    @override
    def start(self, action_name: str) -> None:
        action_leaf: str = action_name.rsplit(".", 1)[-1]
        if action_leaf not in self._record_functions:
            return
        xla_profiler: _XLAProfilerModule = cast(
            _XLAProfilerModule,
            importlib.import_module("torch_xla.debug.profiler")
        )

        if self._server is None:
            self._server = xla_profiler.start_server(self.port)
        recording: _XLARecording
        if action_leaf in self._step_functions:
            recording = xla_profiler.StepTrace(
                action_name,
                step_num=self._get_step_num(action_name)
            )
        else:
            recording = xla_profiler.Trace(action_name)
        recording.__enter__()
        self._recording_map[action_name] = recording

    @override
    def stop(self, action_name: str) -> None:
        recording: _XLARecording | None = self._recording_map.pop(
            action_name,
            None
        )
        if recording is not None:
            recording.__exit__(None, None, None)

    def _get_step_num(self, action_name: str) -> int:
        step_number: int = self._step_recording_map.get(action_name, 0) + 1
        self._step_recording_map[action_name] = step_number
        return step_number

    @override
    def teardown(self, stage: str | None) -> None:
        for recording in self._recording_map.values():
            recording.__exit__(None, None, None)
        self._recording_map.clear()
        super().teardown(stage=stage)
