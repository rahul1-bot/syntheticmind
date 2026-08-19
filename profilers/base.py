from typing import override

from syntheticmind.profilers.profiler import Profiler

__all__: list[str] = ["PassThroughProfiler"]


class PassThroughProfiler(Profiler):
    @override
    def start(self, action_name: str) -> None:
        del action_name

    @override
    def stop(self, action_name: str) -> None:
        del action_name
