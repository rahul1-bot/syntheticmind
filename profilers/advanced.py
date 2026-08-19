from __future__ import annotations

import cProfile
import io
import pstats
from collections import defaultdict
from pathlib import Path
from typing import override

from syntheticmind.profilers.profiler import Profiler

__all__: list[str] = ["AdvancedProfiler"]


class AdvancedProfiler(Profiler):
    def __init__(
        self,
        dirpath: str | Path | None = None,
        filename: str | None = None,
        line_count_restriction: int | float = 1.0,
        dump_stats: bool = False
    ) -> None:
        super().__init__(dirpath=dirpath, filename=filename)
        if isinstance(line_count_restriction, bool) or not isinstance(
            line_count_restriction,
            (int, float)
        ):
            raise TypeError("`line_count_restriction` must be an int or float.")
        if line_count_restriction < 0:
            raise ValueError("`line_count_restriction` must be non-negative.")
        if type(dump_stats) is not bool:
            raise TypeError(f"`dump_stats` must be a bool, got {dump_stats!r}.")
        self.profiled_actions: defaultdict[str, cProfile.Profile] = defaultdict(cProfile.Profile)
        self.line_count_restriction: int | float = line_count_restriction
        self.dump_stats: bool = dump_stats
        self._active_actions: list[str] = []

    @override
    def start(self, action_name: str) -> None:
        if action_name in self._active_actions:
            raise ValueError(f"Attempted to start {action_name} which has already started.")
        for profile in self.profiled_actions.values():
            profile.disable()
        self.profiled_actions[action_name].enable()
        self._active_actions.append(action_name)

    @override
    def stop(self, action_name: str) -> None:
        profile: cProfile.Profile | None = self.profiled_actions.get(action_name)
        if profile is None or action_name not in self._active_actions:
            raise ValueError(
                f"Attempting to stop recording an action ({action_name}) which was never started."
            )
        profile.disable()
        self._active_actions.remove(action_name)
        if self._active_actions:
            self.profiled_actions[self._active_actions[-1]].enable()

    def _dump_stats(self, action_name: str, profile: cProfile.Profile) -> None:
        if self.dirpath is None:
            raise RuntimeError("AdvancedProfiler requires dirpath to dump raw statistics.")
        directory: Path = Path(self.dirpath).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        profile.dump_stats(
            str(directory / self._prepare_filename(action_name=action_name, extension=".prof"))
        )

    @override
    def summary(self) -> str:
        recorded_stats: dict[str, str] = {}
        for action_name, profile in self.profiled_actions.items():
            if self.dump_stats:
                self._dump_stats(action_name, profile)
            stream: io.StringIO = io.StringIO()
            stats: pstats.Stats = pstats.Stats(profile, stream=stream).strip_dirs()
            stats.sort_stats("cumulative").print_stats(self.line_count_restriction)
            recorded_stats[action_name] = stream.getvalue()
        return self._stats_to_str(recorded_stats)

    @override
    def teardown(self, stage: str | None) -> None:
        super().teardown(stage=stage)
        self.profiled_actions.clear()
        self._active_actions.clear()

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            type(self),
            (
                self.dirpath,
                self.filename,
                self.line_count_restriction,
                self.dump_stats
            )
        )
