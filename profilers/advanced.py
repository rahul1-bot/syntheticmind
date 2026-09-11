# This module:
# 1. Defines AdvancedProfiler, the cProfile-backed policy for Python and C call
#    attribution within named framework actions
# 2. Produces cumulative pstats reports and optional raw .prof artifacts for
#    external analysis
#
# Design decisions:
# - Named action profilers are mutually exclusive: entering a nested action
#   pauses its parent and closing it resumes the parent, preventing nested work
#   from being double-attributed
# - The active action list is a strict stack; out-of-order stop requests fail
#   immediately instead of corrupting later attribution
# - Setup creates fresh cProfile objects per Trainer stage, and teardown disables
#   every profile before shared state is released
# - Raw statistics use StoragePath.local_artifact so remote destinations receive
#   a completed local .prof product through the storage transaction boundary
# - __reduce__ reconstructs only configuration, not active profiler internals,
#   making multiprocessing launch independent of unpicklable cProfile state
#
# Author: Rahul Sawhney

import cProfile
import io
import pstats
from collections import defaultdict
from pathlib import Path
from typing import override

from syntheticmind.profilers.profiler import Profiler
from syntheticmind.storage.filesystem import StoragePath
from syntheticmind.primitives.types import TrainerStage

__all__: list[str] = ["AdvancedProfiler"]

type _AdvancedProfilerArguments = tuple[
    str | Path | None,
    str | None,
    int | float,
    bool
]


class AdvancedProfiler(Profiler):
    # Exclusive nested cProfile policy for client runs that need call attribution
    # beyond SimpleProfiler's action-level wall clock.
    @override
    def __init__(
        self,
        dirpath: str | Path | None = None,
        filename: str | None = None,
        line_count_restriction: int | float = 1.0,
        dump_stats: bool = False
    ) -> None:
        # Args:
        #     dirpath: Optional local or remote report and raw-stat directory.
        #     filename: Optional report identity.
        #     line_count_restriction: pstats row limit or fractional restriction.
        #         Default: ``1.0``.
        #     dump_stats: Whether each action also emits a raw .prof artifact.
        #         Default: ``False``.
        super().__init__(dirpath=dirpath, filename=filename)
        if isinstance(line_count_restriction, bool) or not isinstance(
            line_count_restriction,
            int | float
        ):
            raise TypeError("`line_count_restriction` must be an int or float.")
        if line_count_restriction < 0:
            raise ValueError("`line_count_restriction` must be non-negative.")
        if type(dump_stats) is not bool:
            raise TypeError(f"`dump_stats` must be a bool, got {dump_stats!r}.")
        self.profiled_actions: defaultdict[str, cProfile.Profile] = defaultdict(
            cProfile.Profile
        )
        self.line_count_restriction: int | float = line_count_restriction
        self.dump_stats: bool = dump_stats
        self._active_actions: list[str] = []

    @override
    def start(self, action_name: str) -> None:
        # Pauses the active parent, opens the requested exclusive cProfile region,
        # and pushes it onto the nesting stack.
        if action_name in self._active_actions:
            raise ValueError(f"Attempted to start {action_name!r} more than once.")
        if self._active_actions:
            active_name: str = self._active_actions[-1]
            self.profiled_actions[active_name].disable()
        self.profiled_actions[action_name].enable()
        self._active_actions.append(action_name)

    @override
    def stop(self, action_name: str) -> None:
        # Enforces stack order, closes the active region, and resumes its parent.
        if not self._active_actions:
            raise ValueError(
                f"Attempted to stop action {action_name!r} before it was started."
            )
        active_name: str = self._active_actions[-1]
        if active_name != action_name:
            raise ValueError(
                f"Attempted to stop action {action_name!r} while "
                f"{active_name!r} is the active nested action."
            )
        self.profiled_actions[action_name].disable()
        self._active_actions.pop()
        if self._active_actions:
            parent_name: str = self._active_actions[-1]
            self.profiled_actions[parent_name].enable()

    @override
    def setup(
        self,
        stage: TrainerStage,
        local_rank: int | None = None,
        log_dir: str | Path | None = None
    ) -> None:
        # Claims the stage and discards all prior call statistics and nesting.
        super().setup(stage, local_rank, log_dir)
        self.profiled_actions: defaultdict[str, cProfile.Profile] = defaultdict(
            cProfile.Profile
        )
        self._active_actions: list[str] = []

    @override
    def teardown(self, stage: TrainerStage | None) -> None:
        # Disables every backend profile before shared stage teardown.
        profile: cProfile.Profile
        for profile in self.profiled_actions.values():
            profile.disable()
        self._active_actions: list[str] = []
        super().teardown(stage)

    @override
    def summary(self) -> str:
        # Refuses incomplete action scopes, optionally exports raw data, and
        # formats one cumulative-cost pstats table per action.
        if self._active_actions:
            raise RuntimeError(
                f"Cannot summarize active profiler actions: {self._active_actions}."
            )
        recorded_statistics: dict[str, str] = {}
        action_name: str
        profile: cProfile.Profile
        for action_name, profile in self.profiled_actions.items():
            if self.dump_stats:
                self._dump_statistics(action_name, profile)
            stream: io.StringIO = io.StringIO()
            statistics: pstats.Stats = pstats.Stats(
                profile,
                stream=stream
            ).strip_dirs()
            statistics.sort_stats("cumulative").print_stats(
                self.line_count_restriction
            )
            recorded_statistics[action_name] = stream.getvalue()
        return self._format_statistics(recorded_statistics)

    def _dump_statistics(
        self,
        action_name: str,
        profile: cProfile.Profile
    ) -> None:
        # Publishes one action's raw profile through a local or staged remote
        # artifact path.
        if self.dirpath is None:
            raise RuntimeError(
                "AdvancedProfiler requires a directory to dump raw statistics."
            )
        destination: str = StoragePath(self.dirpath).join(
            self._prepare_filename(action_name, ".prof")
        )
        with StoragePath(destination).local_artifact() as local_path:
            profile.dump_stats(str(local_path))

    @override
    def __reduce__(
        self
    ) -> tuple[type[AdvancedProfiler], _AdvancedProfilerArguments]:
        # Reconstructs a clean configured profiler inside launched workers.
        arguments: _AdvancedProfilerArguments = (
            self.dirpath,
            self.filename,
            self.line_count_restriction,
            self.dump_stats
        )
        return type(self), arguments
