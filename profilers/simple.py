import math
import time
from collections import defaultdict
from operator import itemgetter
from pathlib import Path
from typing import override

from syntheticmind.profilers.profiler import Profiler
from syntheticmind.types import TrainerStage

__all__: list[str] = ["SimpleProfiler"]

type _SimpleReportRow = tuple[str, float, float]
type _ExtendedReportRow = tuple[str, float, int, float, float]


class SimpleProfiler(Profiler):
    @override
    def __init__(
        self,
        dirpath: str | Path | None = None,
        filename: str | None = None,
        extended: bool = True
    ) -> None:
        super().__init__(dirpath=dirpath, filename=filename)
        if type(extended) is not bool:
            raise TypeError(f"`extended` must be a bool, got {extended!r}.")
        self.current_actions: dict[str, float] = {}
        self.recorded_durations: defaultdict[str, list[float]] = defaultdict(list)
        self.extended: bool = extended
        self.start_time: float = time.perf_counter()

    @override
    def start(self, action_name: str) -> None:
        if action_name in self.current_actions:
            raise ValueError(f"Attempted to start {action_name!r} more than once.")
        self.current_actions[action_name] = time.perf_counter()

    @override
    def stop(self, action_name: str) -> None:
        end_time: float = time.perf_counter()
        if action_name not in self.current_actions:
            raise ValueError(
                f"Attempted to stop action {action_name!r} before it was started."
            )
        start_time: float = self.current_actions.pop(action_name)
        self.recorded_durations[action_name].append(end_time - start_time)

    @override
    def setup(
        self,
        stage: TrainerStage,
        local_rank: int | None = None,
        log_dir: str | Path | None = None
    ) -> None:
        super().setup(stage, local_rank, log_dir)
        self.current_actions: dict[str, float] = {}
        self.recorded_durations: defaultdict[str, list[float]] = defaultdict(list)
        self.start_time: float = time.perf_counter()

    @override
    def summary(self) -> str:
        prefix: str = f"{self._stage.upper()} " if self._stage is not None else ""
        lines: list[str] = [f"{prefix}Profiler Report"]
        if not self.recorded_durations:
            return "\n".join(lines)
        name_width: int = max(6, max(map(len, self.recorded_durations)))
        if self.extended:
            lines.append(
                f"{'Action':<{name_width}}  {'Mean (s)':>12}  {'Calls':>8}  "
                f"{'Total (s)':>12}  {'Percent':>9}"
            )
            report_result: tuple[list[_ExtendedReportRow], int, float] = (
                self._make_extended_report()
            )
            report: list[_ExtendedReportRow] = report_result[0]
            total_calls: int = report_result[1]
            total_duration: float = report_result[2]
            lines.append(
                f"{'Total':<{name_width}}  {'-':>12}  {total_calls:>8}  "
                f"{total_duration:>12.5f}  {'100.000':>9}"
            )
            lines.extend(
                f"{name:<{name_width}}  {mean:>12.5f}  {calls:>8}  "
                f"{total:>12.5f}  {percentage:>9.3f}"
                for name, mean, calls, total, percentage in report
            )
        else:
            lines.append(
                f"{'Action':<{name_width}}  {'Mean (s)':>12}  {'Total (s)':>12}"
            )
            lines.extend(
                f"{name:<{name_width}}  {mean:>12.5f}  {total:>12.5f}"
                for name, mean, total in self._make_report()
            )
        return "\n".join(lines)

    def _make_extended_report(
        self
    ) -> tuple[list[_ExtendedReportRow], int, float]:
        total_duration: float = max(
            time.perf_counter() - self.start_time,
            1.0e-12
        )
        report: list[_ExtendedReportRow] = []
        action_name: str
        durations: list[float]
        for action_name, durations in self.recorded_durations.items():
            duration_sum: float = math.fsum(durations)
            call_count: int = len(durations)
            report.append(
                (
                    action_name,
                    duration_sum / call_count,
                    call_count,
                    duration_sum,
                    duration_sum / total_duration * 100.0
                )
            )
        report.sort(key=itemgetter(4), reverse=True)
        total_calls: int = sum(row[2] for row in report)
        return report, total_calls, total_duration

    def _make_report(self) -> list[_SimpleReportRow]:
        report: list[_SimpleReportRow] = []
        action_name: str
        durations: list[float]
        for action_name, durations in self.recorded_durations.items():
            duration_sum: float = math.fsum(durations)
            report.append(
                (action_name, duration_sum / len(durations), duration_sum)
            )
        report.sort(key=itemgetter(1), reverse=True)
        return report
