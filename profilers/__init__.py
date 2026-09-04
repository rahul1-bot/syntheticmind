from syntheticmind.profilers.advanced import AdvancedProfiler
from syntheticmind.profilers.base import PassThroughProfiler
from syntheticmind.profilers.profiler import Profiler
from syntheticmind.profilers.pytorch import PyTorchProfiler, RegisterRecordFunction
from syntheticmind.profilers.simple import SimpleProfiler
from syntheticmind.profilers.xla import XLAProfiler
from syntheticmind.exceptions import MisconfigurationError

__all__: list[str] = [
    "AdvancedProfiler",
    "PassThroughProfiler",
    "Profiler",
    "PyTorchProfiler",
    "RegisterRecordFunction",
    "SimpleProfiler",
    "XLAProfiler"
]


def _resolve_profiler(profiler: Profiler | str | None) -> Profiler:
    if profiler is None:
        return PassThroughProfiler()
    if isinstance(profiler, Profiler):
        return profiler
    if not isinstance(profiler, str):
        raise TypeError(
            "`profiler` must be a Profiler instance, a supported string, or None."
        )
    profiler_classes: dict[str, type[Profiler]] = {
        "simple": SimpleProfiler,
        "advanced": AdvancedProfiler,
        "pytorch": PyTorchProfiler,
        "xla": XLAProfiler
    }
    normalized_name: str = profiler.lower()
    profiler_class: type[Profiler] | None = profiler_classes.get(normalized_name)
    if profiler_class is None:
        raise MisconfigurationError(
            "When passing a string for `profiler`, expected one of "
            f"{list(profiler_classes)}, got {profiler!r}."
        )
    return profiler_class()
