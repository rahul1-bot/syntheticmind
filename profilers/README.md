# syntheticmind.profilers: Runtime Measurement

Profilers observe named Trainer, loop, hook, data, optimization, and persistence actions without owning those operations. `Profiler` owns active-stage state, shared setup, scoped action lifetime, report output, and deterministic teardown. Each concrete profiler owns one measurement technique.

| Module | Contents |
|---|---|
| `profiler.py` | Shared profiler lifecycle and the disabled pass-through implementation. |
| `simple.py` | Low-overhead wall-clock duration aggregation by action. |
| `advanced.py` | Python and C call attribution through `cProfile`. |
| `pytorch/` | Operator, device, module, scheduled-step, Chrome trace, and flame-graph profiling through `torch.profiler`. |

XLA profiling is intentionally absent until SyntheticMind owns an executable XLA accelerator and strategy path. A backend-specific profiler without a backend lifecycle is decorative compatibility baggage, not a usable training capability.

## Selecting a profiler

Trainer accepts the names `"simple"`, `"advanced"`, and `"pytorch"`, a configured
Profiler instance, or None for `PassThroughProfiler`.

```python
from syntheticmind import Trainer
from syntheticmind.profilers import AdvancedProfiler


profiler: AdvancedProfiler = AdvancedProfiler(
    dirpath="artifacts/profiles",
    filename="operator_ablation",
    line_count_restriction=0.2,
    dump_stats=True
)

trainer: Trainer = Trainer(profiler=profiler)
```

String selection uses backend defaults. Supply an instance when artifact paths,
report detail, scheduling, activities, or exporter behavior belongs to the
experiment design.

The same profiler instance may serve sequential `fit`, `validate`, `test`, and
`predict` calls. It cannot own concurrent stages. Trainer attaches the active
Module, resolves the report directory and local rank, calls `setup`, measures
named operations through `profile`, then calls `describe`. `describe` emits the
summary and releases stage state even when the summary is empty.

## Action semantics

Trainer, HookDispatcher, loops, and data or checkpoint connectors define the
action names. Examples include model and callback hooks, dataloader setup,
training epochs and batches, stage steps, and checkpoint persistence. Profilers
observe those operations; they do not alter execution order, reduction,
optimization, or storage ownership.

`Profiler.profile` preserves client failures. If measurement cleanup also fails,
the cleanup failure is attached to the causal exception as a note. Stage
teardown attempts profiler reporting alongside every other resource cleanup.

## Reports and rank identity

With both a filename and resolved directory, text reports use stage-, filename-,
and local-rank-qualified names. Supported remote paths append through
`StoragePath`. Without a file destination, reports enter loguru only on local
rank zero. This produces one console report per node while preserving per-rank
files for distributed diagnosis.

`SimpleProfiler` measures elapsed action time with minimal overhead. Its extended
report includes mean, call count, total, and percentage of end-to-end stage wall
time. Because action intervals may overlap, percentages are diagnostic shares,
not a partition guaranteed to total one hundred percent.

`AdvancedProfiler` uses mutually exclusive nested cProfile regions. Entering a
child pauses its parent; action exits must follow stack order. Optional `.prof`
artifacts are suitable for external pstats-compatible analysis.

[`pytorch/README.md`](pytorch/README.md) covers operator activities, ML-aware
schedules, module ranges, Chrome traces, flame stacks, and NVTX mode.

## Custom profiler integration

A custom profiler replaces `start` and `stop` and may replace `summary`. It uses
the inherited setup, output, action-context, and teardown contract.

```python
import time
from pathlib import Path
from typing import override

from syntheticmind.profilers import Profiler


class LatestActionProfiler(Profiler):
    @override
    def __init__(
        self,
        dirpath: str | Path | None = None,
        filename: str | None = None
    ) -> None:
        super().__init__(dirpath=dirpath, filename=filename)
        self.started_at: dict[str, float] = {}
        self.latest_seconds: dict[str, float] = {}

    @override
    def start(self, action_name: str) -> None:
        if action_name in self.started_at:
            raise ValueError(f"Action {action_name!r} is already active.")
        self.started_at[action_name] = time.perf_counter()

    @override
    def stop(self, action_name: str) -> None:
        started_at: float = self.started_at.pop(action_name)
        self.latest_seconds[action_name] = time.perf_counter() - started_at

    @override
    def summary(self) -> str:
        return "\n".join(
            f"{name}: {seconds:.6f}s"
            for name, seconds in self.latest_seconds.items()
        )
```

If a backend cannot represent overlapping or nested actions, it must reject them
explicitly. Runtime handles, live module references, and active measurement
state must not cross multiprocessing reconstruction or persist beyond teardown.

## Related components

`../trainer/hook_dispatcher.py` profiles client hook invocation.
`../trainer/trainer.py` owns profiler selection and lifecycle.
`../loops/` defines iterative action boundaries. `../storage/` stages local-only
profile products for supported remote destinations.
