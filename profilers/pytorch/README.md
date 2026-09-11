# syntheticmind.profilers.pytorch: Operator Measurement

`PyTorchProfiler` owns the high-detail measurement lifecycle backed by
`torch.profiler`. It can record CPU and accelerator activities, input shapes,
memory, FLOPs, stacks, eager module ranges, operator tables, Chrome traces, and
flame-stack products without changing Trainer execution.

| Module | Contents |
|---|---|
| `profiler.py` | Public configuration, backend session, named ranges, summaries, traces, validation, and cleanup. |
| `schedule.py` | Private ML-step-aware adapter around a client torch profiler schedule. |
| `module_recording.py` | Private reversible eager-module forward instrumentation. |

## Client configuration

```python
import torch

from syntheticmind import Trainer
from syntheticmind.profilers import PyTorchProfiler


profiler: PyTorchProfiler = PyTorchProfiler(
    dirpath="artifacts/profiles",
    filename="encoder",
    activities=[torch.profiler.ProfilerActivity.CPU],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
    group_by_input_shapes=True,
    record_module_names=True,
    export_to_chrome=True
)

trainer: Trainer = Trainer(profiler=profiler)
```

Keyword options are checked against the installed `torch.profiler.profile`
signature. Known boolean, callable, timeout, activity, table, sort, and exporter
settings receive stricter semantic validation. Unsupported names fail during
construction instead of silently disappearing into a backend-version mismatch.

## Schedule semantics

Torch exposes one profiler step coordinate, but a training harness interleaves
several ML step concepts. `_ProfilerSchedule` maintains independent counts for
`training_step`, `validation_step`, `test_step`, and `predict_step`. Fit sanity
validation is excluded before the first training step, so it cannot consume a
client training schedule.

Only completed stage-step actions advance the profiler. Setup, transfer,
callback, optimization, and checkpoint actions remain named ranges inside the
current capture without advancing its schedule. Once one step family returns
`RECORD_AND_SAVE`, that family is finished for the current Trainer stage.

The schedule callable must return `ProfilerAction.NONE`, `WARMUP`, `RECORD`, or
`RECORD_AND_SAVE`. SyntheticMind prevents a recorded capture from regressing to
warmup state and losing its active evidence.

## Module ranges

With `record_module_names=True`, every eager module forward receives a range
containing its Python-qualified class and path inside the client model. Shared or
recursive modules use per-identity range stacks. Always-call forward hooks close
ranges when client code raises, and stage teardown removes every handle.

ScriptModule instrumentation is skipped because eager forward hooks do not
provide an equivalent scripted contract. Operator events remain available.

## Trace products

`export_to_chrome=True` emits indexed `.json` traces when a capture becomes
ready. `export_to_flame_graph=True` enables stack capture and exports `.stack`
data using `self_cpu_time_total` or `self_cuda_time_total`. An empty stack capture
fails explicitly because an empty flame product would pretend success.

Torch exporters require local paths. `StoragePath.local_artifact` stages a local
product and publishes it to a supported remote destination after successful
export. Distributed artifact names include the local rank.

An explicit `on_trace_ready` callable remains client-owned. SyntheticMind
installs its exporter only when a schedule exists and no client handler was
provided. Without a schedule or custom handler, the framework exports the final
stage capture during profiler closure.

## NVTX mode

`emit_nvtx=True` replaces collection with NVTX range emission for an external
CUDA timeline. Only `record_shapes` is accepted in this mode. Text summaries,
Chrome traces, flame stacks, schedule callbacks, and unrelated torch-profiler
options are rejected or absent rather than implying artifacts that NVTX mode
does not create.

## Cleanup guarantees

Teardown closes outstanding action ranges, module instrumentation, the torch
profiler session, and the NVTX context independently. If several cleanup
operations fail, the first remains causal and later failures become notes. The
shared Profiler stage state is released in a `finally` path, allowing the same
configured instance to participate in a later stage.

## Related components

`../profiler.py` defines the shared stage and action contract. `../../trainer/`
owns lifecycle setup and reporting. `../../storage/` owns remote publication of
local-only exporter artifacts.
