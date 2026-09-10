# syntheticmind.callbacks: The Lifecycle Callbacks

`callbacks/` extends the trainer through a uniform hook interface. Callbacks observe and act on a run without living inside model or loop code, and callback state participates in checkpoints, so resumable runs restore callback progress alongside model state.

| Module | Contents |
|---|---|
| `callback.py` | `Callback`, the hook surface every callback selectively overrides. |
| `checkpointing/` | The `ModelCheckpoint` lifecycle, trigger schedule, scalar top-k ranking, and artifact path policy. |
| `early_stopping.py` | `EarlyStopping`, which stops a run on a monitored metric with patience. |
| `timer.py` | `Timer`, which enforces resumable wall-time budgets. |
| `finite_value_guard.py` | `FiniteValueGuard`, which rejects non-finite loss before backward and optional non-finite gradients before optimizer stepping. |
| `weight_averaging.py` | `EMAWeightAveraging`, which tracks resumable exponential moving averages and evaluates under averaged parameters. |
| `lr_monitor.py` | `LearningRateMonitor`, which records per-group learning rates. |
| `progress_bar.py` | `ProgressBarCallback`, which reports stage progress. |
| `model_summary/` | The `ModelSummary` callback, layer and model inspection report, and optional Rich renderer. |
| `throughput_monitor.py` | `ThroughputMonitor`, which records synchronized window and epoch throughput over the Strategy's data-parallel sample axis. |
| `device_stats_monitor.py` | `DeviceStatsMonitor`, which records device utilization. |
| `cuda_memory_snapshot.py` | Opt-in bounded CUDA allocator history and failure snapshots. |
| `run_manifest.py` | Versioned run provenance and atomic per-attempt local JSON. |

## Client callbacks

A callback observes or modifies trainer lifecycle state without replacing model
computation. Subclass `Callback`, override only the narrow hook family owned by
the extension, and mark each genuine replacement with `@override`.

```python
from typing import override

import torch

from syntheticmind import Module, Trainer
from syntheticmind.callbacks.callback import Callback


class GradientAudit(Callback):
    @override
    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        module: Module,
        optimizer: torch.optim.Optimizer
    ) -> None:
        del optimizer
        if (trainer.global_step + 1) % 100 == 0:
            gradient_norm: float = sum(
                float(parameter.grad.detach().norm())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            trainer.print(f"gradient norm: {gradient_norm:.4f}")
```

Training computation, loss construction, optimizer selection, and batch
transfer belong to `Module`, `DataModule`, and the loops. A callback should not
become a hidden second model or trainer. If an extension needs configuration
only, constructor state is sufficient. If its decisions must continue through a
checkpoint, implement `state_dict`, `load_state_dict`, and a collision-free
`state_key`.

## Lifecycle ordering

`setup` runs after the execution environment and stage inputs have been
resolved. `teardown` runs while stage state is still available and is attempted
on both normal and exceptional exits. Start hooks precede their corresponding
work; end hooks observe completed module hooks and metric publication according
to the loop that owns the transition.

Optimization hooks have narrower meanings:

1. `on_before_backward` sees the loss before precision-managed backward.
2. `on_after_backward` sees populated gradients.
3. `on_before_optimizer_step` runs before clipping and stepping.
4. `on_before_zero_grad` runs immediately before `Module.optimizer_zero_grad`;
   automatic optimization uses it when a new accumulation window opens.

`global_step` advances only after a managed optimizer mutation. A training
batch callback may therefore run several times at one global step under gradient
accumulation. Step-cadence callbacks must suppress duplicate observations when
that distinction matters.

Validation hooks also cover sanity validation scheduled before fitting.
Callbacks that mutate patience, ranking, telemetry, or other experiment state
must explicitly exclude `trainer.sanity_checking` unless startup probes are a
deliberate part of their semantics.

## Callback state

The Trainer rejects duplicate keys among stateful callbacks. The default class
name is appropriate for one stateless or singleton instance. Several instances
of one stateful callback must derive their key from every configuration field
that changes persisted decision semantics.

Checkpoint restoration is strict. A callback should validate serialized values
instead of coercing arbitrary payloads, and it should store only the minimum
state required to continue the same trajectory. Process-local resources,
absolute monotonic timestamps, live Trainer references, and autograd graphs do
not belong in callback state.

## Built-in callback boundaries

`EarlyStopping`, `Timer`, and `ModelCheckpoint` request cooperative lifecycle
actions; they do not terminate processes directly. Monitoring callbacks publish
through the trainer logger connector and require at least one configured logger.
`EMAWeightAveraging` temporarily changes the model state for evaluation but
preserves live training weights for resumable checkpoints. `ProgressBarCallback`
owns terminal presentation only and never becomes a metrics persistence layer.

## Failure diagnostics and run provenance

Import `CUDAMemorySnapshot` and `RunManifestCallback` from their semantic modules.
The CUDA callback is fit-only, requires a CUDA Strategy, and records bounded
allocator history until teardown. OOM is the default capture policy; snapshots
are local, unique, and rank/step-qualified. Use one recorder per process because
the underlying diagnostic state is process-global.

The manifest callback creates one UUIDv7 directory per rank-zero fit attempt.
Version-2 records capture available code, dependency, execution, compiler, and
elastic identities plus an explicitly supplied data identity. Complete resolved
Trainer policy is added at an active-fit checkpoint assembly hook. The record
links checkpoint attempts, but its existence does not prove checkpoint commit.
It complements, rather than replaces, hyperparameter logging and full trajectory
checkpoints. Neither callback automatically uploads private records.

See [runtime control](../docs/guides/runtime_control.md) for constructors,
conflicts, schema semantics, and unverified GPU boundaries.

## Related Components

Callbacks are composed by `../trainer/connectors/callback_connector.py`, invoked by the profiler-aware `../trainer/hook_dispatcher.py`, scheduled by Trainer and its loops, and persisted by the Trainer checkpoint connector.
