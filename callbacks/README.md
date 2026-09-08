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
| `progress_bar.py` | `ProgressBarCallback`, which renders the run header, live stage progress, epoch and stage records, and failure coordinates. |
| `model_summary/` | The `ModelSummary` callback, layer and model inspection report, and optional Rich renderer. |
| `throughput_monitor.py` | `ThroughputMonitor`, which records synchronized window and epoch throughput over the Strategy's data-parallel sample axis. |
| `device_stats_monitor.py` | `DeviceStatsMonitor`, which records device utilization. |

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

## Progress presentation

`ProgressBarCallback` renders on global rank zero. Its constructor accepts
`refresh_interval` for interactive rewrites, `redirected_interval` for the
permanent heartbeat written when stderr is not a terminal (None disables it),
`max_metrics` for the number of progress-routed metrics on live, epoch, and
stage lines, and `capture_diagnostics`, which routes loguru records and Python
warnings through the renderer while a stage renders so they never splice into
the live line.

A stage begins with a header naming the accelerator, devices, nodes, world
size, Strategy, precision, seed, epoch and step ceilings, accumulation, resume
checkpoint, and logger directory. The interactive line shows a spinner, epoch,
glyph bar, percentage, epoch ETA, throughput, on-step metrics, and the first
optimizer's learning rate. Each completed epoch writes epoch-reduced training
metrics, validation metrics, duration, and a remaining-fit estimate derived from
completed epochs. A failure record carries the epoch, global step, and active
batch. Unknown loader horizons render counts and throughput only. Styling is
disabled when `NO_COLOR` is set or stderr is redirected.

## Related Components

Callbacks are composed by `../trainer/connectors/callback_connector.py`, invoked by the profiler-aware `../trainer/hook_dispatcher.py`, scheduled by Trainer and its loops, and persisted by the Trainer checkpoint connector.
