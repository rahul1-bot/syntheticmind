# Extension Points

SyntheticMind exposes several extension families because they own different ML
or systems concepts. Choosing the wrong family may produce working Python while
silently creating two owners for one lifecycle. That is architectural failure,
not harmless style variance.

## Choose by semantic ownership

| Need | Extend | Do not use |
|---|---|---|
| Change model computation, losses, steps, or optimizer declaration | `Module` | Callback or Trainer subclass |
| Change dataset preparation, loaders, transfer, or data state | `DataModule` | Strategy or callback |
| Observe or influence lifecycle events | `Callback` | Hidden Module side channel |
| Persist completed experiment scalars and hyperparameters | `Logger` | Callback writing metric buffers directly |
| Measure named runtime actions | `Profiler` | Timing callback or ad hoc loop instrumentation |
| Add a device backend | `Accelerator` Protocol | Strategy subclass |
| Change process topology, collectives, wrapping, or checkpoint representation | `Strategy` or `ParallelStrategy` | Accelerator or plugin resolver |
| Change dtype, autocast, backward, scaling, or numerical optimizer behavior | `Precision` | Module-wide manual casts |
| Change checkpoint byte transport | `CheckpointIO` Protocol | ModelCheckpoint subclass |
| Supply externally managed ranks and rendezvous | `ClusterEnvironment` Protocol | Process launcher replacement |
| Apply a reversible distributed layer transformation | `LayerSync` Protocol | Permanent model mutation in Trainer |
| Add checkpoint cadence or retention policy | `Checkpoint` callback | CheckpointIO |
| Support a custom transferable batch container | `DeviceTransferable` Protocol | Framework batch wrapper |

If the requirement spans several rows, compose existing owners first. A new
public abstraction is justified only when no existing concept can represent the
state, algorithm, resource, lifecycle, and failure boundary coherently.

## Regular inheritance contracts

Module, DataModule, Callback, Logger, Profiler, Strategy, ParallelStrategy, and
Precision are stateful regular classes. Their bases provide real lifecycle
behavior. Subclasses reuse that behavior and override only selected semantic
transitions.

Use `@override` from `typing` on every genuine replacement. Do not mark a new
method merely because a Protocol-shaped idea exists elsewhere. Constructors use
`@override` only when the concrete class intentionally replaces a base
constructor contract.

Python 3.14 evaluates annotations lazily. Use direct semantic imports and quoted
forward references where needed. Do not introduce `TYPE_CHECKING` branches as a
circular-import escape hatch.

## Structural contracts

Accelerator, CheckpointIO, ClusterEnvironment, LayerSync, and
DeviceTransferable are Protocols because implementations may come from external
systems and share no appropriate framework state hierarchy. Concrete
implementations explicitly inherit the Protocol and mark implemented contract
methods with `@override`.

A Protocol must remain narrow and behavior-free. It may describe the operations
one consumer requires; it may not become a second home for resolution,
configuration, lifecycle state, or reusable algorithms.

## Module

Subclass Module when the research system changes model behavior. The primary
surface includes:

1. Ordinary PyTorch `forward`.
2. `training_step` and optional validation, test, or prediction steps.
3. `configure_optimizers` for native optimizers and scheduler declarations.
4. `configure_model` for Strategy-aware construction or parallelization.
5. Metric production through `log` and `log_dict`.
6. Optional model, data-transfer, optimization, checkpoint, and callback hooks.

Do not place a free training function beside a Module and then bypass its step
contract. Do not wrap Module inside another class merely to regain `__call__`;
Module already has PyTorch call semantics and hooks.

Use `configure_model` only when construction depends on Strategy initialization,
device mesh, empty initialization, precision-directed dtype, tensor parallelism,
or composable FSDP2. Ordinary layers remain in `__init__`.

## DataModule

Subclass DataModule when the data lifecycle has real ownership beyond a direct
loader. `prepare_data` is global-zero single-copy work. `setup` is process-local.
Stage loader methods construct the sources. Transfer hooks override Module
transfer hooks when both exist. `state_dict` and `load_state_dict` preserve only
client-owned data state.

Do not persist live iterators, worker processes, locks, open files, or Trainer
references. Trainer owns loader cursor coordination, distributed sampler
injection, reload cadence, and restoration order.

## Callback

Subclass Callback for an orthogonal lifecycle observer or policy. Override the
narrowest hook family required by the concept. The Trainer and Module are passed
to every hook; retaining them across runs creates stale ownership.

```python
from typing import override

import torch

from syntheticmind import Module, Trainer
from syntheticmind.callbacks.callback import Callback


class GradientMagnitudeMonitor(Callback):
    @override
    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        module: Module,
        optimizer: torch.optim.Optimizer
    ) -> None:
        del trainer, optimizer
        squared_norm: torch.Tensor = torch.zeros((), device=module.device)
        parameter: torch.nn.Parameter
        for parameter in module.parameters():
            if parameter.grad is not None:
                squared_norm = squared_norm + parameter.grad.detach().square().sum()
        module.log("gradient_norm", squared_norm.sqrt(), on_step=True)
```

Stateful callbacks implement a strict persistence pair and a unique state key.
Callback state should contain the minimum decision trajectory, not duplicated
model, optimizer, logger, or dataloader state.

Monitoring callbacks that depend on final epoch metrics belong to the recognized
EarlyStopping or Checkpoint categories, whose loop position occurs after Module
epoch-end metric publication. A generic callback does not receive that placement
automatically.

## Logger

Subclass Logger to persist finalized Python scalar observations and portable
hyperparameters. Call the base constructor and reuse `_validated_metrics` and
`_normalize_hyperparameters`. Implement `finalize` idempotently.

Logger does not reduce distributed values, inspect Module metric state, choose
logging cadence, or decide rank ownership. The Trainer logger connector performs
those operations before invoking the backend.

See `../../loggers/README.md` for a complete custom implementation pattern.

## Profiler

Subclass Profiler to measure the named action scopes produced by Trainer,
HookDispatcher, loops, and connectors. Implement `start` and `stop`; implement
`summary` when the backend has a textual product. Reuse inherited setup, report
output, context, and teardown unless the technique owns additional resources.

A profiler must define how it handles duplicate, nested, overlapping, and
out-of-order actions. Backend cleanup must be deterministic and cannot replace a
causal client exception. Active handles and runtime sessions do not participate
in multiprocessing reconstruction.

See `../../profilers/README.md` and `../../profilers/pytorch/README.md`.

## Accelerator

Implement Accelerator only for a new device family. It owns class-level
availability and topology discovery, device selection parsing, device creation,
process-local binding, and backend statistics.

It does not choose distributed algorithms, create process groups, wrap models,
select precision, inject samplers, or launch workers. Those are Strategy and
Trainer connector responsibilities.

The instance is supplied through `Trainer(accelerator=accelerator)`. Its parsed
device identities must be stable, explicit, and validated before execution.

## Strategy

Subclass Strategy when execution semantics change model placement or wrapping,
step dispatch, optimizer behavior, initialization, collectives, or checkpoint
representation. Subclass ParallelStrategy only when the implementation genuinely
shares process-group, rank, launcher, collective, and distributed cleanup
behavior.

A new option bundle is not a new Strategy. If all behavior already exists on an
established Strategy, configure that Strategy instead of adding a renamed shell.

Preserve the lifecycle as one coherent system: connection, environment setup,
model setup, step dispatch, backward, optimizer step, exception handling,
checkpoint collaboration, and teardown. Every collective path must preserve
matching participation order across ranks.

## Precision

Subclass Precision when numerical execution changes. Override only transitions
whose meaning differs: initialization context, module or input conversion,
forward context, backward preparation, optimizer stepping, clipping, or
checkpoint state.

`connect` may replace the execution model, optimizers, or schedulers, but the
returned identities must remain coherent with Strategy and checkpoint ownership.
Mixed-precision gradients must be unscaled before client inspection or clipping.

Supply the instance through `Trainer(plugins=[precision])`; do not create a
parallel Trainer precision argument with conflicting meaning.

## CheckpointIO

Implement CheckpointIO for serialization transport or a storage service. It
receives a complete checkpoint mapping and owns save, load, removal,
storage-specific options, resources, and teardown.

It does not assemble participant state, decide whether a checkpoint is durable,
select `"best"` or `"last"`, rank monitored artifacts, or represent sharded
model state. ModelParallelStrategy rejects an external CheckpointIO when its
collective representation must remain Strategy-owned.

## ClusterEnvironment

Implement ClusterEnvironment to expose topology already owned by torchrun, a
scheduler, or another launcher. Provide validated world, node, local, and global
ranks plus rendezvous state and external-process ownership.

Returning external process ownership prevents SyntheticMind from launching
another local worker set. The environment does not create the process group or
perform collectives; ParallelStrategy owns those operations.

## LayerSync

Implement LayerSync for one reversible model transformation surrounding
distributed fit. The transformation must preserve model semantics and return a
Module. `revert` must restore the ordinary client model after Strategy teardown,
including parameters, buffers, modes, children, and relevant configuration.

Permanent architecture changes belong to Module construction, not LayerSync.

## Checkpoint callback

Subclass Checkpoint when retention or save cadence itself changes. Use Trainer's
canonical save operation so full-trajectory assembly, durability checks,
Strategy representation, and CheckpointIO transport remain intact.

A callback must not call `torch.save` over a hand-built subset and label the
result resumable. If the need is only a different transport, implement
CheckpointIO. If the need is topology-specific state, implement it with Strategy.

## DeviceTransferable batches

A custom batch container can participate in recursive placement by implementing
`to(device, non_blocking=False)`. It does not need to inherit a framework base.
The method must preserve the logical batch container and return its transferred
value.

Do not add a generic framework wrapper around every project batch type. Structural
transfer exists precisely so domain objects retain their own semantics.

## Extension acceptance checklist

Before landing an extension, verify:

1. The chosen family owns the requested concept.
2. The implementation replaces only real contract methods and marks them with
   `@override` where applicable.
3. No second owner now exists for lifecycle, state, reduction, persistence,
   topology, or failure handling.
4. Configuration rejects invalid combinations before external mutation.
5. Runtime state has deterministic setup, failure, and teardown behavior.
6. Stateful behavior has a strict checkpoint or process-transfer contract.
7. Distributed operations define rank ownership and collective participation.
8. Optional dependencies remain isolated at their adapter boundary.
9. Client-facing names use established ML vocabulary and current imports.
10. Focused CPU, relevant accelerator, distributed, failure, and resume paths
    provide executable evidence proportional to the capability.

## Related documentation

`../architecture/framework_topology.md` defines ownership and admission.
`model_and_data_integration.md` covers the primary client contracts.
`trainer_configuration.md` covers runtime composition. Each semantic package
README remains authoritative for its extension family.
