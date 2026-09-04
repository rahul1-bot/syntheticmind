# Distributed Execution

SyntheticMind separates device identity, process topology, model execution,
numerical policy, and client parallelization plans. Distributed execution is not
one boolean and world size is not automatically the number of independent
samples processed at once.

## Component ownership

| Concept | Owner |
|---|---|
| Device discovery, parsing, binding, and statistics | Accelerator |
| Process ranks, rendezvous, and externally created workers | ClusterEnvironment |
| Process creation and parent reconciliation | ProcessLauncher |
| Process group, collectives, model placement or wrapping, and gradient synchronization | ParallelStrategy |
| Floating-point conversion and optimizer numerical behavior | Precision plugin |
| Distributed sampler injection | Trainer data connector using Strategy coordinates |
| Client tensor-parallel or composable FSDP2 plan | `Module.configure_model` |
| Distributed checkpoint representation | ModelParallelStrategy checkpoint owner |

The Trainer accelerator connector resolves these collaborators as one
compatibility decision before execution begins.

## Strategy hierarchy

`Strategy` owns universal model connection, wrapped-step dispatch, initialization
context, precision-mediated backward and optimizer stepping, local collective
identity behavior, and ordinary checkpoint collaboration.

`ParallelStrategy` adds process-group lifecycle, ranks, devices, launch policy,
distributed collectives, sampler coordinates, tail-gradient synchronization,
and worker cleanup. `DDPStrategy` and `ModelParallelStrategy` inherit that real
behavior and implement only algorithm-specific deltas.

| Strategy | Execution meaning |
|---|---|
| `SingleDeviceStrategy` | One process and one root device. |
| `DDPStrategy` | One complete model replica and one process per selected device. |
| `ModelParallelStrategy` | One named device mesh supporting data and tensor parallel axes plus client-applied sharding or tensor plans. |

MPS is single-device. The accelerator connector rejects distributed MPS rather
than presenting a process topology the backend does not implement.

## Launch modes

| Trainer strategy | Start method | Operational contract |
|---|---|---|
| `"ddp"` | `popen` | Re-executes the current importable module or real script for nonzero local ranks. |
| `"ddp_spawn"` | `spawn` | Starts fresh interpreters from pickled Trainer and stage state. |
| `"ddp_fork"` | `fork` | Forks the current process for interactive-compatible execution; CUDA is prohibited. |
| configured DDP | `forkserver` when selected | Uses a clean fork server and requires guarded, pickleable entry. |
| external torchrun or scheduler | no local creation | Every externally created worker enters the Trainer stage directly. |

Local launch is one-node only. Multi-node execution requires an external launcher
and a ClusterEnvironment that supplies consistent world size, node rank, local
rank, global rank, rendezvous address, and port.

Spawn and fork require all transported client state to be pickleable. Guard
script entry with the ordinary `if __name__ == "__main__":` boundary when the
start method imports or reconstructs the program.

Multiprocessing prediction requires `return_predictions=False`. Workers do not
send arbitrary model output trees to the dormant parent.

## Rank vocabulary

| Coordinate | Meaning |
|---|---|
| `global_rank` | Unique process index across all nodes. |
| `local_rank` | Process index on the current node. |
| `node_rank` | Node index in the distributed job. |
| `world_size` | Total participating process count. |
| `data_parallel_size` | Number of replicas processing distinct samples. |

For DDP, `data_parallel_size == world_size`. Under tensor parallelism, several
ranks cooperate on one sample, so `data_parallel_size` is only the size of the
named data-parallel mesh axis. ThroughputMonitor and sample accounting use this
value rather than multiplying by the entire world.

## Distributed data

The data connector reconstructs map-style DataLoaders with Strategy-aware
samplers while preserving concrete loader subclass state and required constructor
values.

Training, validation, and test use equal-work distributed sampling where each
rank must execute matching collective structure. Prediction uses an unrepeated
rank-strided sampler so the distributed result covers each source sample once
rather than padding by duplication.

`CombinedLoader` retains nested loader identity and mode. Stateful loader cursors
are stored in stable flattened order. A topology mismatch fails before cursor
mutation.

## DDP model execution

DDP places one complete model replica on each root device and wraps it in
`torch.nn.parallel.DistributedDataParallel`. Stage-step redirection enters
through the wrapper's call path, preserving reducer and forward-hook behavior
while the client Module still owns `training_step`, `validation_step`,
`test_step`, and `predict_step`.

Automatic gradient accumulation enters `DistributedDataParallel.no_sync` on
non-boundary microbatches. If natural loader exhaustion leaves a partial
accumulation tail, Strategy synchronizes the tail before the final optimizer
step. Manual optimizer contexts use the same Strategy-owned synchronization
boundary.

## Model-parallel execution

ModelParallelStrategy creates named `data_parallel` and `tensor_parallel` mesh
dimensions before `Module.configure_model` runs. Client code applies the actual
research plan there:

```python
from typing import override

from torch import nn
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module

from syntheticmind import Module


class ParallelEncoder(Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection: nn.Linear = nn.Linear(1024, 4096)

    @override
    def configure_model(self) -> None:
        parallelize_module(
            self,
            self.device_mesh["tensor_parallel"],
            {"projection": ColwiseParallel()}
        )
```

Composable FSDP2 `fully_shard` may be applied at the same boundary. Legacy FSDP1
wrappers are rejected. The Strategy materializes distributed meta state, places
remaining tensors, reconnects checkpoint ownership, and synchronizes ranks
after the client plan.

Configure mesh dimensions explicitly when their factorization matters:

```python
from syntheticmind import Trainer
from syntheticmind.strategies.model_parallel import ModelParallelStrategy


strategy: ModelParallelStrategy = ModelParallelStrategy(
    data_parallel_size=2,
    tensor_parallel_size=4,
    save_distributed_checkpoint=True
)

trainer: Trainer = Trainer(
    accelerator="cuda",
    devices=8,
    strategy=strategy
)
```

The product of explicit data and tensor sizes must match world size. `"auto"`
allows Strategy to infer the remaining dimension. Model-parallel execution
requires `configure_model`; a model with no parallelization plan fails instead
of running replicated while claiming model parallelism.

## Collectives

Framework and client code call the selected Strategy for collectives. Local
execution returns identity behavior. Distributed execution requires every rank
in the selected group to enter matching collectives in matching order.

`reduce` supports mean, sum, minimum, and maximum. Mean sums before dividing by
the selected group's size. `reduce_boolean_decision(all=True)` requires unanimous
agreement; `all=False` accepts a request from any rank. Callback stop and trigger
policies use these operations to prevent rank divergence.

`Module.all_gather` accepts supported tensors, NumPy arrays, scalars, and nested
collections. `sync_grads=True` selects differentiable gathering; ordinary
gathering does not retain an autograd graph.

Never place a collective behind a rank-zero-only branch unless the operation
itself is explicitly rank-zero-only and no peers must participate. Logging output
may be rank-zero-owned only after required reductions have completed.

## Worker-to-parent state

Spawn and fork workers own live execution. Local rank zero returns a validated
worker payload containing the model, Trainer state, callbacks, DataModule,
precision, metrics, data totals, loader continuation, optimizer and scheduler
state, signals, and stochastic streams. The parent restores that payload instead
of pretending its dormant objects executed training.

CPU forked parameters and buffers are cloned before mutation to prevent shared
storage from changing the parent prematurely. Metric tensors detach and move to
CPU before process transfer.

## Failure and signal behavior

The subprocess observer watches nonzero workers while rank zero may be blocked
inside a collective. The first worker failure terminates siblings and reports the
causal exit to the parent. SIGTERM preserves SIGTERM semantics; unrelated
failures may escalate to kill where the platform supports it.

Signal handlers, environment variables, torch thread count, process observers,
worker handles, process groups, and temporary transport state are restored or
cleared during failure teardown. A successful stage must not leave an owned
worker or process group in a false active state.

## Distributed checkpointing

Ordinary DDP checkpoints gather Strategy state and publish through the configured
CheckpointIO. Model-parallel full and sharded representations require all-rank
participation and remain Strategy-owned. Sharded publication uses immutable
generations and an atomic active pointer; see `checkpoint_trajectory.md`.

Exception checkpoints skip collective formats because peers may already be
unwinding. A single surviving rank cannot safely impersonate an all-rank save.

## Related documentation

`../../strategies/README.md` documents execution policies.
`../../strategies/launchers/README.md` documents process ownership.
`../../accelerators/README.md` documents devices. `../../plugins/README.md`
documents environment, precision, checkpoint, and layer extensions.
