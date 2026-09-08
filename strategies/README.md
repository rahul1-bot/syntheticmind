# syntheticmind.strategies: Training Execution Policies

`Strategy` is the complete policy governing how a model participates in training, validation, testing, and prediction. It owns the model connection, wrapped-model identity, accelerator and precision collaboration, optimizer stepping, step dispatch, collective defaults, initialization context, and ordinary checkpoint behavior.

`ParallelStrategy` is the shared distributed execution policy. It owns parallel devices, cluster ranks, process-group initialization, tensor reduction, distributed collectives, gradient synchronization, process launch, and process-scoped cleanup. DDP supplies its own deferred-backward synchronization context for gradient accumulation; loops and managed optimizers request that policy without importing or inspecting the DDP wrapper. Mean, sum, minimum, and maximum metric reductions execute through the selected Strategy rather than ambient process-global helpers.

`data_parallel_size` names the replicated sample-processing axis independently from total world size. DDP uses the full world, while model parallelism reports only the data-parallel mesh dimension so throughput and sample accounting never multiply by tensor-parallel workers.

`ModelParallelStrategy` exposes the named data-parallel and tensor-parallel mesh dimensions to `Module.configure_model()`. Research modules apply PyTorch tensor-parallel plans, composable FSDP2 `fully_shard` operations, or both at that boundary; the Strategy owns process topology, materialization, precision compatibility, and distributed checkpoint state without inventing separate wrapper strategies for every composition. Legacy FSDP1 wrappers are rejected explicitly.

```text
Strategy
├── SingleDeviceStrategy
└── ParallelStrategy
    ├── DDPStrategy
    └── ModelParallelStrategy
```

| Module | ML ownership |
|---|---|
| `strategy.py` | Universal training execution semantics and private wrapped-step forwarding. |
| `compiler.py` | Immutable compilation policy and reversible region-forward binding lifecycle. |
| `parallel.py` | Shared distributed process, rank, device, collective, and launcher semantics. |
| `single_device.py` | The named one-process, one-device execution policy. |
| `ddp.py` | PyTorch `DistributedDataParallel` wrapping and DDP-specific gradient behavior. |
| `model_parallel.py` | Device-mesh construction, model parallelization, and model-parallel state semantics. |
| `model_initialization.py` | Strategy-aware empty initialization, typed torch-function interception, meta-tensor materialization, and parameter reset. |
| `launchers/` | Internal process launch policy, subprocess execution, multiprocessing execution, validated worker state transfer, environment reconstruction, and failure observation. |

Files are admitted by ML or distributed-systems concepts, not by line-count pressure. Shared behavior remains on `Strategy` or `ParallelStrategy`; concrete strategy modules contain only behavior that changes the selected execution algorithm. Internal implementation mechanisms do not become public strategy concepts merely because they are reusable.

`checkpoint.py` defines the private Strategy checkpoint contract and ordinary single-file state implementation. The model-parallel checkpointing package supplies the topology-specific distributed implementation. Full-trajectory assembly, path selection, restoration order, and resume state belong exclusively to the Trainer checkpoint connector; `CheckpointIO` owns storage transport.

## Client selection

```python
from syntheticmind import Trainer

trainer: Trainer = Trainer(
    accelerator="cuda",
    strategy="ddp",
    devices=4,
    num_nodes=1
)
```

`single_device` executes one process. `ddp` re-executes an importable script or module. `ddp_spawn`, `ddp_fork`, and a configured `forkserver` strategy use torch multiprocessing. Externally launched torchrun or scheduler workers bypass local process creation and enter Trainer directly.

Local process launch supports one node. Multi-node jobs must be created by torchrun or a scheduler supplying consistent world, node, local, global-rank, and rendezvous state through `ClusterEnvironment`.

## Custom strategies

A custom Strategy is justified only when execution changes model placement or wrapping, process topology, collectives, gradient synchronization, initialization, or checkpoint representation. A new name that merely bundles existing options is not a Strategy.

Subclass `Strategy` for one-process policies and `ParallelStrategy` when the implementation genuinely shares process-group, launcher, rank, and collective semantics. Preserve `connect`, `setup_environment`, `setup`, step dispatch, backward, optimizer step, teardown, and exception behavior as one coherent lifecycle.

## Distributed collectives

Trainer, callbacks, metrics, and Module code call Strategy collectives unconditionally. Local execution provides identity behavior. Parallel execution requires every participating rank to enter matching collectives in matching order before rank-specific output returns.

`reduce_boolean_decision(all=True)` requires unanimous agreement. `all=False` accepts a request from any rank. Scalar mean reduction sums before dividing by the selected group's world size. `all_gather(sync_grads=True)` uses differentiable distributed communication; ordinary gathering does not retain a graph.

## DDP

DDP wraps one placed model replica per process. Stage-step redirection enters through the wrapper's `__call__` path so reducer and forward hooks remain active while client `training_step` still owns the research computation. Accumulation and manual optimizer contexts use `DistributedDataParallel.no_sync` through Strategy rather than inspecting the wrapper from loops.

## Model parallelism

`ModelParallelStrategy` creates named `data_parallel` and `tensor_parallel` axes. The client applies its plan in `Module.configure_model`:

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

Tensor-parallel ranks cooperate on the same samples, so distributed sampling and throughput use only the data-parallel axis. Full and sharded checkpoint paths require all-rank participation.

## Compiler and extended topology

`TorchCompiler` and its immutable `CompilationConfiguration` are exported from
`syntheticmind.strategies`. Attach a compiler through `strategy.attach_compiler`
before passing that Strategy to Trainer. Selected tensor forwards compile after
placement and before DDP wrapping; model, parameter, and state-dict identities
remain unchanged. Stage teardown restores bindings, and checkpoint continuation
validates compiler policy. There is no alternate Trainer compilation entrypoint.

The default model-parallel mesh remains two-dimensional DP/TP. Keyword-only
`context_parallel_size` and `pipeline_parallel_size` extend topology when either
exceeds one, require explicit DP/TP dimensions, and reject automatic optimization.
The client owns context-parallel attention, gradients, and pipeline schedules;
mesh creation alone does not implement those algorithms.

`checkpoint_write="asynchronous"` selects the Strategy-owned sharded transaction;
full-format async output is rejected. Slurm and TorchElastic environments are
auto-selected at the external-launch boundary without changing process ownership.
See [runtime control](../docs/guides/runtime_control.md) for construction, resume,
compatibility, and verification limits.

## Related components

[`launchers/README.md`](launchers/README.md) documents process ownership. [`model_parallel_checkpointing/README.md`](model_parallel_checkpointing/README.md) documents distributed checkpoint publication. `../accelerators/` owns backend devices and `../plugins/precision/` owns numerical execution.
