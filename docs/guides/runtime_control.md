# Runtime Control

These opt-in controls extend existing ML owners. They do not add another Trainer,
change the five package-root contracts, or make every backend interchangeable.
The [runtime probes](runtime_verification.md) state what has actually been tested.

## Ownership and imports

| Capability | Public selection | Runtime owner |
|---|---|---|
| Compiled tensor regions | `TorchCompiler` and `CompilationConfiguration` from `syntheticmind.strategies` | Selected Strategy |
| Float8 training | `TorchAOFloat8Precision` from `syntheticmind.plugins.precision.torchao` | Precision lifecycle |
| Slurm tasks | `SLURMEnvironment` from `syntheticmind.plugins` | External scheduler facts; ParallelStrategy creates groups |
| Elastic restart facts | `TorchElasticEnvironment` from `syntheticmind.plugins` | External launcher; checkpoint connector selects continuation |
| Asynchronous sharded persistence | `ModelParallelStrategy(checkpoint_write="asynchronous")` | Model-parallel checkpoint owner |
| Activation recomputation | `ActivationCheckpointing` from `syntheticmind.core.activation_checkpointing` | Client-selected model region |
| CUDA failure diagnostics | `CUDAMemorySnapshot` from `syntheticmind.callbacks.cuda_memory_snapshot` | Fit callback |
| Run provenance | `RunManifestCallback` and `RunManifest` from `syntheticmind.callbacks.run_manifest` | Fit observation callback |
| Extended mesh axes | `ModelParallelStrategy(context_parallel_size=..., pipeline_parallel_size=...)` | Topology only; computation remains client-owned |

TorchAO is deliberately absent from ordinary package re-exports. Importing
SyntheticMind without TorchAO installed still works. Its adapter is imported
explicitly only by clients selecting that numerical path.

## Compiler selection and lifecycle

Construct `TorchCompiler(backend="inductor", mode="default",
graph_break_policy="allow", shape_policy="auto")`, attach it through
`strategy.attach_compiler(compiler)`, then pass that Strategy to Trainer.
There is no `Trainer(compiler=...)` argument or separate compiler plugin family.

`module_paths` selects distinct, non-overlapping submodules. The default is the
root forward; an explicit tuple such as `("encoder", "decoder")` selects regional
compilation. Unknown paths, aliased instances, overlapping regions, and existing
`nn.Module.compile` bindings fail rather than double-owning execution.

Strategy places the model, binds compiled forwards, and then applies DDP wrapping
when selected. The same module and Parameter identities remain intact; compiler
wrappers do not add `_orig_mod` checkpoint keys. Calls still enter the module's
inherited PyTorch hook path. Training steps, logging, optimizer policy, and DDP's
temporary step redirection are not themselves the compiled tensor region.

The backend modes are `default`, `reduce-overhead`, `max-autotune`, and
`max-autotune-no-cudagraphs`. Shape policy maps `auto` to PyTorch's automatic
specialization, `dynamic` to dynamic shapes, and `static` to static shapes.
`graph_break_policy="forbid"` requests a full graph for each selected forward;
`"allow"` permits PyTorch graph breaks, not silent recovery from backend errors.
Compilation is lazy: binding is not evidence that capture or backend execution
succeeded. Actual execution exceptions propagate normally.

Teardown restores the owned forward bindings without resetting process-wide
compiler caches. A client replacement of an active binding is reported instead
of overwritten. Compiler policy is immutable, survives sequential stages, and
cannot be replaced while its bindings are active.

Checkpoint schema 2 gains the optional `compiler_configuration` extension,
containing version-1 `CompilationConfiguration` or null. A legacy checkpoint
without the field represents eager execution. Fit rejects changed compiler
policy before restoration; evaluation may deliberately select a different
execution policy. No compiled callable or cache is loaded from a checkpoint.
There are no built-in graph-break counters, recompile reports, AOT export, cache
provenance, or compilation-latency measurements in this implementation.

## TorchAO numerical policy

Use `Trainer(accelerator="cuda", plugins=[TorchAOFloat8Precision(recipe="rowwise")])`.
The supported recipe names are `tensorwise`, `rowwise`, and `rowwise_with_gw_hp`.
Do not combine the plugin with a separate Trainer precision argument. The
resolved identity is `torchao-float8`; it is not a supported string constructor
shortcut for loading the optional package.

Parameters and floating inputs use BF16 storage. Eligible Linear operations use
TorchAO FP8 computation; this is not blanket conversion of all tensors to FP8.
The adapter requires NVIDIA CUDA, compute capability at least 8.9, and native
Linear input/output widths divisible by 16. It validates these conditions before
changing dtype. Custom Linear subclasses retain their semantics.

For TP or FSDP2, the client must convert native regions through the configured
precision plugin before applying its sharding plan in `configure_model`.
Already-sharded native Linear weights are rejected. ModelParallelStrategy admits
the policy, but that admission is not hardware validation of every composition.
The version-1 precision state stores the recipe and rejects a changed recipe on
load. Reconstruct the same conversion policy for continuation.

The CPU probe imports TorchAO 0.18 and tests configuration, state, and rejection
without model mutation. CUDA FP8 numerics, compiled FP8, and FSDP2 performance
remain unverified. Missing vendor CUDA-extension messages on a CPU runner are
not evidence that those kernels executed successfully.

## Scheduler environments and restart

An explicitly supplied ClusterEnvironment wins. Otherwise ParallelStrategy checks
TorchElastic identity first, standard `LOCAL_RANK` launch state second, Slurm task
state third, and finally the local fallback. This avoids launching another
worker group inside torchrun or srun.

Slurm reads `SLURM_PROCID`, `SLURM_LOCALID`, `SLURM_NODEID`, and `SLURM_NTASKS`.
The task count must equal configured devices times nodes. `MASTER_ADDR` and
`MASTER_PORT` take precedence; otherwise the first hostname is resolved with a
bounded `scontrol show hostnames` call and the port derives from the Slurm job ID.
An explicit port is appropriate when several independent groups share one job.
SyntheticMind does not submit jobs, provision nodes, or requeue Slurm jobs.

TorchElastic owns process restart. SyntheticMind validates its rank, rendezvous,
run ID, restart count, and restart budget. On a fit with restart count greater
than zero, an omitted checkpoint selects `last`; if no durable last artifact
exists, fit raises instead of starting over. Supply a ModelCheckpoint with an
explicit stable `dirpath` and `save_last=True` when constructing Trainer, or pass
an explicit checkpoint path. Model-provided callbacks and unresolved default
artifact directories are not a substitute for a known restart location.

Exact elastic continuation requires the checkpoint world size. Partial-epoch
continuation also rejects changed world size even without partial metric windows.
DCP's ability to reshard tensors does not make changed sample order, effective
batch size, optimizer trajectory, and rank-local state exactly reproducible.
Topology-changing fine-tuning is a new trajectory, not this resume contract.
The probes test launcher facts and the missing-checkpoint guard; they do not
claim a real multi-node worker-loss/restart campaign.

## Asynchronous sharded checkpointing

Select `checkpoint_write="asynchronous"` on ModelParallelStrategy with sharded
output enabled. The default remains synchronous. Full-format asynchronous output
is rejected, and an ordinary `AsyncCheckpointIO` plugin does not replace this
Strategy-owned distributed representation.

A save first prepares an immutable generation and writes Trainer metadata. The
caller then uses PyTorch's blocking async stager to capture an independent CPU
snapshot before training can mutate its source tensors. One owned worker calls
DCP `save` on a dedicated Gloo process group, agrees on failures, publishes the
active-generation pointer, and prunes old generations. Training collectives stay
on their own group. This is an asynchronous transaction around DCP's synchronous
write, not an invocation of DCP `async_save` with shared training collectives.

Only one save may be outstanding. Reads, subsequent saves, removal, last-link
publication, and teardown wait for the publication future. The previous active
generation remains readable until the new pointer commits. Rank-zero retention
waits locally while every rank's background worker participates; no extra
rank-zero-only collective is introduced. An upload/publication failure remains
observable and cannot promote the failed generation.

A save returning means staging and submission completed, not durable publication.
The ordinary Trainer barrier does not change that meaning. CPU staging and
metadata I/O still block; retention and `save_last="link"` can force a wait and
reduce overlap. Available host memory must accommodate the snapshot. Background
group operations have a bounded timeout; arbitrary node loss is not transparent
recovery. Exception cleanup drains the checkpoint owner before training groups
are destroyed. Remote asynchronous storage and CUDA/NCCL combinations are not
covered by the current CPU tests.

## Activation memory and CUDA failure records

Wrap a selected client region in `ActivationCheckpointing(region)` when
constructing the model. Its non-reentrant forward retains PyTorch hook semantics;
no-grad calls bypass recomputation. `saved_operations` accepts an explicit tuple
of PyTorch operator overloads for selective checkpointing. RNG defaults to
`preserve`; `advance` is an explicit eager trade-off and is rejected under
compilation because that execution path preserves RNG. Reserved checkpoint
keyword names cannot silently consume model inputs.

The wrapper is part of the model graph and changes its child state-dict paths.
Reconstruct the same architecture for checkpoint loading. It does not choose
layers automatically, change batch size, retry OOMs, or implement a compiled
activation-memory budget.

`CUDAMemorySnapshot(dirpath=Path(...))` is an opt-in fit callback. It requires a
CUDA Strategy, bounds allocator-history entries, and writes a UUID-, rank-, and
step-qualified local snapshot on OOM by default. `failure_policy="any_failure"`
records other failures too. Teardown disables its recorder. Use one recorder per
process: PyTorch allocator history is process-global, including external users.
Snapshots may include source paths and stacks; they are not uploaded by this
callback. The adapter uses version-dependent PyTorch allocator diagnostics;
actual GPU recording is not covered by CPU rejection tests.

## Run provenance

Add `RunManifestCallback(dirpath=Path(...), repository_root=Path(...),
data_identity="dataset-version")` to Trainer callbacks. Rank zero creates a
separate UUIDv7 directory for each fit attempt and atomically publishes JSON.
The record includes available Git identity/dirty state, framework dependency
versions, platform, device/Strategy/precision, compiler policy, run ceilings,
accumulation, and elastic identity. Data identity is supplied explicitly rather
than inferred by scanning private datasets. Environment dumps, source diffs,
credentials, and dataset contents are not included.

The current manifest schema is version 2. Unknown versions are rejected; no
migration of the earlier unreviewed feature-branch schema is provided. Complete
resolved Trainer configuration is added when a checkpoint is assembled during
the active fit. It remains null before that event or when checkpointing is
absent. The manifest ID enters that checkpoint, allowing the next attempt to
record its parent. An assembly hook is not a commit notification: neither a
manifest nor a referenced attempt proves that a checkpoint was published.

The record complements existing hyperparameter logging and checkpoints. It does
not independently bundle client code, model construction, dataset bytes, or all
application settings. The latest rank-zero destination is `manifest_path`.

## Extended CP/PP mesh boundary

The original ModelParallelStrategy positional arguments retain their meaning.
New mesh sizes and checkpoint-write mode are keyword-only. With CP=PP=1 the
original two-dimensional `data_parallel`/`tensor_parallel` mesh is preserved.
An extended request requires explicit DP and TP sizes; all four dimensions must
multiply to world size. Sampling and throughput still count the DP axis only.

An extended mesh is topology, not an implementation of distributed attention or
a pipeline schedule. Automatic optimization is rejected. The client must supply
manual step semantics, correct losses, communication, gradient synchronization,
microbatch scheduling, and checkpoint-compatible model construction. Native
context-parallel attention, GPipe/1F1B scheduling, and their end-to-end trajectory
guarantees have not been implemented by this change.

## Verification and remaining scope

The CPU workflow at commit `960eeef4b485865519b948f15290ef32a37e58c4`
[passed its executable probes](https://github.com/rahul1-bot/syntheticmind/actions/runs/34275739096)
on Python 3.14.7, PyTorch 2.14.0+cpu, Pydantic 2.13.5, and TorchAO 0.18.0.
Full logs contain single-process, both-rank distributed, and boundary PASS records.
That includes real CPU Inductor, unchanged parameter/checkpoint identity, fit
resume equality, selective activation gradients, manifest publication, DDP/TP,
asynchronous snapshot isolation, training-group concurrency, rank-zero retention,
and injected publication failure preserving the previous generation.

The workflow records exact dependencies and uses runtime execution rather than
linting or static type checking. GPU FP8/FSDP2, CUDA memory recording, remote
async storage, real multi-node restart, native CP/PP computation, and compiler
telemetry/AOT remain outside that evidence. Review those boundaries before using
new controls in consequential accelerator runs.
