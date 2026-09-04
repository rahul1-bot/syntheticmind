# Checkpoint Trajectory

A SyntheticMind checkpoint is either a resumable training trajectory or a model
weights artifact. Those are different products. A weights file can initialize or
evaluate a model; it cannot claim to continue optimizer, data, metric, scheduler,
callback, precision, or stochastic state.

## Ownership

Checkpoint behavior is deliberately split by concept:

| Concern | Owner |
|---|---|
| Stage path selection and schema validation | Trainer checkpoint connector |
| Full trajectory assembly and participant restore order | Trainer checkpoint connector |
| Model and optimizer representation | Active Strategy checkpoint owner |
| Full versus sharded model-parallel representation | ModelParallelStrategy checkpoint owner |
| Serialization transport and storage options | CheckpointIO plugin |
| Periodic cadence, monitoring, top-k, last, and retention | ModelCheckpoint callback |
| Filesystem identity and staging | StoragePath |

ModelCheckpoint never assembles optimizer state. CheckpointIO never decides
which artifact ranks highest. Strategy never chooses periodic cadence. This
separation keeps retention, representation, transport, and continuation from
becoming one irrecoverable blob.

## Schema version 2

Every artifact includes:

| Field | Meaning |
|---|---|
| `schema_version` | Exact SyntheticMind schema identifier, currently `2`. |
| `epoch` | Current epoch for partial continuation or next epoch after a completed boundary. |
| `global_step` | Count of completed managed optimizer mutations. |
| `state_dict` | Strategy-produced model representation. |
| `trainer_configuration` | Effective trajectory-critical Trainer policy. |
| `metadata` | Artifact type, torch version, Strategy name, and world size. |
| model hyperparameters | Captured constructor metadata when the Module participates in hyperparameter capture. |

A training checkpoint additionally carries, when applicable:

1. Current Python, NumPy, torch CPU, and active accelerator random streams.
2. Optimizer and learning-rate scheduler states in configured order.
3. Callback states keyed by collision-free callback identity.
4. DataModule hyperparameters and state keyed by concrete type.
5. Precision-plugin state keyed by concrete type.
6. Completed batch and global batch coordinates for a partial epoch.
7. Epoch-start random state required for deterministic prefix replay.
8. Ordered stateful CombinedLoader cursor state.
9. One strict partial-epoch metric state mapping per checkpoint rank.

A weights-only artifact omits all continuation fields and records
`metadata.checkpoint_type="weights"`.

## Save transaction

`Trainer.save_checkpoint` and ModelCheckpoint both enter the Trainer's canonical
assembly path. A full save requires an attached Module and a durable optimization
boundary. The connector collects current loop continuation, rank-local metric
windows, loader state, random state, Strategy model and optimizer state,
schedulers, callbacks, DataModule, precision, and effective configuration.

Precision and callback `on_save_checkpoint` hooks run before the Module hook.
These hooks may add namespaced client state but must not delete or reinterpret
framework trajectory fields.

Strategy then persists the representation through its checkpoint owner and
CheckpointIO. When the caller requests synchronization, every participating rank
barriers only after publication succeeds.

## Callback retention transaction

ModelCheckpoint divides policy into four private state machines:

1. Schedule owns step, epoch, time, validation deferral, and resume coordinates.
2. Ranking owns scalar top-k qualification and retained path scores.
3. Paths own naming, containment, collision versions, last aliases, and safe
   removal eligibility.
4. Transaction owns joint mutation, persistence, rollback, and post-commit
   pruning.

Logical schedule and ranking state advance before serialization so the artifact
contains the state created by its own save. If persistence fails, all logical
state rolls back. A superseded artifact is removed only after its replacement
commits. The active resume checkpoint is protected from callback pruning.

Monitored values must be scalar and are broadcast from global zero. A trigger
that fires before its monitor exists defers until validation supplies it.
Rejected top-k candidates may still create a fresh last artifact; `save_last`
never points at an older best artifact while claiming to represent the latest
training state.

## Path selection

Every stage accepts a concrete checkpoint path, `"best"`, `"last"`, or None.
An explicit stage argument overrides mutable `trainer.ckpt_path` with a warning.
When validation, test, or prediction omits a model after an earlier fit, None
defaults to the previous fit's best artifact.

`"best"` resolves the first configured ModelCheckpoint best path. `"last"`
considers valid last artifacts and selects the newest available candidate.
Unsupported selectors such as `"hpc"` and `"registry"` fail unless their real
integrations are present.

Path selection validates the artifact through the active Strategy checkpoint
owner before loading. A plain file is not automatically a valid sharded
checkpoint directory, and a directory is not automatically a valid full
checkpoint.

## Restoration order

The complete schema is validated before live participant mutation. Fit also
validates effective configuration compatibility and rejects weights-only
continuation.

Restoration then follows this order:

1. Module, callback, and precision load hooks inspect the validated payload.
2. Strategy restores model representation and, during fit, optimizer state.
3. Scheduler states restore in current configured order.
4. Current random streams restore.
5. Callback state restores from isolated deep copies.
6. DataModule state restores when it was not already required for early loader
   construction.
7. Precision-plugin state restores.
8. Trainer and loop epoch, global step, batch, loader, metric, and epoch-start
   random coordinates apply.

Optimizer objects and schedulers are constructed before restoration. This is why
the checkpoint metadata is preloaded early but the complete trajectory is not
applied until model and optimization configuration exist.

## Configuration compatibility

The artifact stores resolved policy, not merely raw constructor arguments. A fit
may increase its terminal epoch or step ceiling, but trajectory-changing drift is
rejected. Examples include incompatible accumulation, stage limits, validation
cadence, clipping, determinism, sampler policy, reload cadence, precision, or
barebones behavior.

Callback identities and scheduler cardinality must match. Partial-epoch metric
state requires the same world size because each rank owns a distinct accumulator
window. Stateful loader topology must match stable flattened order before any
cursor is mutated.

## Exact partial-epoch continuation

Stopping at `max_steps` inside a loader does not run epoch-end hooks or epoch
schedulers. The checkpoint retains the current epoch rather than incrementing it.

For stateful loaders, the framework restores each owned cursor. For ordinary
loaders, it restores the epoch-start random state, reconstructs the epoch, replays
the consumed prefix without applying model work, then restores the current random
state before the next unconsumed batch. Partial scalar and TorchMetrics windows
resume from their rank-local state so final epoch metrics match uninterrupted
execution.

Graph-retaining scalar metrics and custom process-group reductions cannot cross
a resumable partial-epoch boundary because their live execution state is not a
portable checkpoint representation.

## Model reconstruction

`Module.load_from_checkpoint` is a model reconstruction API, not a Trainer
resume operation. It loads model hyperparameters, applies explicit argument
precedence, constructs the subclass, invokes `configure_model`, lets the Module
inspect the checkpoint, loads model state under resolved strictness, and places
the result on the mapped device.

It does not recreate optimizer, scheduler, callback, DataModule, loop, or random
trajectory state. Use `Trainer.fit(..., ckpt_path=...)` for continuation.

## Full and sharded model-parallel publication

Full model-parallel checkpoints use one atomic artifact path and require
all-rank participation for representation. Sharded checkpoints use an immutable
generation directory plus a versioned `active_generation` pointer. Publication
of that pointer is the visibility boundary.

The pointer retains the active and immediately previous generation identifiers.
An incomplete generation never becomes readable as current. Failed shard,
metadata, or pointer publication preserves the previously active generation;
later successful publication may prune abandoned generation debris.

Local pointer publication uses same-directory temporary replacement and file and
directory synchronization. Supported remote publication uses the filesystem
transaction boundary. Cross-format path collisions fail without deleting the
existing artifact.

## Failure policy

Schema, metadata, participant shape, and configuration failures occur before the
corresponding live state is mutated. Callback load mappings are deep-copied so
one callback cannot corrupt the payload observed by another.

`save_on_exception=True` is best-effort. It runs only for a durable fit state and
skips checkpoint formats requiring all-rank collectives because peer processes
may already be unwinding. A diagnostic-save failure is logged without replacing
the original training exception.

## Related documentation

`../../trainer/connectors/README.md` documents connector ownership.
`../../callbacks/checkpointing/README.md` documents retention policy.
`../../strategies/model_parallel_checkpointing/README.md` documents generation
publication. `../../plugins/README.md` documents CheckpointIO substitution.
