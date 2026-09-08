# Checkpoint Connector Ownership

The Trainer checkpoint connector owns checkpoint selection, canonical payload assembly, strict schema validation, complete training-state restoration, random-stream continuation, loader continuation, configuration compatibility, and the ordering between hooks and live-state mutation.

The selected Strategy owns how model and optimizer state is represented under its execution topology. `CheckpointIO` owns storage transport. `ModelCheckpoint` owns retention policy and save cadence. Module, DataModule, callbacks, precision, and loops contribute state through their existing lifecycle contracts.

SyntheticMind writes canonical private schema version 2. Every artifact records whether it is a resumable training checkpoint or a weights-only model artifact. Training checkpoints include the resolved Trainer configuration and immutable Python, NumPy, torch CPU, and active CUDA or MPS random state. Mid-epoch continuation retains batch progress, epoch-start state, and strict rank-local scalar or TorchMetrics accumulator state for stateful and ordinary dataloaders. Weights-only artifacts omit optimizer, scheduler, callback, precision, DataModule, loop, metric-continuation, and random state and are rejected as fit-continuation sources.

Run ceilings may increase when continuing training. Configuration that changes the trajectory—accumulation, training or validation limits, validation cadence, clipping, determinism, sampler policy, synchronization, reload cadence, or precision—must remain compatible and is validated before model or optimizer mutation.

The optional `compiler_configuration` extension contains a versioned immutable
policy; an absent field in a legacy schema-v2 checkpoint means eager execution.
Fit rejects compiler-policy drift before restoration. On TorchElastic restart,
missing explicit state resolves to a durable `last` artifact or fails. Exact
restart requires unchanged world size; partial-epoch consumed batches also reject
a changed world independently of metric state.

Asynchronous sharded publication remains Strategy-owned. A returned save is not
a commit notification; read, retention, last-link, subsequent-save, and teardown
boundaries fence publication. Run-manifest assembly records are provenance only.
See [runtime control](../../docs/guides/runtime_control.md).
