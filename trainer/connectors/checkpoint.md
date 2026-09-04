# Checkpoint Connector Ownership

The Trainer checkpoint connector owns checkpoint selection, canonical payload assembly, strict schema validation, complete training-state restoration, random-stream continuation, loader continuation, configuration compatibility, and the ordering between hooks and live-state mutation.

The selected Strategy owns how model and optimizer state is represented under its execution topology. `CheckpointIO` owns storage transport. `ModelCheckpoint` owns retention policy and save cadence. Module, DataModule, callbacks, precision, and loops contribute state through their existing lifecycle contracts.

SyntheticMind writes canonical private schema version 2. Every artifact records whether it is a resumable training checkpoint or a weights-only model artifact. Training checkpoints include the resolved Trainer configuration and immutable Python, NumPy, torch CPU, and active CUDA or MPS random state. Mid-epoch continuation retains batch progress, epoch-start state, and strict rank-local scalar or TorchMetrics accumulator state for stateful and ordinary dataloaders. Weights-only artifacts omit optimizer, scheduler, callback, precision, DataModule, loop, metric-continuation, and random state and are rejected as fit-continuation sources.

Run ceilings may increase when continuing training. Configuration that changes the trajectory—accumulation, training or validation limits, validation cadence, clipping, determinism, sampler policy, synchronization, reload cadence, or precision—must remain compatible and is validated before model or optimizer mutation.
