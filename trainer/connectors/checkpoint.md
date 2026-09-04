# Checkpoint Connector Ownership

The Trainer checkpoint connector owns checkpoint selection, canonical payload assembly, complete training-state restoration, random-stream continuation, loader continuation, and the ordering between hooks and live-state mutation.

The selected Strategy owns how model and optimizer state is represented under its execution topology. `CheckpointIO` owns storage transport. `ModelCheckpoint` owns retention policy and save cadence. Module, DataModule, callbacks, precision, and loops contribute state through their existing lifecycle contracts.

SyntheticMind writes one canonical private checkpoint schema. Historical alias keys and duplicated singular or plural state representations are not emitted or restored.
