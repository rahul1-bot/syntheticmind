# syntheticmind.trainer.connectors: Runtime Composition Boundaries

Connectors translate Trainer configuration into connected runtime components. Each connector owns one complete vertical boundary, including validation, attachment order, conflicts, and teardown handoff.

`AcceleratorConnector` is the first reconstructed boundary. It replaces separate topology, plugin, and precision resolvers with one decision over accelerator, devices, nodes, Strategy, precision, cluster environment, checkpoint I/O, and layer synchronization.

Connectors are internal implementation concepts. They do not introduce alternate fit, validate, test, predict, checkpoint, or logging entrypoints.

| Module | Ownership |
|---|---|
| `accelerator_connector.py` | Accelerator, devices, nodes, Strategy, precision, environment, checkpoint transport, and layer compatibility. |
| `callback_connector.py` | Callback defaults, model callback composition, state identity, and checkpoint-last ordering. |
| `checkpoint_connector.py` | Path policy, schema-v2 trajectory, validation, participant restore, and exact continuation. |
| `data_connector.py` | Source precedence, loader construction, limits, distributed samplers, reload, and totals. |
| `logger_connector/` | Logger resolution, metric views, accumulation, routing, and hyperparameters. |
| `signal_connector.py` | Nested SIGTERM registration, worker propagation, chaining, and restoration. |

## Accelerator composition

Resolution order is plugins, accelerator, devices, Strategy, topology, checkpoint transport, cluster environment, layer synchronization, precision, then precision attachment. A collaborator supplied through a configured Strategy cannot also be supplied through Trainer plugins. Duplicate ownership fails before runtime.

Automatic accelerator resolution prefers CUDA, then MPS, then CPU. Distributed topology requires a distributed Strategy. MPS remains single-device. Precision aliases normalize once, float16 CPU mixed precision becomes bfloat16, and an explicit precision plugin cannot be combined with a separate precision value.

## Callback composition

Defaults are added before execution: ModelCheckpoint when checkpointing is enabled, Timer where timing or progress requires it, ProgressBarCallback when enabled, and ModelSummary when enabled. Only one progress callback is allowed. Checkpoint callbacks execute last.

`Module.configure_callbacks` runs once per Module and stage. Its callback subclasses replace compatible Trainer callback types. Callback state keys must remain unique before checkpoint restoration.

## Checkpoint trajectory

The checkpoint connector assembles model representation, effective Trainer configuration, optimizer and scheduler state, callbacks, DataModule, precision, random streams, progress, loader cursor, and rank-local partial metric state. It validates the entire schema before live state mutation.

Restoration order is Strategy model and optimizer state, schedulers, random streams, callbacks, DataModule, precision, and logical trajectory coordinates. Weights-only artifacts are rejected for fit continuation.

## Data composition

Source precedence is direct Trainer data, then DataModule, then Module hooks. Plain training collections become `max_size_cycle`; evaluation and prediction become `sequential`. An explicit CombinedLoader retains its selected mode except prediction requires sequential identity.

Sampler injection reconstructs the concrete DataLoader subclass and required custom constructor values. Training and evaluation use standard equal-work DistributedSampler behavior. Prediction uses unrepeated rank-strided indices.

## Signal lifecycle

SIGTERM handling is main-thread-only and nesting-aware. The first signal marks state and propagates to launcher-owned workers, then chains a pre-existing callable handler. Teardown restores only a handler SyntheticMind still owns. Pickle state removes process-bound handlers.

## Related components

[`logger_connector/README.md`](logger_connector/README.md) documents experiment observation. `../trainer.py` sequences connectors but does not duplicate their policy.
