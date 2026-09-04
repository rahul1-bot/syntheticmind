# syntheticmind.primitives: Framework Vocabulary

`primitives/` is the dependency-inward language shared by SyntheticMind subsystems. It contains type relationships and failure categories, not algorithms or runtime services.

| Module | Contents |
|---|---|
| `types.py` | Recursive aliases and structural typing for stages, batches, outputs, hyperparameters, state, checkpoints, optimization, and device transfer. |
| `exceptions.py` | The framework exception hierarchy and process-termination signal. |

## Admission boundary

A definition belongs here only when at least two independent semantic packages require it and importing it introduces no dependency on a concrete runtime implementation.

The package may contain:

1. Recursive type aliases describing values that cross subsystem boundaries.
2. Structural Protocols implemented by client or backend objects.
3. Framework exception categories with stable meanings.

The package may not contain:

1. Algorithms or orchestration.
2. Mutable runtime state.
3. Configuration resolution or implementation selection.
4. Filesystem, network, device, or process resources.
5. Concrete accelerators, strategies, callbacks, loaders, models, or plugins.

This boundary prevents `primitives/` from becoming a renamed utility package.

## Important distinctions

`Batch` describes nested values entering a model and admits custom `DeviceTransferable` containers. `ModelOutput` describes values returned from steps and prediction. `StateDict` is the string-keyed state of one participant. `CheckpointDict` is the wider serialized trajectory aggregate. `OptimizerState` retains integer parameter identifiers and is therefore not interchangeable with either mapping.

`DeviceTransferable` is structural. A client batch type participates in recursive placement by implementing `to(device, non_blocking=False)`; it does not inherit from SyntheticMind.

## Failure vocabulary

`FrameworkError` is the shared recoverable framework boundary. `MisconfigurationError` identifies invalid component combinations or user configuration. `CheckpointError` identifies storage, schema, or publication failures. `SIGTERMException` retains `SystemExit` semantics so termination is not accidentally swallowed as an ordinary training error.

## Related components

Every semantic package may depend inward on these definitions. `primitives/` must not import those packages in return.
