# syntheticmind.callbacks.checkpointing: Checkpoint Policy

The checkpoint callback family translates training lifecycle events into durable checkpoint policy without owning checkpoint serialization. `ModelCheckpoint` coordinates the public callback contract, the private schedule owns trigger and resume coordinates, ranking owns scalar top-k state, and paths own filename, version, containment, link, and retention decisions. The Trainer checkpoint connector still owns full-trajectory assembly, Strategy owns topology-specific model and optimizer representation, and `CheckpointIO` owns transport.

| Module | Contents |
|---|---|
| `callback.py` | The checkpoint callback category used for ordering and extension. |
| `configuration.py` | Strict immutable public checkpoint configuration and cross-field trigger invariants. |
| `model_checkpoint.py` | Public lifecycle orchestration and canonical Trainer save requests. |
| `schedule.py` | Step, epoch, time, deferred-validation, and resumed trigger state. |
| `ranking.py` | Scalar top-k qualification, ranking, and strict persisted score state. |
| `paths.py` | Filename interpolation, versioning, containment, links, and retention boundaries. |
| `transaction.py` | Save-state mutation, rollback, last-artifact behavior, persistence metrics, and post-commit pruning. |
