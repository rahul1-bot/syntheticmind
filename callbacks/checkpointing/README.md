# syntheticmind.callbacks.checkpointing: Checkpoint Policy

The checkpoint callback family translates training lifecycle events into durable checkpoint policy without owning checkpoint serialization. `ModelCheckpoint` coordinates the public callback contract, the private schedule owns trigger and resume coordinates, ranking owns scalar top-k state, and paths own filename, version, containment, link, and retention decisions. The Trainer checkpoint connector still owns full-trajectory assembly, Strategy owns topology-specific model and optimizer representation, and `CheckpointIO` owns transport.

Batch triggers record intent at the public callback hook and persist only after optimizer, hook, metric, validation, and step-scheduler cadence is durable. Triggers from a terminal batch or validation boundary carry forward through epoch metrics and epoch schedulers before monitoring callbacks persist them. A resumable direct save inside an incomplete automatic batch fails explicitly; a weights-only snapshot remains available because it carries no continuation claim.

| Module | Contents |
|---|---|
| `callback.py` | The checkpoint callback category used for ordering and extension. |
| `configuration.py` | Strict immutable public checkpoint configuration and cross-field trigger invariants. |
| `model_checkpoint.py` | Public lifecycle orchestration and canonical Trainer save requests. |
| `schedule.py` | Step, epoch, time, deferred-validation, and resumed trigger state. |
| `ranking.py` | Scalar top-k qualification, ranking, and strict persisted score state. |
| `paths.py` | Filename interpolation, versioning, containment, links, and retention boundaries. |
| `transaction.py` | Save-state mutation, rollback, last-artifact behavior, persistence metrics, and post-commit pruning. |

## Configuring retention

```python
from syntheticmind import Trainer
from syntheticmind.callbacks.checkpointing import ModelCheckpoint


checkpoint: ModelCheckpoint = ModelCheckpoint(
    dirpath="artifacts/checkpoints",
    filename="{epoch}-{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=3,
    save_last="link",
    every_n_epochs=1
)

trainer: Trainer = Trainer(callbacks=[checkpoint], max_epochs=100)
```

The monitor must be a scalar in `trainer.callback_metrics` when a trigger is
resolved. Client models normally produce it with `self.log("val_loss", value,
on_epoch=True)`. A step or wall-time trigger that fires before validation has
created the monitored value is carried to validation; it is not silently saved
as an unmonitored checkpoint.

`save_top_k=-1` retains every due artifact. `save_top_k=0` disables ranked
artifacts. Without a monitor, `save_top_k=1` means latest-only and values other
than `-1`, `0`, or `1` are invalid because no ordering exists. `save_last=True`
writes an independent latest artifact. `save_last="link"` creates a local alias
to a ranked artifact when possible and cannot target remote storage.

Only one periodic trigger family may be active per callback: optimizer steps,
completed epochs, or elapsed training time. When no trigger is specified, the
callback saves once per epoch. Several checkpoint policies may coexist when
their callback state keys differ.

## Durability and continuation

A full training checkpoint is emitted only at a durable trajectory boundary.
The artifact includes the model, optimizers, schedulers, precision state,
callbacks, DataModule, progress, dataloader continuation, metric windows, and
random streams. A direct full save from inside an incomplete automatic
optimization batch is rejected. `save_weights_only=True` is permitted there
because the artifact makes no continuation claim.

The callback does not serialize tensors itself. It asks the Trainer checkpoint
connector to assemble state, Strategy to represent topology-specific model and
optimizer state, and CheckpointIO to persist the resulting artifact. Every rank
must participate when the active checkpoint format uses collectives.

On a failed write, schedule, ranking, and last-path state roll back to the
pre-save snapshot. A superseded artifact is removed only after its replacement
has committed. The active resume checkpoint is never pruned as part of that
transaction.

## Artifact selection

After fitting, `best_model_path`, `best_model_score`, `best_k_models`, and
`last_model_path` expose the retained policy state. Stage entry points accept
those concrete paths directly, while Trainer also resolves the selectors
`"best"` and `"last"` against configured checkpoint callbacks.

```python
trainer.test(ckpt_path="best")
trainer.predict(ckpt_path=checkpoint.last_model_path)
```

Changing `dirpath` between checkpoint creation and restoration resets ranking
paths because their artifact identities belong to the old destination.
Scheduling progress remains resumable. `to_yaml` writes the current path-to-score
ranking for external experiment inspection.

## Exception artifacts

`save_on_exception=True` attempts a diagnostic artifact only during fit, outside
fast development, and only when the current trajectory is durable. Collective
checkpoint formats are skipped because other ranks may already be unwinding.
Failure of the diagnostic save is reported without replacing the original
training exception.
