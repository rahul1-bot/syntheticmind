# syntheticmind.loops: The Execution Loops

`loops/` implements the stage execution the trainer dispatches to. The loops define hook order, batch transfer, metric aggregation, optimizer and scheduler stepping, and validation cadence, so these semantics are identical for every model the framework runs.

| Module | Contents |
|---|---|
| `loop.py` | `Loop`, the shared stateful attachment and stop behavior for stage loops. |
| `fit_loop.py` | `FitLoop`, the outer training driver covering epoch iteration, validation scheduling, stopping conditions, and checkpoint cadence. |
| `training_epoch_loop.py` | `TrainingEpochLoop`, one training epoch covering batch iteration, forward and backward execution under automatic or manual optimization, gradient accumulation and clipping, optimizer and scheduler stepping, and step-level logging. |
| `evaluation_loop.py` | `EvaluationLoop`, validation and test execution with metric aggregation over complete dataloaders. |
| `prediction_loop.py` | `PredictionLoop`, inference execution that collects model outputs without loss computation. |
| `optimization/automatic.py` | Automatic loss normalization, closure execution, zeroing, and backward ownership. |
| `optimization/manual.py` | Manual-optimization training-step result normalization while the Module owns backward and optimizer calls. |

Batch limits (`limit_train_batches`, `limit_val_batches`, `limit_test_batches`, and `limit_predict_batches`) bound any loop explicitly, and bounded runs are visible in resolved configurations rather than implicit.

## Related Components

The loops are dispatched by `../trainer/trainer.py`, transfer batches through the data hook boundary, and aggregate metrics through the current metric owner.
