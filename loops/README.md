# syntheticmind.loops: The Execution Loops

`loops/` implements the stage execution the Trainer dispatches. Loops own stage and batch cadence, metric aggregation, optimization progress, scheduler stepping, validation timing, and continuation coordinates. Trainer owns the shared DataModule-or-Module batch-transfer hook transaction so training, validation, testing, and prediction cannot drift into separate transfer orders.

| Module | Contents |
|---|---|
| `loop.py` | `Loop`, the shared stateful attachment and stop behavior for stage loops. |
| `fit_loop.py` | `FitLoop`, the outer training driver covering epoch iteration, validation scheduling, stopping conditions, and checkpoint cadence. |
| `training_epoch_loop.py` | `TrainingEpochLoop`, one training epoch composed from explicit resume, fetch, transfer, hook, optimization, metric, validation, scheduler, and continuation transitions. |
| `evaluation_loop.py` | `EvaluationLoop`, validation and test execution with metric aggregation over complete dataloaders. |
| `prediction_loop.py` | `PredictionLoop`, inference execution that collects model outputs without loss computation. |
| `optimization/automatic.py` | Automatic loss normalization, closure execution, zeroing, and backward ownership. |
| `optimization/manual.py` | Manual-optimization training-step result normalization while the Module owns backward and optimizer calls. |

Batch limits (`limit_train_batches`, `limit_val_batches`, `limit_test_batches`, and `limit_predict_batches`) bound any loop explicitly, and bounded runs are visible in resolved configurations rather than implicit.

## Related Components

The loops are dispatched by `../trainer/trainer.py`, transfer batches through the data hook boundary, and aggregate metrics through the current metric owner.
