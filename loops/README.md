# syntheticmind.loops: The Execution Loops

`loops/` implements the stage execution the Trainer dispatches. Loops own stage and batch cadence, metric aggregation, optimization progress, scheduler stepping, validation timing, and continuation coordinates. Trainer owns the shared DataModule-or-Module batch-transfer hook transaction so training, validation, testing, and prediction cannot drift into separate transfer orders.

| Module | Contents |
|---|---|
| `loop.py` | `Loop`, the shared stateful attachment and stop behavior for stage loops. |
| `fit_loop.py` | `FitLoop`, the outer training driver covering epoch iteration, validation scheduling, stopping conditions, and checkpoint cadence. |
| `training_epoch_loop.py` | `TrainingEpochLoop`, one training epoch composed from explicit resume, fetch, transfer, hook, optimization, metric, validation, scheduler, continuation, and rank-local partial-epoch metric state. |
| `evaluation_loop.py` | `EvaluationLoop`, validation and test execution with metric aggregation over complete dataloaders. |
| `prediction_loop.py` | `PredictionLoop`, inference execution that collects model outputs without loss computation. |
| `optimization/automatic.py` | Automatic loss normalization, closure execution, zeroing, and backward ownership. |
| `optimization/manual.py` | Manual-optimization training-step result normalization while the Module owns backward and optimizer calls. |

Batch limits (`limit_train_batches`, `limit_val_batches`, `limit_test_batches`, and `limit_predict_batches`) bound any loop explicitly, and bounded runs are visible in resolved configurations rather than implicit.

Reaching `max_steps` or another stop condition inside a dataloader epoch does not fabricate an epoch boundary. Epoch-end hooks and epoch schedulers wait until that loader epoch actually completes; resumable checkpoints retain the loader cursor plus strict per-rank scalar and TorchMetrics accumulator state so a higher step ceiling continues the same metric and optimization trajectory.

## Fit lifecycle

`FitLoop` owns the outer epoch sequence. It dispatches training start, prepares sampler epoch state, invokes `TrainingEpochLoop`, finalizes only completed epochs, advances Trainer epoch state, and dispatches training end. Monitoring callbacks run after Module epoch-end metrics and plateau scheduling have settled.

`TrainingEpochLoop` owns one logical loader epoch:

1. Restore or capture epoch-start randomness.
2. Resolve stateful cursor continuation or stateless prefix replay.
3. Fetch, precision-convert, and transfer one batch.
4. Dispatch batch-start control hooks.
5. Execute manual or automatic optimization.
6. Record loader and logical progress.
7. Dispatch batch-end hooks and metric observations.
8. Run validation, scheduler, logger, and checkpoint cadence.
9. Close a partial accumulation tail at genuine loader exhaustion.
10. Persist continuation or publish epoch metrics.

Reaching `max_steps` inside the loader pauses this lifecycle. The epoch index, metric windows, loader cursor, and schedulers remain on the current epoch. A later fit with a higher ceiling continues the same trajectory.

## Automatic optimization

Automatic optimization normalizes training-step output, divides loss by the accumulation factor, zeros at the start of each accumulation window, executes precision-aware backward, suppresses DDP synchronization on non-boundary microbatches, clips unscaled gradients, and steps the managed optimizer. `global_step` advances only after a real managed optimizer mutation.

If natural loader exhaustion leaves an incomplete accumulation window, Strategy synchronizes the tail gradients and the loop performs one final step. A `training_step` that returns None skips its batch; when every microbatch of an accumulation window is skipped, the boundary produces no gradients and the loop advances neither `global_step` nor step-interval schedulers. Distributed training rejects `training_step=None` because ranks could diverge in collective order.

## Manual optimization

Manual mode runs the client training step with gradients enabled and normalizes only its reporting output. The Module calls `manual_backward`, managed optimizer `step`, `zero_grad`, and optional `toggle_model` directly. Automatic accumulation and clipping remain disabled unless the client invokes the corresponding operations deliberately.

See [`optimization/README.md`](optimization/README.md) for the output contracts.

## Validation and test

`EvaluationLoop` serves validation and test because both reduce scalar observations under a no-gradient lifecycle. Sequential multiple loaders produce one result mapping per loader. Other composition modes produce one result over the combined batch tree.

Validation inside fit uses `no_grad`; standalone evaluation may use `inference_mode` when the backend supports it. Dataloader-index requirements are validated before the first batch. Non-monitoring validation epoch-end callbacks and the Module hook contribute final metrics before early stopping and checkpoint callbacks run.

## Prediction

`PredictionLoop` remains separate because it retains arbitrary outputs rather than reducing metrics. Retained values move to CPU after each batch. Multiple loaders return one output list per loader. Multiprocessing launch requires `return_predictions=False` because the parent does not reconstruct arbitrary worker output trees.

## Validation cadence

`ValidationSchedule` supports epoch eligibility, integer batches, cross-epoch integer batches, fractions of a sized epoch, final exhaustion of an unsized epoch, cooperative-stop validation, and monotonic wall-time intervals. Time cadence restarts after validation completes so validation duration cannot trigger an immediate second pass.

## Checkpoint durability

A checkpoint is durable only outside an incomplete batch and, under automatic optimization, after a complete optimizer boundary. Batch and validation triggers may record save intent before that point; persistence waits until model, optimizer, scheduler, metric, cursor, and RNG state describe one coherent continuation boundary.

## Related components

`../trainer/` owns outer setup and teardown. `../data/` owns loader construction and composition. `../core/module.py` exposes client step and optimization methods. `../strategies/` owns distributed synchronization.
