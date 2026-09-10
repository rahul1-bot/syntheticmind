# Training Lifecycle

SyntheticMind treats each public Trainer call as a bounded state transaction.
`fit`, `validate`, `test`, and `predict` claim stage state, resolve execution,
enter one local or launched runtime, and attempt complete teardown before the
stage is released.

## Public entry and process launch

A public stage call first resolves the Module, data source, checkpoint selector,
and model-provided callbacks. Direct dataloaders and a DataModule are mutually
exclusive. Omitting a model after an earlier call reuses the attached Module;
otherwise a stage requires an explicit Module.

Single-device execution enters the local stage transaction directly. A local
distributed Strategy delegates to its configured launcher. Each worker then
re-enters the same Trainer-owned local transaction with reconstructed process
rank, environment, stochastic state, and arguments. Externally launched workers
already have rank state and do not create another worker set.

## Fit transaction

One local fit follows this order:

1. Claim fit state, clear prior metric views and continuation state, reset loops,
   attach Module and optional DataModule, and record train and validation sources.
2. Register signal handling and initialize the Strategy environment.
3. Attach the profiler and establish its stage and rank identity.
4. Run `prepare_data` on global zero, synchronize ranks, then run DataModule,
   callback, and Module `setup` hooks in that order on every process.
5. If resuming, load and validate checkpoint metadata early enough to expose the
   restored epoch, step, and DataModule state before loader and optimizer
   construction.
6. Request training data, resolve validation cadence, bind the accelerator
   device, run `configure_model` in the precision construction context, apply
   fit-only layer synchronization, convert precision, and let Strategy place or
   wrap the model.
7. Invoke `configure_optimizers`, normalize optimizer and scheduler declarations,
   connect precision, create managed `SyntheticOptimizer` adapters, and validate
   automatic versus manual policy.
8. Restore the complete checkpoint trajectory and move restored optimizer state
   to the active device.
9. Dispatch callback then Module `on_fit_start`, log captured hyperparameters,
   and run bounded sanity validation only for a fresh non-resumed fit.
10. Enter `FitLoop`, execute eligible epochs, honor cooperative stop conditions,
    dispatch callback then Module `on_fit_end`, and check termination state.
11. Preserve any worker-owned continuation state, attempt full teardown, clear
    stage context, and reconcile launcher results to the parent.

Model construction precedes optimizer construction because distributed plans,
lazy layers, precision conversion, or model replacement may change the parameter
graph consumed by `configure_optimizers`.

## Opt-in runtime participants

A selected compiler binds tensor forwards after placement and before DDP wrapping;
its bindings are restored by Strategy teardown. Compilation is lazy and does not
move hooks, optimizer policy, or step orchestration into the captured region.
Compiler configuration is validated during fit resume before restoration.

A run manifest is written at `on_fit_start`; its complete Trainer-policy projection
is updated at checkpoint assembly, not at durable publication. CUDA allocator
history is acquired by its fit callback and released during callback teardown.

Asynchronous sharded saves capture an independent CPU snapshot on the caller,
then write and publish through an owned worker and separate communication group.
A returned save or ordinary Trainer barrier does not prove async commit. Reads,
retention, last-link creation, subsequent saves, and teardown fence publication.
On model-parallel failure, checkpoint work drains before training groups are
released. These ordering details are part of the
[runtime control contract](../guides/runtime_control.md).

## Training epoch transaction

`FitLoop` owns epoch eligibility and completion. `TrainingEpochLoop` owns the
actual loader epoch:

1. Capture or restore epoch-start random state.
2. Restore a stateful loader cursor or prepare deterministic prefix replay.
3. Fetch, precision-convert, and transfer one batch.
4. Dispatch training batch-start hooks.
5. Execute automatic or manual optimization.
6. Advance logical batch and loader continuation coordinates.
7. Dispatch batch-end hooks and commit metric observations.
8. Evaluate validation, logger, scheduler, and checkpoint cadence.
9. Synchronize and step an incomplete accumulation tail only at natural loader
   exhaustion.
10. Publish epoch metrics and advance the epoch only after the loader epoch truly
    completes.

`global_step` counts completed managed optimizer mutations. It does not count
microbatches, skipped closures, validation batches, or raw calls to
`training_step`.

## Automatic optimization

Automatic optimization requires one optimizer. The loop normalizes
`training_step` output, extracts a tensor loss, divides it by the accumulation
factor, zeros gradients at the opening of each accumulation window, performs
precision-aware backward, and defers DDP synchronization on non-boundary
microbatches.

At a real optimizer boundary, precision processing exposes gradients to
`on_before_optimizer_step`, applies configured clipping, invokes the optimizer,
advances global step, and advances eligible step schedulers. A natural partial
tail synchronizes its gradients before the final step.

Distributed `training_step` cannot return None because rank-dependent skipped
work can break collective order. Local execution may skip that batch without
advancing optimizer progress.

## Manual optimization

Set `module.automatic_optimization = False` when the model must control objective
order, optimizer order, or update frequency. `training_step` then invokes
`self.optimizers()`, `manual_backward`, managed optimizer `step` and `zero_grad`,
and optional optimizer toggling directly.

Trainer-level gradient accumulation and automatic clipping are rejected in
manual mode. The client owns those decisions explicitly, but managed operations
still pass through Strategy, precision, callback hooks, and global-step
accounting.

## Validation inside fit

Validation may run by epoch, integer batch interval, fraction of a sized loader,
cross-epoch batch interval, monotonic time interval, natural exhaustion, or a
cooperative stop request. Time cadence restarts after validation so the duration
of validation cannot immediately trigger another pass.

Fit validation uses `torch.no_grad`, pauses active training time, and restores
the model's prior modes afterward. Sanity validation is bounded, marked through
`trainer.sanity_checking`, and its metrics and progress views are cleared before
training.

At validation epoch end, ordinary callbacks run first, then the Module hook,
then its newly logged metrics are merged. EarlyStopping and Checkpoint callbacks
run only after this final metric view exists. The same split applies at training
epoch end, after which plateau scheduler and monitoring decisions see settled
state.

## Standalone validation and test

Validation and test share `EvaluationLoop` because both execute no-gradient
steps and reduce metrics. They retain distinct stage names, hook families, and
data sources.

Each stage prepares its environment and data, configures the model and precision,
optionally restores the selected checkpoint, runs one or several loaders, and
returns one `dict[str, float]` per sequential dataloader. Other CombinedLoader
modes produce one result mapping over their combined batch topology.

Standalone evaluation may use `torch.inference_mode` when configured and
backend-safe. Validation nested inside fit stays on `torch.no_grad` because it
must return to training-compatible tensor behavior.

## Prediction

Prediction has its own loop because it retains arbitrary model outputs instead
of reducing metric state. Each retained output moves to CPU after its batch.
Sequential multiple loaders return one output list per loader.

`return_predictions=False` executes all computation and hooks without retaining
the output tree. Multiprocessing launch requires this setting because the parent
does not attempt to reconstruct arbitrary client output structures from workers.

## Hook ordering

Callback start hooks generally precede the corresponding Module hook. End hooks
are scheduled by the owning loop according to their consumer semantics. In
particular, monitoring callbacks run after Module epoch-end metrics; presentation
and timing callbacks retain their earlier lifecycle positions.

HookDispatcher provides four guarantees:

1. Callback, Module, DataModule, and Strategy calls enter a stable named
   profiling range.
2. Module hooks expose the active hook name to `Module.log` validation.
3. Real overrides are discovered against their owning base contract.
4. Wrapped distributed models route stage steps through Strategy without
   bypassing wrapper forward hooks.

Callbacks must not assume every hook runs after the Module hook merely because
its name ends in `_end`. The owning loop and callback category define that order.

## Durable boundaries

A resumable checkpoint is valid only when model, optimizer, scheduler, metrics,
loader cursor, random streams, and logical progress describe one coherent
trajectory. An incomplete automatic batch is not durable. A callback may record
save intent during a batch or validation event, but persistence waits for the
loop's durability boundary.

Stopping inside an epoch does not manufacture epoch completion. The current
epoch, partial metric windows, loader state or replay coordinate, scheduler
position, and epoch-start random state remain resumable.

## Failure and teardown

The first execution failure remains causal. Callback, DataModule, and Strategy
exception hooks are attempted and secondary failures become notes. Teardown then
attempts, without early exit:

1. Model-mode restoration when execution did not complete normally.
2. Signal restoration, Strategy teardown, checkpoint transport teardown, layer
   synchronization reversal, precision teardown, and cluster teardown.
3. Owned CombinedLoader reset and persistent-worker release.
4. DataModule teardown, ordinary callback teardown, Module teardown, logger
   finalization, and progress callback teardown.
5. Profiler report generation and teardown.

If no execution failure exists, the first cleanup failure becomes causal and the
remaining failures become notes. Stage state is cleared after cleanup so a later
public call cannot inherit a false active-stage identity.

## Related documentation

`../../trainer/README.md` documents public stage calls and configuration.
`../../loops/README.md` documents iterative execution. `checkpoint_trajectory.md`
documents continuation. `distributed_execution.md` documents worker and
collective behavior.
