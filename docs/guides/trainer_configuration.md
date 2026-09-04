# Trainer Configuration

Trainer configuration is executable policy. It determines which devices and
processes exist, how a batch becomes an optimizer mutation, when validation
runs, which state can resume, and which observability systems are active.

Configuration is validated before client data, model execution, process groups,
or artifact mutation. Resolved policy is then stored in training checkpoints so
continuation cannot silently change trajectory-critical semantics.

## Configuration domains

| Domain | Parameters |
|---|---|
| Execution | `accelerator`, `strategy`, `devices`, `num_nodes`, `precision`, `plugins`, `sync_batchnorm` |
| Run limits | `max_epochs`, `min_epochs`, `max_steps`, `min_steps`, `max_time` |
| Data | stage batch limits, `overfit_batches`, `use_distributed_sampler`, validation cadence, sanity validation, loader reload cadence |
| Optimization | `accumulate_grad_batches`, gradient clipping, deterministic algorithms, benchmarking, anomaly detection |
| Observation | `logger`, callbacks, logger cadence, progress, model summary, profiler, hyperparameter autologging |
| Infrastructure | `default_root_dir`, stochastic `seed` |

Trainer is keyword-only. This makes a run configuration reviewable and prevents a
new parameter from changing the meaning of positional client code.

## Accelerator and device selection

`accelerator="auto"` chooses an available CUDA backend, then MPS, then CPU.
Explicit values are `"cpu"`, `"cuda"`, `"mps"`, and the compatibility selector
`"gpu"`, or a structural Accelerator instance.

GPU `devices` accepts a positive count, explicit integer list, numeric or
comma-separated identifier string, `-1` for all visible devices, or `"auto"`.
Duplicate, negative, empty, or unavailable identifiers fail before Strategy
construction. CPU integers represent worker-process count because CPU devices
have no indexed accelerator slots.

```python
from syntheticmind import Trainer


cpu_trainer: Trainer = Trainer(
    accelerator="cpu",
    devices=1,
    precision="32-true"
)

mps_trainer: Trainer = Trainer(
    accelerator="mps",
    devices=1,
    precision="16-mixed"
)

ddp_trainer: Trainer = Trainer(
    accelerator="cuda",
    strategy="ddp",
    devices=[0, 1, 2, 3],
    precision="bf16-mixed"
)
```

Multiple selected devices require a distributed Strategy. MPS is single-device.
Unavailable CUDA fails rather than falling back when CUDA was requested
explicitly.

## Strategy selection

`strategy="auto"` chooses a policy compatible with resolved devices.
`"single_device"` executes in one process. `"ddp"`, `"ddp_spawn"`,
`"ddp_fork"`, and `"ddp_notebook"` select supported DDP launch behavior. A
configured Strategy instance is used when process backend, timeout, start method,
mesh, or checkpoint representation belongs to the experiment.

```python
from syntheticmind import Trainer
from syntheticmind.strategies.model_parallel import ModelParallelStrategy


strategy: ModelParallelStrategy = ModelParallelStrategy(
    data_parallel_size=2,
    tensor_parallel_size=4,
    save_distributed_checkpoint=True
)

trainer: Trainer = Trainer(
    accelerator="cuda",
    devices=8,
    strategy=strategy,
    precision="bf16-mixed"
)
```

Local launch supports one node. Set `num_nodes>1` only with an externally launched
environment that supplies the actual multi-node topology. Do not use a larger
node count as a request for Trainer to create remote machines.

## Precision

| Value | Parameter storage | Forward policy | Gradient scaling |
|---|---|---|---|
| `32-true` | Usually float32 | Full precision | None |
| `64-true` | Float64 | Float64 | None |
| `16-true` | Float16 | Float16 | None |
| `bf16-true` | Bfloat16 | Bfloat16 | None |
| `16-mixed` | Float32 master state | Float16 autocast | Yes |
| `bf16-mixed` | Float32 master state | Bfloat16 autocast | None |
| `transformer-engine` | Bfloat16 fallback | FP8 eligible operations | None |
| `transformer-engine-float16` | Float16 fallback | FP8 eligible operations | None |

Precision compatibility depends on Accelerator, Strategy, and explicitly supplied
plugins. CPU float16 mixed requests resolve to bfloat16 mixed behavior. Transformer
Engine requires CUDA and the supported optional package API. Model-parallel
restrictions are checked before execution.

Supply a configured Precision instance through `plugins`. Do not also pass a
conflicting precision value.

## Epoch, step, and time limits

`max_epochs` limits complete loader epochs. `max_steps` limits completed managed
optimizer mutations. `max_time` configures a Timer budget. `-1` means unbounded
for supported epoch or step ceilings.

Minimum epochs and steps delay cooperative stopping from callbacks; they do not
force execution beyond a hard maximum. A hard step or time stop inside a loader
retains a partial epoch rather than firing epoch-end hooks.

When time is the only terminal policy, Trainer resolves an unbounded epoch
ceiling and lets Timer govern. When an explicit epoch or step ceiling also
exists, the first satisfied hard condition wins.

```python
trainer: Trainer = Trainer(
    max_epochs=200,
    min_epochs=20,
    max_steps=100_000,
    max_time="02:12:00:00"
)
```

Timer strings use `DD:HH:MM:SS`. A `timedelta` or a non-negative integer mapping
of timedelta units is also accepted, and the resolved duration must be positive
and finite.

## Batch limits

`limit_train_batches`, `limit_val_batches`, `limit_test_batches`, and
`limit_predict_batches` accept an absolute non-negative integer or an appropriate
fraction of a sized loader. None resolves to the complete loader.

Fractions require a finite known length. Unsized iterable sources require an
explicit integer when they must be bounded. A zero limit disables that stage's
batch execution without pretending the corresponding client step exists.

`overfit_batches` intentionally reuses a stable bounded training subset for
training and fit validation. It is an experiment policy, not an automatic data
quality guarantee.

## Validation cadence

`check_val_every_n_epoch` determines eligible epochs. `val_check_interval`
determines placement within an eligible training horizon.

Supported interval forms include:

1. A positive integer training-batch interval.
2. A fraction of a sized epoch.
3. A monotonic duration string, timedelta, or unit mapping.
4. None or `1.0` for the conventional epoch-end placement.

Cross-epoch integer or time cadence can operate when epoch eligibility is None.
Unsized loaders validate at natural exhaustion where fractional placement cannot
be computed. A cooperative stop request may force validation so monitoring state
is not skipped before termination.

`num_sanity_val_steps` bounds the pre-fit validation probe. It runs only for a
fresh fit, not a resumed one. Sanity metrics are cleared and do not count toward
early-stopping patience, checkpoint ranking, or ordinary telemetry.

## Optimization policy

`accumulate_grad_batches` is a positive automatic-optimization microbatch count.
`gradient_clip_val` is a non-negative threshold and
`gradient_clip_algorithm` is `"norm"` or `"value"`. Gradients are exposed to
pre-step hooks after mixed-precision unscale and before clipping.

Manual optimization rejects Trainer-managed accumulation and clipping. The
Module must invoke its chosen accumulation and `clip_gradients` boundaries
explicitly.

`detect_anomaly=True` wraps fit in autograd anomaly detection. It is a diagnostic
cost, not a production default.

## Determinism and stochastic state

`seed` applies Python, NumPy, torch CPU, and selected accelerator streams.
`deterministic=True` requests deterministic PyTorch algorithms;
`deterministic="warn"` requests warning mode. `benchmark` controls cuDNN
benchmark policy where applicable.

Seeding and deterministic kernels are separate. A seed does not make a
nondeterministic kernel deterministic. Deterministic mode does not imply every
run should use the same seed.

Trainer snapshots and restores the active random streams through checkpoints and
worker state transfer. Do not initialize unused CUDA or MPS backends merely to
inspect their generators.

## Logging and callbacks

`logger=False` disables durable experiment records. None or True selects the
default logger policy. One Logger or an iterable enables explicit backends.
`log_every_n_steps` controls Trainer step-metric persistence, not callback hook
frequency.

`enable_checkpointing`, `enable_progress_bar`, and `enable_model_summary` control
default callback insertion. Explicit callback instances may conflict with a
disabled feature; checkpoint and progress conflicts fail rather than pretending
the instance is active.

Only one ProgressBarCallback is allowed. Checkpoint callbacks are ordered after
ordinary callbacks. A Timer is inserted when progress or `max_time` requires it;
an explicit Timer takes precedence over Trainer `max_time` with a warning.

## Profiler

`profiler` accepts None, `"simple"`, `"advanced"`, `"pytorch"`, or a configured
Profiler. String selection uses defaults. A configured instance should own paths,
schedule, activities, trace products, and backend-specific policy.

Profiling is a stage lifecycle. Reusing one instance sequentially is supported;
concurrent stage ownership is rejected.

## Data policy

`use_distributed_sampler=True` permits Strategy-aware sampler replacement for
map-style loaders. Disable it only when client sampling already owns correct
rank partitioning.

`reload_dataloaders_every_n_epochs=0` retains prepared fit loaders. A positive
value reconstructs them at that completed-epoch cadence and recomputes validation
scheduling against the new horizon.

## Fast development

`fast_dev_run=True` or a positive integer bounds every stage for a short
integration pass. Trainer uses one epoch, disables durable experiment logging,
suppresses checkpoint and early-stopping action, disables sanity validation, and
fixes validation cadence.

Use it to prove data, model, transfer, optimization, and stage wiring. It is not
evidence that full scheduling, convergence, checkpoint retention, or long-run
memory behavior is correct.

## Barebones measurement

`barebones=True` removes observation and diagnostic overhead for runtime
measurement. It rejects explicitly requested logging, profiling, checkpointing,
progress, model summary, sanity validation, fast-development, or anomaly features
that contradict that promise.

Barebones is not a general speed flag. It intentionally changes observability and
therefore should not be used for an experiment that requires those records.

## Hyperparameter autologging

`enable_autolog_hparams=True` merges captured Module and DataModule constructor
hyperparameters and emits them once through configured loggers. Conflicting keys
with different values fail instead of silently choosing one owner.

Call `save_hyperparameters` in client constructors for values that should support
reconstruction and experiment identity. Live models, tensors with several
elements, and process resources are not hyperparameters.

## Checkpoint continuation

Every stage accepts `ckpt_path`; mutable `trainer.ckpt_path` provides a default
selector. Fit requires a training checkpoint for continuation. Validation, test,
and prediction may load a model artifact or training checkpoint according to the
requested `weights_only` loading policy.

The effective Trainer configuration is checked before fit restoration. Terminal
ceilings may increase, but changes to trajectory-critical data, optimization,
precision, sampler, validation, and deterministic policy fail.

## Configuration review checklist

Before launching a consequential experiment, verify:

1. Accelerator availability and exact device identities.
2. Strategy launch model, node ownership, and script pickleability or
   importability.
3. Precision compatibility with model operations and optimizer.
4. Effective batch size across per-device batch, accumulation, and data-parallel
   replicas.
5. Validation and checkpoint cadence against actual loader length.
6. The metric names consumed by schedulers, early stopping, and checkpointing.
7. Whether outputs, logs, profiler traces, and checkpoints have owned paths.
8. Whether a resume changes any trajectory-critical setting.
9. Whether distributed samplers or client samplers own partitioning.
10. Whether the chosen stop conditions can terminate the run.

## Related documentation

`../../trainer/README.md` is the public Trainer reference.
`../architecture/training_lifecycle.md` explains execution order.
`../architecture/distributed_execution.md` explains topology.
`../architecture/checkpoint_trajectory.md` explains continuation compatibility.
