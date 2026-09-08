# syntheticmind.trainer: Lifecycle Orchestration

`trainer/` owns the complete fit, validation, test, and prediction lifecycles. `trainer.py` is the canonical implementation of the single public `Trainer` stage entrypoint, while connectors own its internal runtime composition boundaries.

| Capability | Ownership |
|---|---|
| `trainer.py` | Public stage entrypoints and top-level lifecycle sequencing across connectors and loops. |
| `configuration.py` | Private validated run policy, scalar invariants, default resolution, and fast-development, overfit, and barebones interactions. |
| `hook_dispatcher.py` | Private callback, module, data-module, and Strategy hook invocation, profiling, context, and override discovery. |
| `states.py` | Trainer function, stage, status, progress, interruption, and stop state. |
| `console.py` | Rank-safe live-line, record, metric, styling, and diagnostics-capture rendering for the Trainer and progress callbacks. |
| `connectors/accelerator_connector.py` | Accelerator, devices, nodes, Strategy, precision, cluster environment, checkpoint I/O, and layer-synchronization compatibility. |
| `connectors/callback_connector.py` | Callback validation, defaults, model callback attachment, ordering, and checkpoint identities. |
| `connectors/checkpoint_connector.py` | Checkpoint selection, full-trajectory assembly, restoration, and resume lifecycle. |
| `connectors/data_connector.py` | Module, DataModule, dataloader, sampler, transfer, and reload attachment. |
| `connectors/logger_connector/` | Metric routing, aggregation, logger emission, callback metrics, and progress metrics. |
| `connectors/signal_connector.py` | Process-signal registration, propagation, interruption, and teardown. |

Connector names are not a quota. A connector is admitted only when one configuration family crosses several framework components and requires a single ordering and compatibility owner. Optimizer construction remains a Strategy responsibility because distributed and model-parallel setup can change when parameters become available.

Checkpoint continuation is selected explicitly through each stage's `ckpt_path` argument or the mutable `trainer.ckpt_path` policy. The obsolete constructor-level `resume_from_checkpoint` alias is not part of the current API.

## Construction

Trainer configuration falls into six domains:

1. Execution: `accelerator`, `strategy`, `devices`, `num_nodes`, `precision`, `plugins`, and `sync_batchnorm`.
2. Run limits: `max_epochs`, `min_epochs`, `max_steps`, `min_steps`, and `max_time`.
3. Data: per-stage batch limits, `overfit_batches`, distributed sampler policy, validation cadence, sanity validation, and loader reload cadence.
4. Optimization: accumulation, gradient clipping, deterministic execution, benchmarking, and anomaly detection.
5. Observation: logger, callbacks, step logging cadence, progress, model summary, profiler, and automatic hyperparameters.
6. Infrastructure: root artifact directory and stochastic seed.

Scalar policy is validated before any client model, dataloader, filesystem, device, process, or random stream is touched. Accelerator and Strategy compatibility resolves before callbacks, logging, data, or loops are constructed.

```python
from syntheticmind import Trainer

trainer: Trainer = Trainer(
    accelerator="cuda",
    strategy="ddp",
    devices=4,
    precision="bf16-mixed",
    max_epochs=100,
    accumulate_grad_batches=4,
    callbacks=[checkpoint, early_stopping],
    logger=experiment_logger,
    seed=731
)
```

## Fit

`fit` accepts direct training and validation loaders or one DataModule. Direct loaders and `datamodule` are mutually exclusive. When neither direct data nor DataModule supplies a stage, Trainer invokes the Module dataloader hook.

```python
trainer.fit(
    model,
    datamodule=datamodule,
    ckpt_path=resume_path
)
```

One local fit follows this sequence:

1. Claim fit state and clear prior observable and continuation state.
2. Attach Module, DataModule, sources, and model-provided callbacks.
3. Register SIGTERM handling and initialize Strategy environment.
4. Prepare data on global rank zero, barrier, and run process-local setup.
5. Preload checkpoint metadata when continuing.
6. Request training data and resolve validation cadence.
7. Configure model, layer synchronization, precision, Strategy wrapping, optimizers, schedulers, and managed optimizer adapters.
8. Validate and restore the complete checkpoint trajectory.
9. Dispatch fit start, optional sanity validation, FitLoop, and fit end.
10. Preserve multiprocessing state and attempt every teardown operation.

`current_epoch` advances only after the training loader exhausts. `global_step` advances only after a managed optimizer mutation. A stop inside an epoch retains loader, metric, scheduler, and stochastic continuation state.

## Validation and test

`validate` and `test` return one `dict[str, float]` per sequential dataloader. Passing an explicit model evaluates its current state unless a checkpoint path is also supplied. Omitting the model after a previous fit reuses the attached Module and defaults checkpoint selection to `"best"` when no path is supplied.

```python
validation_results: list[dict[str, float]] = trainer.validate(
    model,
    datamodule=datamodule,
    verbose=False
)

test_results: list[dict[str, float]] = trainer.test(
    datamodule=datamodule,
    ckpt_path="best"
)
```

Standalone evaluation may use `torch.inference_mode`. Validation scheduled inside fit uses `torch.no_grad` to preserve training-compatible tensor behavior.

## Prediction

`predict` returns one CPU output list or one list per sequential dataloader. `return_predictions=False` executes hooks and model computation without retaining outputs. Multiprocessing launch requires disabled return because arbitrary worker output trees are not reconciled into the parent.

```python
predictions = trainer.predict(
    model,
    datamodule=datamodule,
    return_predictions=True
)
```

## Checkpoint selection

Every stage accepts a concrete path, `"best"`, `"last"`, or `None`. `trainer.ckpt_path` provides mutable default policy and an explicit stage argument takes precedence with warning.

`"best"` uses the first ModelCheckpoint best path. `"last"` inspects configured callback state and valid last artifacts and chooses the newest. Unsupported `"hpc"` and `"registry"` values fail unless those integrations become real capabilities.

A weights-only save is a model artifact and cannot resume fit. A training checkpoint contains model, optimizer, scheduler, callbacks, DataModule, precision, configuration, progress, loader cursor, random streams, and partial metric windows.

## Strategy-aware construction

Use `init_module` when construction must follow selected device, precision, or empty-initialization policy:

```python
with trainer.init_module(empty_init=True):
    model: LargeModel = LargeModel(configuration)
```

Model-parallel clients apply tensor-parallel or composable FSDP2 plans in `Module.configure_model`, where `device_mesh` is available.

## Runtime inspection

Trainer exposes resolved accelerator, Strategy, precision, scaler, devices, ranks, node and world size, callbacks, loggers, current epoch, global step, effective batch counts, active dataloaders, metric views, stage flags, interruption, SIGTERM, and estimated automatic stepping batches.

`callback_metrics` is the tensor view for monitoring and client inspection. `logged_metrics` contains logger-routed tensors. `progress_bar_metrics` contains Python floats for the active or latest visible stage. Every mapping is defensive.

`Trainer.print` writes from local rank zero on every node. `Module.print` writes only from global rank zero. Both route through the active progress callback so an interactive live line is cleared before client output appears. Verbose validation and test results, and the barebones stage marker, use the same renderer for the same reason.

## Development and barebones modes

`fast_dev_run` bounds every stage, uses one epoch, disables durable experiment loggers, suppresses checkpoint and early-stopping actions, disables sanity validation, and fixes validation cadence for a short integration run.

`barebones=True` measures minimal runtime overhead. It rejects explicitly requested logger, profiler, checkpoint, progress, model-summary, sanity, development, or anomaly features that contradict the promise instead of silently ignoring them.

## Failure and teardown

The first execution failure remains causal. Callback, DataModule, and Strategy exception hooks are all attempted; their failures become notes. Teardown then attempts exact mode restoration, signals, Strategy, checkpoint transport, layer synchronization, precision, cluster environment, dataloaders, DataModule, callbacks, Module, loggers, progress, and profiler. One failed cleanup cannot prevent later resources from releasing.

## Related components

[`connectors/README.md`](connectors/README.md) describes runtime composition. `../loops/README.md` describes iterative execution. `../core/README.md` describes client model contracts. `../strategies/README.md` describes execution policy.
