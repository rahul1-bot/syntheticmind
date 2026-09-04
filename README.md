# syntheticmind: A PyTorch Research Training Framework

`syntheticmind/` is an independent PyTorch research framework for constructing and controlling machine-learning training systems. It is project-agnostic by design: domain models, datasets, simulation environments, and experiment policy remain downstream of the framework contracts rather than leaking into the reusable training lifecycle.

| Subpackage | Contents |
|---|---|
| `core/` | The `Module` contract, hook interfaces, model-facing optimization, metric production, checkpoint participation, and hyperparameter surface. |
| `data/` | The `DataModule` lifecycle and nested `CombinedLoader` data-composition policies. |
| `trainer/` | The `Trainer`, validated run configuration, deterministic execution policy, live Trainer state, and accelerator, callback, checkpoint, data, logger, and signal connectors. |
| `loops/` | The fit, training-epoch, evaluation, and prediction loops the trainer dispatches to. |
| `accelerators/` | The device abstraction covering CPU, CUDA, and Apple MPS. |
| `strategies/` | The stateful execution-policy hierarchy for single-device, distributed data-parallel, tensor-parallel, and composable fully sharded execution. |
| `plugins/` | Checkpoint I/O, cluster-environment, layer-synchronization, and numerical-precision extension families. |
| `callbacks/` | The lifecycle callbacks: checkpointing, early stopping, exponential moving averages, model inspection, monitoring, and numerical guards. |
| `loggers/` | The metric-logger interface with CSV and TensorBoard implementations. |
| `profilers/` | Runtime action measurement and trace export. |
| `storage/` | Path-bound local and remote filesystem resources shared by checkpoints, loggers, profilers, and run artifacts. |
| `rng/` | Process-wide Python, NumPy, torch CPU, CUDA, and MPS random streams, seeding, and immutable trajectory snapshots. |
| `primitives/` | Dependency-inward framework type, structural, and exception vocabulary without runtime behavior. |

## Contracts

The public surface follows established PyTorch research vocabulary. Models subclass `Module`, which is itself a `torch.nn.Module`, and express behavior through `forward`, `training_step`, `validation_step`, `test_step`, `predict_step`, and `configure_optimizers`. Metrics are emitted through `Module.log` and `Module.log_dict` with explicit step and epoch reduction, logger visibility, and distributed-synchronization semantics. Data behavior is expressed through `DataModule` preparation, setup, teardown, per-stage dataloader construction, state persistence, and checkpoint reconstruction.

`Trainer`, `Module`, `DataModule`, `DatasetDataModule`, and `CombinedLoader` are available from the `syntheticmind` package root. Their implementations remain owned by the semantic modules and subpackages shown above. Nested dataloader composition lives in `syntheticmind.data.combined_loader`, outside the Trainer connector package, because both model-facing data construction and Trainer stage execution consume it.

Framework-wide type contracts live in `syntheticmind.primitives.types`, and the framework exception vocabulary lives in `syntheticmind.primitives.exceptions`. The `primitives` package is the dependency-inward language shared by semantic subsystems; it admits no algorithms, resources, configuration resolution, or mutable runtime behavior.

`Trainer` orchestrates the fit, validate, test, and predict stages under parameters covering epoch ceilings, accelerator and strategy selection, precision, gradient accumulation and clipping, validation cadence, sanity validation, per-stage batch limits, stochastic stream seeding, deterministic execution, and checkpoint resume. Manual optimization through `manual_backward`, `optimizer_step`, `toggle_optimizer`, and `untoggle_optimizer` supports research systems with alternating or otherwise model-controlled optimizer updates.

## Checkpoints

Checkpoint schema version 2 is native and resumable. A durable training checkpoint carries model, optimizer, scheduler, callback, datamodule, resolved Trainer configuration, batch progress, epoch, step, random-number-generator state, and any rank-local partial-epoch scalar or TorchMetrics window, so an interrupted run continues from its last durable transition rather than fabricating an epoch boundary. Random state covers Python, NumPy, torch CPU, and the active CUDA or MPS backend through the same immutable schema used for multiprocessing transfer. A weights-only artifact is explicitly non-resumable. Local execution uses an interchangeable structural `CheckpointIO` contract. Model-parallel execution owns full or sharded distributed state directly, and sharded writes publish immutable generations through an atomic active-generation pointer.

## Related Components

Downstream research projects consume the package-root contracts and specialize models, data, callbacks, and parallelization plans without changing framework ownership. Focused runtime probes and source references remain outside the tracked production package.

The package-level documentation begins with [`data/README.md`](data/README.md), [`primitives/README.md`](primitives/README.md), and [`rng/README.md`](rng/README.md). Source comments and future Markdown documentation follow [`docs/project_guidelines/documentation_style.md`](docs/project_guidelines/documentation_style.md).
