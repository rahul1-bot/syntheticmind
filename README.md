# syntheticmind: The PyTorch Training Framework

`syntheticmind/` is an independent PyTorch research framework and the training substrate of the VOCODE study. It is study-agnostic by design: nothing in this package references vocoders, audio, or the `vocode` package, so the framework boundary separates reusable training machinery from experiment-specific logic.

| Subpackage | Contents |
|---|---|
| `core/` | The `Module` and `DataModule` contracts, hook interfaces, model-facing optimizer contract, checkpoint hooks, and hyperparameter surface. |
| `trainer/` | The `Trainer`, validated run configuration, live Trainer state, and accelerator, callback, checkpoint, data, logger, and signal connectors. |
| `loops/` | The fit, training-epoch, evaluation, and prediction loops the trainer dispatches to. |
| `accelerators/` | The device abstraction covering CPU, CUDA, and Apple MPS. |
| `strategies/` | The stateful execution-policy hierarchy for single-device, distributed data-parallel, tensor-parallel, and composable fully sharded execution. |
| `plugins/` | Checkpoint I/O, cluster-environment, layer-synchronization, and numerical-precision extension families. |
| `callbacks/` | The lifecycle callbacks: checkpointing, early stopping, exponential moving averages, model inspection, monitoring, and numerical guards. |
| `loggers/` | The metric-logger interface with CSV and TensorBoard implementations. |
| `profilers/` | Runtime action measurement and trace export. |
| `storage/` | Local and remote filesystem boundaries shared by checkpoints, loggers, profilers, and run artifacts. |

## Contracts

The public surface follows established PyTorch research vocabulary. Models subclass `Module`, which is itself a `torch.nn.Module`, and express behavior through `forward`, `training_step`, `validation_step`, `test_step`, `predict_step`, and `configure_optimizers`. Metrics are emitted through `Module.log` and `Module.log_dict` with explicit step and epoch reduction, logger visibility, and distributed-synchronization semantics. Data behavior is expressed through the `DataModule` methods `prepare_data`, `setup`, `teardown`, and the per-stage dataloader constructors.

`Trainer`, `Module`, `DataModule`, `DatasetDataModule`, and `CombinedLoader` are available from the `syntheticmind` package root. Their implementations remain owned by the semantic modules and subpackages shown above. Nested dataloader composition lives in `syntheticmind.combined_loader`, outside the Trainer connector package, because both model-facing data construction and Trainer stage execution consume it.

Framework-wide type contracts live in `syntheticmind.types`, and the framework exception vocabulary lives in `syntheticmind.exceptions`. These dependency-direction contracts sit at the package root because every semantic subsystem consumes them; they are not utility behavior and do not belong to any one runtime concept.

`Trainer` orchestrates the fit, validate, test, and predict stages under parameters covering epoch ceilings, accelerator and strategy selection, precision, gradient accumulation and clipping, validation cadence, sanity validation, per-stage batch limits, deterministic seeding, and checkpoint resume. Manual optimization through `manual_backward`, `optimizer_step`, `toggle_optimizer`, and `untoggle_optimizer` supports multi-optimizer research models; the adversarial vocoders of this study use this path for their alternating generator and discriminator updates.

## Checkpoints

Checkpoints are native and resumable. A durable checkpoint carries model, optimizer, scheduler, callback, datamodule, epoch, step, and random-number-generator state, so an interrupted run continues from its last durable state rather than restarting. Local execution uses an interchangeable structural `CheckpointIO` contract. Model-parallel execution owns full or sharded distributed state directly, and sharded writes publish immutable generations through an atomic active-generation pointer.

## Related Components

Each subpackage carries its own `README.md`. The study client in `../vocode/` layers on this framework, and the test suite under `../tests/` exercises the framework through that client.
