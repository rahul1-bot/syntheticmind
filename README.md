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

## Minimal experiment

```python
from typing import override

import torch
from torch.utils.data import DataLoader, TensorDataset

from syntheticmind import Module, Trainer
from syntheticmind.primitives.types import Batch, TrainingStepOutput


class Regressor(Module):
    def __init__(self, input_width: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.network: torch.nn.Linear = torch.nn.Linear(input_width, 1)

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

    @override
    def training_step(
        self,
        batch: Batch,
        batch_idx: int
    ) -> TrainingStepOutput:
        del batch_idx
        if not isinstance(batch, list | tuple) or len(batch) != 2:
            raise TypeError("Regressor expects an input-target pair.")
        inputs, targets = batch
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("Regressor inputs must be tensors.")
        if not isinstance(targets, torch.Tensor):
            raise TypeError("Regressor targets must be tensors.")
        prediction: torch.Tensor = self(inputs)
        loss: torch.Tensor = torch.nn.functional.mse_loss(prediction, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss}

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


inputs: torch.Tensor = torch.randn(128, 16)
targets: torch.Tensor = torch.randn(128, 1)
loader: DataLoader = DataLoader(
    TensorDataset(inputs, targets),
    batch_size=16,
    shuffle=True
)

trainer: Trainer = Trainer(
    accelerator="auto",
    precision="32-true",
    max_epochs=10,
    seed=731
)
trainer.fit(Regressor(input_width=16), train_dataloaders=loader)
```

This example uses automatic optimization. The loop owns zeroing, backward,
accumulation, clipping, optimizer mutation, scheduler cadence, metrics, and
progress. The model owns its computation and optimizer declaration.

## Contracts

The public surface follows established PyTorch research vocabulary. Models subclass `Module`, which is itself a `torch.nn.Module`, and express behavior through `forward`, `training_step`, `validation_step`, `test_step`, `predict_step`, and `configure_optimizers`. Metrics are emitted through `Module.log` and `Module.log_dict` with explicit step and epoch reduction, logger visibility, and distributed-synchronization semantics. Data behavior is expressed through `DataModule` preparation, setup, teardown, per-stage dataloader construction, state persistence, and checkpoint reconstruction.

`Trainer`, `Module`, `DataModule`, `DatasetDataModule`, and `CombinedLoader` are available from the `syntheticmind` package root. Their implementations remain owned by the semantic modules and subpackages shown above. Nested dataloader composition lives in `syntheticmind.data.combined_loader`, outside the Trainer connector package, because both model-facing data construction and Trainer stage execution consume it.

Framework-wide type contracts live in `syntheticmind.primitives.types`, and the framework exception vocabulary lives in `syntheticmind.primitives.exceptions`. The `primitives` package is the dependency-inward language shared by semantic subsystems; it admits no algorithms, resources, configuration resolution, or mutable runtime behavior.

`Trainer` orchestrates the fit, validate, test, and predict stages under parameters covering epoch ceilings, accelerator and strategy selection, precision, gradient accumulation and clipping, validation cadence, sanity validation, per-stage batch limits, stochastic stream seeding, deterministic execution, and checkpoint resume. Manual optimization through `manual_backward`, `optimizer_step`, `toggle_optimizer`, and `untoggle_optimizer` supports research systems with alternating or otherwise model-controlled optimizer updates.

## Runtime capabilities

The current private core implements:

1. CPU, CUDA, and Apple MPS device selection with single-device execution.
2. DDP through subprocess, spawn, fork, notebook-oriented fork, and externally
   launched process environments where backend constraints permit.
3. Named data- and tensor-parallel device meshes with client tensor-parallel or
   composable FSDP2 plans.
4. Full, true-half, mixed, and optional Transformer Engine precision policies.
5. Automatic and manual optimization, accumulation, clipping, schedulers,
   multiple dataloaders, validation cadence, and exact partial-epoch continuation.
6. Scalar and TorchMetrics logging, CSV and TensorBoard records, lifecycle
   callbacks, model inspection, runtime profilers, and local or remote artifacts.
7. Full and sharded model-parallel checkpoints with atomic publication policy.

Capabilities are admitted only when their complete runtime, state, failure, and
teardown paths exist. SyntheticMind does not claim Fabric parity, XLA execution,
distributed MPS, local multi-node provisioning, universal upstream Trainer
options, public release readiness, or compatibility with vendor internals.

## Lifecycle at a glance

Trainer validates configuration before touching client resources. A stage then
resolves model and data ownership, establishes process and device execution,
runs data and callback setup, configures or restores model and optimization
state, enters its loop, and attempts every teardown participant even after
failure.

Fit owns complete loader epochs. `global_step` advances only after a managed
optimizer mutation. A hard stop inside an epoch retains its loader cursor,
partial metric windows, scheduler position, random streams, and current epoch;
it does not fabricate epoch completion.

See [`docs/architecture/training_lifecycle.md`](docs/architecture/training_lifecycle.md)
for the exact transaction and hook boundaries.

## Checkpoints

Checkpoint schema version 2 is native and resumable. A durable training checkpoint carries model, optimizer, scheduler, callback, datamodule, resolved Trainer configuration, batch progress, epoch, step, random-number-generator state, and any rank-local partial-epoch scalar or TorchMetrics window, so an interrupted run continues from its last durable transition rather than fabricating an epoch boundary. Random state covers Python, NumPy, torch CPU, and the active CUDA or MPS backend through the same immutable schema used for multiprocessing transfer. A weights-only artifact is explicitly non-resumable. Local execution uses an interchangeable structural `CheckpointIO` contract. Model-parallel execution owns full or sharded distributed state directly, and sharded writes publish immutable generations through an atomic active-generation pointer.

## Related Components

Downstream research projects consume the package-root contracts and specialize models, data, callbacks, and parallelization plans without changing framework ownership. The committed regression suite under [`tests/`](tests/README.md) is the executable evidence for the framework and runs on CPU through `python -m pytest`; `pyproject.toml` makes the checkout installable with `python -m pip install -e ".[test]"`, and the `tests` workflow runs the suite on every push. Comparative source references remain outside the tracked package.

The documentation map begins at [`docs/README.md`](docs/README.md). The principal
cross-package references are:

| Document | Purpose |
|---|---|
| [`framework_topology.md`](docs/architecture/framework_topology.md) | Concept ownership, dependency direction, and extension admission. |
| [`training_lifecycle.md`](docs/architecture/training_lifecycle.md) | Fit, evaluation, prediction, hook order, durability, and teardown. |
| [`checkpoint_trajectory.md`](docs/architecture/checkpoint_trajectory.md) | Schema-v2 state, restoration, exact continuation, and publication. |
| [`distributed_execution.md`](docs/architecture/distributed_execution.md) | Devices, ranks, launch modes, data and tensor axes, and collectives. |
| [`model_and_data_integration.md`](docs/guides/model_and_data_integration.md) | Current Module and DataModule client patterns. |
| [`trainer_configuration.md`](docs/guides/trainer_configuration.md) | Configuration domains, compatibility, and execution recipes. |
| [`extension_points.md`](docs/guides/extension_points.md) | Selecting and implementing the correct framework extension boundary. |

Every semantic package carries its own README. Source comments and Markdown
follow [`docs/project_guidelines/documentation_style.md`](docs/project_guidelines/documentation_style.md).
