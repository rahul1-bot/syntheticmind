# Model and Data Integration

A client experiment enters SyntheticMind through two primary contracts:

1. `Module` owns model computation, research steps, optimizer declarations,
   metric production, and model-specific lifecycle hooks.
2. `DataModule` owns dataset preparation, process-local construction, dataloaders,
   transfer customization, data state, and data teardown.

Trainer composes them. Client code does not subclass Trainer merely to inject a
model step or a dataloader.

## Minimal trainable Module

```python
from typing import override

import torch
from torch import nn

from syntheticmind import Module
from syntheticmind.primitives.types import Batch, StepOutput, TrainingStepOutput


class Classifier(Module):
    def __init__(self, input_width: int, class_count: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.network: nn.Linear = nn.Linear(input_width, class_count)

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
            raise TypeError("Classifier expects an input-target pair.")
        inputs, targets = batch
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("Classifier inputs must be tensors.")
        if not isinstance(targets, torch.Tensor):
            raise TypeError("Classifier targets must be tensors.")
        logits: torch.Tensor = self(inputs)
        loss: torch.Tensor = torch.nn.functional.cross_entropy(logits, targets)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=int(inputs.shape[0])
        )
        return {"loss": loss}

    @override
    def validation_step(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> StepOutput:
        del batch_idx, dataloader_idx
        if not isinstance(batch, list | tuple) or len(batch) != 2:
            raise TypeError("Classifier expects an input-target pair.")
        inputs, targets = batch
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("Classifier inputs must be tensors.")
        if not isinstance(targets, torch.Tensor):
            raise TypeError("Classifier targets must be tensors.")
        logits: torch.Tensor = self(inputs)
        loss: torch.Tensor = torch.nn.functional.cross_entropy(logits, targets)
        accuracy: torch.Tensor = (logits.argmax(dim=-1) == targets).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True)
        return {"loss": loss.detach(), "accuracy": accuracy.detach()}

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=3e-4)
```

`Module` already inherits `torch.nn.Module`; inheriting both creates an
unnecessary diamond. `forward` remains the ordinary PyTorch inference
computation. Stage-step methods express training-system semantics around that
computation.

Automatic optimization requires a tensor loss returned directly or under the
`"loss"` mapping key. Other mapping fields are available to batch-end hooks but
do not become metrics automatically. Publish metrics deliberately through
`self.log` or `self.log_dict`.

## Metric semantics

`Module.log` records an observation at the active hook. It does not immediately
write a backend file.

| Option | Meaning |
|---|---|
| `on_step` | Publish an immediate value for the current step cadence. |
| `on_epoch` | Accumulate and publish at the owning epoch boundary. |
| `prog_bar` | Make the value eligible for bounded progress presentation. |
| `logger` | Route the value to configured experiment loggers. |
| `reduce_fx` | Scalar epoch reduction: mean, sum, min, or max. |
| `sync_dist` | Reduce a scalar through the active Strategy. |
| `batch_size` | Explicit weight for epoch means when cardinality cannot be inferred. |
| `rank_zero_only` | Mark a value produced only on global zero and incompatible with distributed reduction. |

Training-step hooks default to step publication. Validation and test step hooks
default to epoch publication. Epoch hooks permit epoch values only. Prediction,
transfer, configuration, checkpoint, and teardown hooks reject logging because
they have no coherent metric cadence.

For distributed epoch means, SyntheticMind reduces numerator and denominator,
not the already averaged rank values. A registered `torchmetrics.Metric` retains
its own state and synchronization behavior.

## DataModule lifecycle

```python
from pathlib import Path
from typing import override

import torch
from torch.utils.data import DataLoader, TensorDataset

from syntheticmind import DataModule
from syntheticmind.primitives.types import TrainerStage


class FeatureDataModule(DataModule):
    def __init__(self, dataset_root: Path, batch_size: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.dataset_root: Path = dataset_root
        self.batch_size: int = batch_size
        self.train_dataset: TensorDataset | None = None
        self.validation_dataset: TensorDataset | None = None

    @override
    def prepare_data(self) -> None:
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(self.dataset_root)

    @override
    def setup(self, stage: TrainerStage) -> None:
        if stage != "fit":
            return
        train_inputs: torch.Tensor = torch.load(
            self.dataset_root / "train_inputs.pt",
            weights_only=True
        )
        train_targets: torch.Tensor = torch.load(
            self.dataset_root / "train_targets.pt",
            weights_only=True
        )
        validation_inputs: torch.Tensor = torch.load(
            self.dataset_root / "validation_inputs.pt",
            weights_only=True
        )
        validation_targets: torch.Tensor = torch.load(
            self.dataset_root / "validation_targets.pt",
            weights_only=True
        )
        self.train_dataset = TensorDataset(train_inputs, train_targets)
        self.validation_dataset = TensorDataset(
            validation_inputs,
            validation_targets
        )

    @override
    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Training data has not been configured.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    @override
    def val_dataloader(self) -> DataLoader:
        if self.validation_dataset is None:
            raise RuntimeError("Validation data has not been configured.")
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
```

`prepare_data` runs only on global zero and is followed by a barrier. Use it for
single-copy preparation or validation. `setup` runs in every process; use it for
process-local dataset construction. Do not create rank-local state in
`prepare_data`.

DataModule teardown is attempted after every stage, including failure. External
handles opened during setup belong to that lifecycle and should close there.

## Direct datasets

Projects that already own constructed datasets can use the native adapter:

```python
from syntheticmind import DataModule


datamodule: DataModule = DataModule.from_datasets(
    train_dataset=train_dataset,
    val_dataset=validation_dataset,
    batch_size=64,
    num_workers=8
)
```

Dataset trees may contain nested dictionaries, lists, or tuples. SyntheticMind
builds one DataLoader per leaf and reconstructs the topology. Map-style training
data shuffles by default; iterable data keeps its own order; evaluation and
prediction do not shuffle.

## Data source precedence

For each stage, source precedence is:

1. Dataloader passed directly to the Trainer stage call.
2. Dataloader method on the supplied DataModule.
3. Dataloader hook on the Module.

Direct data and a DataModule are mutually exclusive in one stage call. Prefer a
DataModule when preparation, state, transfer, or several stages share ownership.
Use direct data for a genuinely local experiment input, not as a way to bypass a
coherent data lifecycle.

## Batch transfer

DataModule and Module both expose `on_before_batch_transfer`,
`transfer_batch_to_device`, and `on_after_batch_transfer`. When both provide an
override, DataModule owns the operation. Trainer executes the same transaction
for training, validation, testing, prediction, and model-summary example input.

Standard tensors and nested containers move recursively. A custom batch type can
participate structurally by implementing `to(device, non_blocking=False)`.
Transfer hooks should preserve batch semantics; they are not a substitute for
model forward or augmentation that belongs to dataset construction.

## Multiple dataloaders

Return a nested loader collection from a client hook or construct
`CombinedLoader` explicitly:

```python
from syntheticmind import CombinedLoader


training_loaders: CombinedLoader = CombinedLoader(
    {
        "labeled": labeled_loader,
        "unlabeled": unlabeled_loader
    },
    mode="max_size_cycle"
)
```

`min_size` stops at the shortest leaf. `max_size` yields None for exhausted
leaves. `max_size_cycle` cycles shorter leaves to the longest horizon.
`sequential` consumes leaves one at a time and retains their flattened indices.

Trainer defaults plain training collections to `max_size_cycle` and evaluation
or prediction collections to `sequential`. An explicit CombinedLoader retains
its mode except prediction requires sequential identity.

## Stateful data continuation

DataModule `state_dict` and `load_state_dict` preserve data state owned by the
client. Stateful DataLoaders or samplers preserve their own cursor mappings.
Trainer owns flattening order, progress coordinates, checkpoint timing, and
restoration order.

Persist semantic state, not live iterators, workers, locks, or open file handles.
The reconstructed loader topology must match before cursor restoration.

## Optimizers and schedulers

`configure_optimizers` may return native optimizer and scheduler declaration
forms normalized by `core/optimizer.py`. Automatic optimization supports one
optimizer. Step schedulers advance after real optimizer progress; epoch
schedulers advance only after complete loader epochs; plateau schedulers wait
until their monitored metric exists.

Multiple optimizers require manual optimization:

```python
self.automatic_optimization = False
generator, discriminator = self.optimizers()

with discriminator.toggle_model():
    discriminator.zero_grad()
    discriminator_loss: torch.Tensor = self.discriminator_loss(batch)
    self.manual_backward(discriminator_loss)
    discriminator.step()
```

Managed optimizers preserve Strategy synchronization, precision, hooks,
checkpoint state, and global-step accounting. Requesting raw optimizers opts out
of those managed operations deliberately.

## Model-provided callbacks

Override `configure_callbacks` when a callback belongs to the model's reusable
training contract. Trainer callbacks remain appropriate for run-specific policy.
A model callback subclass replaces a compatible Trainer callback type according
to inheritance; unrelated callbacks remain ordered, and checkpoint callbacks
run last.

Stateful callbacks need a unique `state_key`. Do not retain the Trainer or Module
inside callback state; both are passed explicitly to every hook.

## Running the experiment

```python
from syntheticmind import Trainer
from syntheticmind.callbacks.checkpointing import ModelCheckpoint
from syntheticmind.callbacks.early_stopping import EarlyStopping
from syntheticmind.loggers import CSVLogger


checkpoint: ModelCheckpoint = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=3,
    save_last=True
)
early_stopping: EarlyStopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=10
)
logger: CSVLogger = CSVLogger("artifacts/logs", name="classifier")

trainer: Trainer = Trainer(
    accelerator="auto",
    precision="32-true",
    max_epochs=100,
    callbacks=[checkpoint, early_stopping],
    logger=logger,
    seed=731
)

model: Classifier = Classifier(input_width=512, class_count=10)
data: FeatureDataModule = FeatureDataModule(
    dataset_root=Path("datasets/features"),
    batch_size=64
)
trainer.fit(model, datamodule=data)
validation_results: list[dict[str, float]] = trainer.validate(
    datamodule=data,
    ckpt_path="best"
)
```

`validate` omits the model here because Trainer already owns it from fit. With no
explicit model and no checkpoint selector, standalone evaluation defaults to the
best checkpoint. Passing `ckpt_path="best"` makes that decision visible.

## Large and model-parallel construction

Construct layers inside `Trainer.init_module` when precision-directed dtype or
empty initialization must apply during allocation. Put tensor-parallel or
composable FSDP2 transformations in `Module.configure_model`, where the named
device mesh is available.

Ordinary models should construct ordinary layers in `__init__`. Do not move every
layer into `configure_model` merely because the hook exists.

## Related documentation

`../../core/README.md` documents Module metrics, hooks, optimization, and
reconstruction. `../../data/README.md` documents data composition.
`../architecture/training_lifecycle.md` documents execution order.
`trainer_configuration.md` documents Trainer policy.
