# syntheticmind.data: Data Lifecycle and Composition

`data/` defines how client datasets enter SyntheticMind and how one or several dataloaders become the uniform stream consumed by the execution loops. The package does not own optimization, distributed process policy, or stage orchestration.

| Module | Contents |
|---|---|
| `datamodule.py` | `DataModule`, the client-extensible data lifecycle, and `DatasetDataModule`, the direct dataset-backed implementation. |
| `combined_loader.py` | `CombinedLoader`, the nested multi-loader composition and continuation boundary. |

## DataModule integration

Subclass `DataModule` when a project needs preparation, stage-specific construction, custom batch transfer, external resources, or resumable data state. `prepare_data` is for single-copy work such as downloading or validating a corpus. `setup` is for process-local construction and receives the active Trainer stage. The four dataloader methods expose only the stages the project supports.

```python
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from syntheticmind import DataModule
from syntheticmind.primitives.types import TrainerStage


class ImageDataModule(DataModule):
    def __init__(self, dataset_root: Path, batch_size: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._dataset_root: Path = dataset_root
        self._batch_size: int = batch_size
        self._train_dataset: torch.utils.data.Dataset | None = None

    def prepare_data(self) -> None:
        validate_dataset(self._dataset_root)

    def setup(self, stage: TrainerStage) -> None:
        if stage == "fit":
            self._train_dataset = build_training_dataset(self._dataset_root)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True
        )
```

`state_dict` and `load_state_dict` form the data continuation contract. A stateful sampler or streaming source should persist only the state it owns; loader cursor coordination and checkpoint ordering remain framework responsibilities.

## Direct datasets

`DataModule.from_datasets` and `DatasetDataModule` cover ordinary projects that already hold constructed datasets. Dataset trees may be nested dictionaries, lists, or tuples. SyntheticMind builds one `DataLoader` per leaf and reconstructs the original topology.

```python
datamodule: DataModule = DataModule.from_datasets(
    train_dataset=train_dataset,
    val_dataset=validation_dataset,
    batch_size=32,
    num_workers=8
)
```

Map-style training datasets shuffle by default. Iterable datasets retain their own ordering. Validation, test, and prediction loaders do not shuffle.

## Multiple dataloaders

Client loader factories may return a nested collection directly. Trainer selects the standard composition policy for the active stage. Construct `CombinedLoader` explicitly when the research method requires a non-default policy.

| Mode | Semantics |
|---|---|
| `min_size` | Stop when the first loader exhausts. |
| `max_size` | Continue to the longest loader and return `None` for exhausted leaves. |
| `max_size_cycle` | Cycle shorter loaders until the longest horizon completes. |
| `sequential` | Consume one loader completely before advancing to the next. |

```python
from syntheticmind import CombinedLoader

combined: CombinedLoader = CombinedLoader(
    {
        "labeled": labeled_loader,
        "unlabeled": unlabeled_loader
    },
    mode="max_size_cycle"
)
```

Every iteration yields `(batch, batch_idx, dataloader_idx)`. Non-sequential modes reconstruct the supplied tree. Sequential mode returns the current leaf and its flattened loader index.

## Distributed and resumable behavior

The Trainer data connector replaces samplers without discarding concrete `DataLoader` subclass state. Training uses equal-work distributed sampling where required. Prediction uses an unrepeated rank-strided sampler so distributed output contains every sample exactly once. Stateful loaders are captured and restored in stable flattened order, and topology mismatches fail before any cursor mutates.

## Related components

`../core/hooks.py` defines the transfer lifecycle. `../trainer/connectors/data_connector.py` resolves sources, limits, samplers, and reload cadence. `../loops/` consumes the resulting `CombinedLoader` without acquiring data ownership.
