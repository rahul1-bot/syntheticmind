# syntheticmind.loggers: Experiment Records

The Trainer logger connector owns metric reduction, scalar conversion, rank-zero routing, default selection, and multi-logger fan-out. Logger implementations receive completed scalar observations and portable hyperparameters; they own only destination identity, persistence, buffering, and finalization.

| Module | Contents |
|---|---|
| `logger.py` | Stateful logger inheritance contract, run-directory identity, metric validation, and portable hyperparameter normalization. |
| `csv_logger.py` | Resumable CSV metrics with transactional schema widening plus deterministic JSON hyperparameters on local or fsspec storage. |
| `tensorboard.py` | Local TensorBoard scalar events and hyperparameter records through a lazily acquired writer. |

Disabled logging is represented by an empty logger collection. No fake experiment accepts arbitrary attributes or silently absorbs misspelled backend calls.

## Configuring experiment records

Pass one logger or an iterable of distinct loggers to Trainer. The connector
fans each completed observation out in configuration order and exposes the first
logger's directory as `trainer.log_dir`.

```python
from syntheticmind import Trainer
from syntheticmind.loggers import CSVLogger, TensorBoardLogger


csv_logger: CSVLogger = CSVLogger(
    save_dir="artifacts/logs",
    name="ablation",
    version=3
)
tensorboard_logger: TensorBoardLogger = TensorBoardLogger(
    save_dir="artifacts/events",
    name="ablation",
    version=3
)

trainer: Trainer = Trainer(logger=[csv_logger, tensorboard_logger])
```

Integer versions produce a `version_N` segment. String versions are used as one
literal segment. Names and string versions cannot contain separators or parent
traversal. `version=None` leaves the version level absent; automatic version
discovery is not inferred by Logger itself.

`logger=False` disables durable logging. `logger=None` or `True` asks Trainer for
its default logger policy. Fast-development and barebones modes may suppress or
reject logging according to their documented runtime contract.

## Metric ownership

Client models produce observations with `Module.log` and `Module.log_dict`.
Loops compute step or epoch values, Strategy performs requested reduction, and
the logger connector converts finalized scalar tensors to Python floats. A
Logger therefore receives `dict[str, float]` plus one non-negative integer step;
it must not reach back into Module buffers or perform another distributed
reduction.

Only global zero persists ordinary experiment records. `trainer.callback_metrics`
and the other in-memory metric views retain their own rank semantics and should
not be treated as backend files.

## Hyperparameters

When autologging is enabled, Trainer merges captured Module and DataModule
constructor hyperparameters and emits them once. Supported values normalize to
portable JSON-compatible data: scalar Python and NumPy values, scalar tensors,
bytes, dates, datetimes, durations, paths, devices, dtypes, namespaces, and
nested mappings or containers.

Live modules, vector tensors or arrays, non-finite floats, unsupported objects,
and invalid names fail explicitly. Sets are stored in deterministic order.
Backends do not fall back to `repr`, because a process-specific representation
is not experiment metadata.

## CSV semantics

`CSVLogger` writes `metrics.csv` and `hparams.json`. The first metric column is
always `step`; clients cannot reuse that name as a metric. When a later
observation introduces a new metric, the table is widened transactionally and
prior rows receive empty cells for the new field.

Local replacements use a synchronized same-directory temporary artifact before
atomic publication. fsspec destinations use their transaction boundary. A new
logger instance can continue an existing compatible table and validates its
header before appending.

## TensorBoard semantics

`TensorBoardLogger` requires local storage. The SummaryWriter dependency is
loaded only when a metric or hyperparameter is first emitted. Nested portable
hyperparameter values become deterministic JSON strings because TensorBoard's
hyperparameter surface accepts only scalars.

## Custom logger integration

A client backend subclasses `Logger`, calls `super().__init__`, and implements
the three lifecycle operations. Reuse `_validated_metrics` and
`_normalize_hyperparameters` rather than inventing a second input contract.

```python
import json
from pathlib import Path
from typing import override

from syntheticmind.loggers import Logger
from syntheticmind.primitives.types import HyperparameterDict


class MemoryLogger(Logger):
    @override
    def __init__(
        self,
        save_dir: str | Path,
        name: str = "memory",
        version: int | str | None = None
    ) -> None:
        super().__init__(save_dir=save_dir, name=name, version=version)
        self.metric_records: list[tuple[int, dict[str, float]]] = []
        self.hyperparameters: str | None = None

    @override
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        values: dict[str, float] = self._validated_metrics(metrics, step)
        self.metric_records.append((step, values))

    @override
    def log_hyperparams(self, params: HyperparameterDict) -> None:
        self.hyperparameters = json.dumps(
            self._normalize_hyperparameters(params),
            allow_nan=False,
            sort_keys=True
        )

    @override
    def finalize(self) -> None:
        return None
```

Production backends replace the in-memory records with their owned transport.
`finalize` must be idempotent because Trainer attempts all logger cleanup even
when another lifecycle component fails.

## Related components

`../trainer/connectors/logger_connector/` owns reduction, routing, fan-out, and
rank policy. `../core/metric.py` owns logged observations. `../storage/` owns
local and fsspec path behavior. Monitoring callbacks publish through the same
trainer connector rather than writing backend files directly.
