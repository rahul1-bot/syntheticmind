# syntheticmind.trainer: Lifecycle Orchestration

`trainer/` is the active migration target for the complete fit, validation, test, and prediction lifecycles. The public `Trainer` remains temporarily implemented in `core/trainer.py` while its runtime slices move into the connectors below; after that bounded migration, `trainer/trainer.py` becomes its canonical location. `Trainer` remains the single stage entrypoint throughout.

| Capability | Ownership |
|---|---|
| `trainer.py` | Public stage entrypoints and top-level lifecycle sequencing. |
| `states.py` | Trainer function, stage, status, progress, interruption, and stop state. |
| `call.py` | Central deterministic dispatch of module, datamodule, callback, and Strategy hooks. |
| `connectors/accelerator_connector.py` | Accelerator, devices, nodes, Strategy, precision, cluster environment, checkpoint I/O, and layer-synchronization compatibility. |
| `connectors/callback_connector.py` | Callback validation, defaults, model callback attachment, ordering, and checkpoint identities. |
| `connectors/checkpoint_connector.py` | Checkpoint selection, full-trajectory assembly, restoration, and resume lifecycle. |
| `connectors/data_connector.py` | Module, DataModule, dataloader, sampler, transfer, and reload attachment. |
| `connectors/logger_connector/` | Metric routing, aggregation, logger emission, callback metrics, and progress metrics. |
| `connectors/signal_connector.py` | Process-signal registration, propagation, interruption, and teardown. |

Connector names are not a quota. A connector is admitted only when one configuration family crosses several framework components and requires a single ordering and compatibility owner. Optimizer construction remains a Strategy responsibility because distributed and model-parallel setup can change when parameters become available.
