# syntheticmind.trainer: Lifecycle Orchestration

`trainer/` owns the complete fit, validation, test, and prediction lifecycles. `trainer.py` is the canonical implementation of the single public `Trainer` stage entrypoint, while connectors own its internal runtime composition boundaries.

| Capability | Ownership |
|---|---|
| `trainer.py` | Public stage entrypoints, top-level lifecycle sequencing, and deterministic hook dispatch. |
| `states.py` | Trainer function, stage, status, progress, interruption, and stop state. |
| `seed.py` | Python, NumPy, PyTorch, accelerator, and dataloader-worker random-state initialization. |
| `console.py` | Rank-safe lifecycle and metric rendering for the Trainer and progress callbacks. |
| `connectors/accelerator_connector.py` | Accelerator, devices, nodes, Strategy, precision, cluster environment, checkpoint I/O, and layer-synchronization compatibility. |
| `connectors/callback_connector.py` | Callback validation, defaults, model callback attachment, ordering, and checkpoint identities. |
| `connectors/checkpoint_connector.py` | Checkpoint selection, full-trajectory assembly, restoration, and resume lifecycle. |
| `connectors/data_connector.py` | Module, DataModule, dataloader, sampler, transfer, and reload attachment. |
| `connectors/logger_connector/` | Metric routing, aggregation, logger emission, callback metrics, and progress metrics. |
| `connectors/signal_connector.py` | Process-signal registration, propagation, interruption, and teardown. |

Connector names are not a quota. A connector is admitted only when one configuration family crosses several framework components and requires a single ordering and compatibility owner. Optimizer construction remains a Strategy responsibility because distributed and model-parallel setup can change when parameters become available.
