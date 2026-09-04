# syntheticmind.plugins: Execution Extensions

`plugins/` contains extension boundaries selected during Trainer construction.

| Module | Ownership |
|---|---|
| `checkpoint_io.py` | Structural checkpoint storage, local PyTorch persistence, wrapping, and asynchronous snapshots. |
| `environments.py` | Structural cluster topology plus the local environment implementation. |
| `layer_sync.py` | Structural layer synchronization plus reversible synchronized batch normalization. |
| `precision/` | Full, true-half, mixed, and Transformer Engine numerical execution policies. |

Protocols contain contracts only. Concrete lifecycle, storage, network, and conversion behavior remains on the selected implementation. Plugin categorization and cross-plugin compatibility belong to the Trainer accelerator connector rather than to a standalone resolver or temporary runtime-data model.
