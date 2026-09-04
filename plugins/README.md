# syntheticmind.plugins: Execution Extensions

`plugins/` contains extension boundaries selected during Trainer construction.

| Module | Ownership |
|---|---|
| `io.py` | Structural checkpoint storage, local PyTorch persistence, wrapping, and asynchronous snapshots. |
| `environments.py` | Structural cluster topology plus the local environment implementation. |
| `layer_sync.py` | Structural layer synchronization plus reversible synchronized batch normalization. |
| `resolver.py` | Strict immutable resolved plugin configuration and conflict detection. |

Protocols contain contracts only. Concrete lifecycle, storage, network, and conversion behavior remains on the selected implementation.
