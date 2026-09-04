# syntheticmind.plugins: Execution Extensions

`plugins/` contains extension boundaries selected during Trainer construction.

| Module | Ownership |
|---|---|
| `checkpoint_io.py` | Structural checkpoint storage, local PyTorch persistence, wrapping, and asynchronous snapshots. |
| `environments.py` | Structural cluster topology plus the local environment implementation. |
| `layer_sync.py` | Structural layer synchronization plus reversible synchronized batch normalization. |
| `precision/` | Full, true-half, mixed, and Transformer Engine numerical execution policies. |

Protocols contain contracts only. Concrete lifecycle, storage, network, and conversion behavior remains on the selected implementation. Plugin categorization and cross-plugin compatibility belong to the Trainer accelerator connector rather than to a standalone resolver or temporary runtime-data model.

## Client configuration

Supply one plugin or a list through Trainer. The connector categorizes each structural contract, rejects duplicate families, and verifies compatibility with the selected Accelerator and Strategy.

```python
from syntheticmind import Trainer
from syntheticmind.plugins import AsyncCheckpointIO, TorchSyncBatchNorm

trainer: Trainer = Trainer(
    accelerator="cuda",
    strategy="ddp",
    devices=4,
    plugins=[
        AsyncCheckpointIO(),
        TorchSyncBatchNorm()
    ]
)
```

`AsyncCheckpointIO` wraps the resolved checkpoint transport and persists caller-thread snapshots through one ordered worker. `TorchSyncBatchNorm` is valid only where distributed model wrapping can synchronize the converted layers.

## Custom checkpoint storage

A custom `CheckpointIO` implementation owns serialization transport, storage-specific options, resources, and teardown. It receives a complete `CheckpointDict`; it does not decide which participants enter the payload, when a save is due, which artifact ranks highest, or how model-parallel state is represented.

Local default persistence publishes through a same-directory atomic replacement. Remote default persistence serializes into a seekable temporary file and copies within the fsspec transaction. A custom backend must state and enforce its own durability contract.

## Custom cluster environments

A `ClusterEnvironment` provides rank, world size, node rank, local rank, rendezvous address and port, validation, and whether workers were created externally. Strategy owns process-group creation, collective order, and cleanup. Returning `creates_processes_externally=True` prevents SyntheticMind from launching another local worker set.

## Layer synchronization

`LayerSync` is a reversible structural transformation around distributed fitting. `TorchSyncBatchNorm` converts before DDP wrapping and restores ordinary batch-normalization modules afterward while preserving parameters, buffers, modes, children, and quantization configuration.

## Precision

Precision plugins own numerical conversion, forward contexts, backward preparation, gradient scaling, optimizer stepping, clipping, and checkpoint state. See [`precision/README.md`](precision/README.md) for the complete policy matrix.

## Related components

`../trainer/connectors/accelerator_connector.py` composes these plugins. `../strategies/` consumes cluster, precision, checkpoint, and layer policies without transferring their ownership.
