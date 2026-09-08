# syntheticmind.strategies.model_parallel_checkpointing: Distributed State

`model_parallel_checkpointing/` owns full and sharded checkpoint representation for `ModelParallelStrategy`. It separates distributed model and optimizer state from ordinary Trainer trajectory metadata and publishes sharded artifacts through immutable generations.

| Module | Contents |
|---|---|
| `checkpoint.py` | Strategy checkpoint lifecycle, all-rank path agreement, preload, and deferred restore. |
| `checkpoint_io.py` | Full and sharded distributed state extraction, persistence, and application. |
| `sharded_store.py` | Collective shard storage, metadata coordination, failure broadcast, and pruning. |
| `asynchronous.py` | Bounded CPU staging, an isolated background communication group, publication futures, and deterministic drain. |
| `generation.py` | UUIDv7 generation identities and atomic active-generation pointer publication. |

## Full checkpoints

Full format aggregates model and optimizer state on CPU through PyTorch distributed state-dict APIs. Global rank zero publishes one ordinary checkpoint file through `TorchCheckpointIO`. Every rank still participates in distributed state extraction before rank-zero storage.

A full artifact contains distributed participant state and ordinary Trainer metadata together. Restoration validates optimizer-state cardinality before applying model state. Integer-keyed optimizer dictionaries are rekeyed by parameter name when composable FSDP2 requires it.

## Sharded checkpoints

A sharded checkpoint root contains immutable generations:

```text
checkpoint/
├── active_generation
└── generations/
    ├── <previous-uuidv7>/
    │   ├── .metadata
    │   ├── __*.distcp
    │   └── meta.pt
    └── <active-uuidv7>/
        ├── .metadata
        ├── __*.distcp
        └── meta.pt
```

PyTorch writes rank-partitioned state and `.metadata`. SyntheticMind stores Trainer trajectory metadata in `meta.pt`. The generation becomes visible only when both exist and the active-generation pointer is published.

## Publication transaction

One synchronous save follows this order:

1. Global rank zero creates a new immutable generation directory and broadcasts any failure.
2. Every rank writes its distributed state through the selected PyTorch StorageWriter.
3. Global rank zero writes Trainer metadata and broadcasts any failure.
4. Global rank zero verifies completeness and publishes the pointer.
5. Global rank zero prunes generations outside the active and previous retained set.
6. Every rank reaches the final barrier.

The previous generation remains readable until the new pointer commits. Complete but unpublished generations cannot become active by modification time. A pruning failure warns after successful publication and does not invalidate the newly committed checkpoint.

Local pointer publication writes and flushes a same-directory temporary file, fsyncs it, atomically replaces the pointer, and fsyncs the parent directory. Remote publication uses the resolved fsspec transaction.

## Asynchronous publication

`ModelParallelStrategy(checkpoint_write="asynchronous")` retains the same sharded
format and generation pointer. Preparation and Trainer metadata writing occur
on the caller, followed by an independent CPU snapshot through PyTorch's blocking
async stager. One owned worker executes DCP `save` and pointer publication on a
separate Gloo process group, never on the training collective stream.

Only one save can be outstanding. Returning from save confirms staging and
submission, not durable publication. Read, remove, subsequent-save, last-link,
and teardown boundaries fence the publication future. Rank-zero retention can
wait locally because every rank already owns its participating background worker.
Failures remain visible and preserve the previous active generation.

Teardown drains the worker before releasing its communication group; exception
cleanup drains checkpoint work before destroying training groups. Metadata and
staging still block, and retention or last-link publication can reduce overlap.
The two-rank CPU probe covers snapshot isolation, independent training
collectives, retention, and injected publication failure. CUDA/NCCL and remote
asynchronous backends require separate validation. See
[the control guide](../../docs/guides/runtime_control.md).

## Compatibility and collisions

Complete legacy flat sharded directories remain readable and migrate on their next successful save. A full file cannot overwrite a sharded directory, and a sharded generation cannot replace a full file at the same path. Format collisions fail before destructive mutation.

Model-parallel checkpoints reject external CheckpointIO plugins because their representation and publication require Strategy-owned collective semantics. Clients select full or sharded format through `ModelParallelStrategy(save_distributed_checkpoint=...)`.

## Related components

`../model_parallel.py` owns device-mesh execution. `../checkpoint.py` defines the Strategy checkpoint contract. `../../trainer/connectors/checkpoint_connector.py` assembles and restores the remaining training trajectory.
