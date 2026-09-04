# syntheticmind.strategies: Training Execution Policies

`Strategy` is the complete policy governing how a model participates in training, validation, testing, and prediction. It owns the model connection, wrapped-model identity, accelerator and precision collaboration, optimizer stepping, step dispatch, collective defaults, initialization context, and ordinary checkpoint behavior.

`ParallelStrategy` is the shared distributed execution policy. It owns parallel devices, cluster ranks, process-group initialization, distributed collectives, gradient synchronization, process launch, and process-scoped cleanup. DDP and model-parallel execution inherit these semantics instead of reproducing them.

```text
Strategy
├── SingleDeviceStrategy
└── ParallelStrategy
    ├── DDPStrategy
    └── ModelParallelStrategy
```

| Module | ML ownership |
|---|---|
| `strategy.py` | Universal training execution semantics and private wrapped-step forwarding. |
| `parallel.py` | Shared distributed process, rank, device, collective, and launcher semantics. |
| `single_device.py` | The named one-process, one-device execution policy. |
| `ddp.py` | PyTorch `DistributedDataParallel` wrapping and DDP-specific gradient behavior. |
| `model_parallel.py` | Device-mesh construction, model parallelization, and model-parallel state semantics. |
| `launcher.py` and `launchers/` | Internal subprocess and multiprocessing launch implementations. |

Files are admitted by ML or distributed-systems concepts, not by line-count pressure. Shared behavior remains on `Strategy` or `ParallelStrategy`; concrete strategy modules contain only behavior that changes the selected execution algorithm. Internal implementation mechanisms do not become public strategy concepts merely because they are reusable.

The existing strategy-checkpoint component and model-parallel checkpointing package remain private transitional machinery. Their final ownership is the Trainer checkpoint connector for full-trajectory coordination, the selected Strategy for topology-specific state representation, and `CheckpointIO` for storage transport.
