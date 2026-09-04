# syntheticmind.strategies: Training Execution Policies

`Strategy` is the complete policy governing how a model participates in training, validation, testing, and prediction. It owns the model connection, wrapped-model identity, accelerator and precision collaboration, optimizer stepping, step dispatch, collective defaults, initialization context, and ordinary checkpoint behavior.

`ParallelStrategy` is the shared distributed execution policy. It owns parallel devices, cluster ranks, process-group initialization, tensor reduction, distributed collectives, gradient synchronization, process launch, and process-scoped cleanup. DDP and model-parallel execution inherit these semantics instead of reproducing them. Mean, sum, minimum, and maximum metric reductions therefore execute through the selected Strategy rather than ambient process-global helpers.

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
| `model_initialization.py` | Strategy-aware empty initialization, materialization, and parameter-reset interception. |
| `launchers/` | Internal process launch policy, subprocess execution, multiprocessing execution, worker state transfer, environment reconstruction, and failure observation. |

Files are admitted by ML or distributed-systems concepts, not by line-count pressure. Shared behavior remains on `Strategy` or `ParallelStrategy`; concrete strategy modules contain only behavior that changes the selected execution algorithm. Internal implementation mechanisms do not become public strategy concepts merely because they are reusable.

`checkpoint.py` defines the private Strategy checkpoint contract and ordinary single-file state implementation. The model-parallel checkpointing package supplies the topology-specific distributed implementation. Full-trajectory assembly, path selection, restoration order, and resume state belong exclusively to the Trainer checkpoint connector; `CheckpointIO` owns storage transport.
