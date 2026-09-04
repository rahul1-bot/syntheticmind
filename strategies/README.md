# syntheticmind.strategies: Execution Ownership

`strategies/` owns process initialization, model placement or wrapping, backward execution, synchronization, distributed collectives, and strategy-specific checkpoint state. Accelerators resolve and bind devices; strategies decide how those devices participate in a run.

| Module | Ownership |
|---|---|
| `strategy.py` | The structural `Strategy` contract consumed by the Trainer and loops. It contains no shared state or default behavior. |
| `parallel.py` | The smaller structural contract for cluster-backed strategies. |
| `single_device.py` | One process and one unwrapped model on one device. |
| `ddp.py` | One model replica per process under `DistributedDataParallel`. |
| `model_parallel.py` | Composable tensor and fully sharded model-parallel execution without pretending to be DDP. |
| `distributed_runtime.py` | Process-group resource ownership, rank topology, collectives, and process-scoped cleanup. |
| `forward_redirection.py` | Wrapped step dispatch through the inherited PyTorch module call path. |
| `checkpoint_connector.py` | The earned checkpoint contract and the local `CheckpointIO` implementation. |
| `model_parallel_topology.py` | Device-mesh construction, data-parallel sampler coordinates, and module mesh binding. |
| `model_parallel_checkpointing/` | Full and sharded distributed state, immutable checkpoint generations, atomic publication, recovery, and pruning. |
| `launcher.py` and `launchers/` | Subprocess and multiprocessing launch policy plus parent-worker state reconciliation. |

## Runtime Boundaries

`Trainer` wires a strategy to its model, optimizers, precision plugin, cluster environment, and checkpoint implementation. Local checkpoints delegate storage to the selected `CheckpointIO`. Model-parallel checkpoints instead own PyTorch distributed state-dict capture and restoration directly.

Distributed process groups survive individual fit, validation, test, and prediction stages because live DDP or DTensor state can outlive one stage. Failure handling and process exit release the process-scoped resource.
