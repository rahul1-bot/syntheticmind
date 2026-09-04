# syntheticmind.utilities: Transitional Shared Infrastructure

`utilities/` is being dissolved into semantic owners. A module remains here only until its execution slice is reconstructed; the package is not an approved destination for new behavior.

| Module | Contents |
|---|---|
| `seed.py` | `SeedManager`, deterministic seeding across Python, NumPy, and PyTorch, including worker seeding for dataloaders. The study's recorded training seed and its evaluation seeds pass through this module. |
| `checkpoint.py` | Transitional low-level checkpoint file persistence consumed by checkpoint I/O implementations. |
| `data_transfer.py` | `move_data_to_device`, recursive device transfer for tensors and arbitrarily nested containers. |
| `distributed.py` | `DistributedUtils` and the rank helpers (`get_rank`, `get_local_rank`, `get_world_size`, `is_distributed`, `is_rank_zero`, `barrier`, `all_reduce`, and `rank_zero_only`), which degrade gracefully to single-process semantics. |
| `metric_accumulator.py` | `MetricAccumulator`, the step-level and epoch-level metric aggregation behind the `Module.log` contract. |
| `compile.py` | `apply_compile` and `apply_channels_last`, model compilation and memory-format helpers. |
| `exceptions.py` | `FrameworkError` and its family: `MisconfigurationError`, `TrainingInterruptedError`, `CheckpointError`, and `DistributedError`, the framework's explicit error vocabulary. |
| `types.py` | `DeviceTransferable`, the structural type contract used by device transfer. |

New optimizer and scheduler construction utilities are prohibited. Research modules return real PyTorch optimizers and schedulers through `configure_optimizers`; core normalizes that declaration and loops execute it.
