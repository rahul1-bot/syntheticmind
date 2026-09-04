# syntheticmind.utilities: Transitional Shared Infrastructure

`utilities/` is being dissolved into semantic owners. A module remains here only until its execution slice is reconstructed; the package is not an approved destination for new behavior.

| Module | Contents |
|---|---|
| `checkpoint.py` | Transitional low-level checkpoint file persistence consumed by checkpoint I/O implementations. |
| `distributed.py` | Transitional rank and collective operations pending Strategy and callback ownership reconstruction. |
| `exceptions.py` | `FrameworkError` and its family: `MisconfigurationError`, `TrainingInterruptedError`, `CheckpointError`, and `DistributedError`, the framework's explicit error vocabulary. |
| `types.py` | `DeviceTransferable`, the structural type contract used by device transfer. |

New optimizer and scheduler construction utilities are prohibited. Research modules return real PyTorch optimizers and schedulers through `configure_optimizers`; core normalizes that declaration and loops execute it.
