# syntheticmind.utilities: Transitional Shared Infrastructure

`utilities/` is being dissolved into semantic owners. A module remains here only until its execution slice is reconstructed; the package is not an approved destination for new behavior.

| Module | Contents |
|---|---|
| `distributed.py` | Transitional rank and collective operations pending Strategy and callback ownership reconstruction. |
| `combined_loader.py` | Transitional nested-loader composition pending ownership by the training data lifecycle. |
| `model_helpers.py` | Transitional hook inspection and module-mode behavior pending ownership by their lifecycle concepts. |
| `model_summary.py` | Transitional model inspection behavior pending ownership by the model-summary callback family. |

New optimizer and scheduler construction utilities are prohibited. Research modules return real PyTorch optimizers and schedulers through `configure_optimizers`; core normalizes that declaration and loops execute it.
