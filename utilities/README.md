# syntheticmind.utilities: Transitional Shared Infrastructure

`utilities/` is being dissolved into semantic owners. A module remains here only until its execution slice is reconstructed; the package is not an approved destination for new behavior.

| Module | Contents |
|---|---|
| `model_summary.py` | Transitional model inspection behavior pending ownership by the model-summary callback family. |

New optimizer and scheduler construction utilities are prohibited. Research modules return real PyTorch optimizers and schedulers through `configure_optimizers`; core normalizes that declaration and loops execute it.
