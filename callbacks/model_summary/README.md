# syntheticmind.callbacks.model_summary: Model Inspection

The model-summary callback family owns one fit-start inspection lifecycle. `ModelSummary` selects depth and rank-zero presentation, the private report measures parameters, storage size, module modes, tensor shapes, and FLOPs, and the Rich adapter renders the same report through the optional Rich dependency.

Example-input forwarding is transactional. The report restores every nested module mode and the Python, NumPy, torch CPU, CUDA, or MPS random state even when transfer hooks, model forwarding, or FLOP measurement fails.

| Module | Contents |
|---|---|
| `callback.py` | The public `ModelSummary` callback and plain terminal renderer. |
| `report.py` | Private layer inspection, model accounting, shape capture, FLOP measurement, and state restoration. |
| `rich.py` | The optional `RichModelSummary` rendering adapter. |
