# syntheticmind.callbacks.model_summary: Model Inspection

The model-summary callback family owns one fit-start inspection lifecycle. `ModelSummary` selects depth and rank-zero presentation, the private report measures parameters, storage size, module modes, tensor shapes, and FLOPs, and the Rich adapter renders the same report through the optional Rich dependency.

Example-input forwarding is transactional. Attached models use the Trainer's canonical DataModule-over-Module transfer precedence and shared multi-backend random-state snapshot; detached models use their Module hooks directly. The report restores every nested module mode and Python, NumPy, torch CPU, CUDA, or MPS random state even when transfer, forwarding, or FLOP measurement fails.

| Module | Contents |
|---|---|
| `callback.py` | The public `ModelSummary` callback and plain terminal renderer. |
| `report.py` | Private layer inspection, model accounting, shape capture, FLOP measurement, and state restoration. |
| `rich.py` | The optional `RichModelSummary` rendering adapter. |

## Client integration

Trainer installs `ModelSummary(max_depth=1)` when model-summary output is
enabled and no client callback of that family is present. Configure a custom
depth by supplying the callback explicitly:

```python
from syntheticmind import Trainer
from syntheticmind.callbacks.model_summary import ModelSummary


summary: ModelSummary = ModelSummary(max_depth=2)
trainer: Trainer = Trainer(callbacks=[summary], enable_model_summary=False)
```

`max_depth=0` disables report construction, `1` selects direct children, and
`-1` selects the complete descendant hierarchy. Aggregate totals always describe
the complete model even when displayed rows are depth-limited.

Set `Module.example_input_array` when the report should include realized input
and output shapes and measured FLOPs:

```python
import torch

from syntheticmind import Module


class Encoder(Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.network: torch.nn.Linear = torch.nn.Linear(width, width)
        self.example_input_array: torch.Tensor = torch.zeros(1, width)
```

The example follows the active Trainer's precision-input conversion and batch
transfer policy when the model is attached. A detached model follows its own
transfer hooks. The forward executes under `torch.no_grad`, and the report
restores every nested module mode and supported random stream even when model
execution or FLOP measurement fails.

Lazy parameters that remain uninitialized contribute zero and produce a warning.
A representative example input should materialize them when exact parameter
accounting is required.

## Presentation extensions

Subclass `ModelSummary` and override `summarize` to change presentation without
reimplementing measurement. The method receives aligned report columns plus
aggregate parameter, storage, mode, and FLOP values. `_ModelSummaryReport` and
`_LayerSummary` remain private because clients should not couple renderers to
measurement internals.

`RichModelSummary` is imported from
`syntheticmind.callbacks.model_summary.rich` only when the optional `rich`
package is installed. The base package deliberately exports the dependency-free
plain renderer so ordinary imports do not acquire optional presentation
dependencies.
