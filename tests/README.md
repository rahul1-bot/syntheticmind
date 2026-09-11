# syntheticmind.tests: The Regression Suite

`tests/` is the committed executable evidence for the framework. Every module
pins one client-visible contract family against small CPU-only models so a
change that breaks resumption, optimization progress, presentation, or metric
records fails here before it reaches a research project.

| Module | Contract family |
|---|---|
| `conftest.py` | Source-checkout import path and deterministic terminal environment. |
| `support.py` | The seeded regression model, data builders, and stream doubles. |
| `test_optimization.py` | Skipped batches, accumulation windows, step ceilings, scheduler cadence. |
| `test_checkpointing.py` | Exact continuation, ranked retention, asynchronous transport failure. |
| `test_lifecycle.py` | Fit, validate, test, predict, early stopping, multiple loaders, CSV records. |
| `test_console.py` | Renderer formatting, styling, live line, records, diagnostics capture. |
| `test_progress_bar.py` | Stage headers, epoch and failure records, interactive output, configuration. |

## Running

```bash
python -m pip install -e ".[test]"
python -m pytest
```

A source checkout whose directory is named `syntheticmind` also runs without
installation. The suite sets `NO_COLOR` and a fixed `COLUMNS` width so console
assertions are identical on developer terminals and continuous-integration
runners. No linter or static type checker participates; the verification policy
in `../GHOST.md` admits executable evidence only.

## Adding evidence

A new capability or a fixed defect adds one test that reads public Trainer,
Module, callback, or renderer views. Tests use test classes with typed methods,
follow the source comment conventions in
`../docs/project_guidelines/documentation_style.md`, and must complete in
seconds on CPU. Multi-process launcher tests carry the `distributed` marker so
they can be deselected where process spawning is unavailable.
