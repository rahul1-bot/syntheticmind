# syntheticmind.core: Researcher-Facing Contracts

`core/` defines the model, data, optimization, hook, saving, and hyperparameter surfaces implemented or consumed by research code. Trainer orchestration belongs to `../trainer/` and is not part of the core concept.

| Module | Contents |
|---|---|
| `module.py` | `Module`, the model contract, with its live `LoggedMetric` records. |
| `datamodule.py` | `DataModule`, the data contract, and `DatasetDataModule`. |
| `hooks.py` | The `ModelHooks`, `DataHooks`, and `CheckpointHooks` lifecycle contracts. |
| `optimizer.py` | Managed optimizer access, accepted `configure_optimizers` declarations, scheduler configuration, and declaration parsing. |
| `saving.py` | Model reconstruction from canonical SyntheticMind checkpoints and explicit hyperparameter sources. |

`Trainer` binds these contracts from `../trainer/trainer.py`. Its connectors own accelerator composition, callback composition, data execution, experiment observation, checkpoint trajectories, and signal lifecycle.

`Module` defines forward computation, per-stage steps, optimizer declaration, manual optimization, and metric declaration. `DataModule` defines stage data ownership. `optimizer.py` normalizes live PyTorch optimizers and schedulers returned by the model; SyntheticMind does not invent optimizer or scheduler factory configurations that rename PyTorch algorithms.

## Related Components

The trainer executes any `Module` and `DataModule` pair without knowledge of the downstream research domain. Models, datasets, and experiments enter the harness only through these contracts.
