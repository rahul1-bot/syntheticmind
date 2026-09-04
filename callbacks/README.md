# syntheticmind.callbacks: The Lifecycle Callbacks

`callbacks/` extends the trainer through a uniform hook interface. Callbacks observe and act on a run without living inside model or loop code, and callback state participates in checkpoints, so resumable runs restore callback progress alongside model state.

| Module | Contents |
|---|---|
| `callback.py` | `Callback`, the hook surface every callback selectively overrides. |
| `checkpointing/` | The `ModelCheckpoint` lifecycle, trigger schedule, scalar top-k ranking, and artifact path policy. |
| `early_stopping.py` | `EarlyStopping`, which stops a run on a monitored metric with patience. |
| `timer.py` | `TimerCallback`, which enforces wall-time budgets. |
| `finite_value_guard.py` | `FiniteValueGuard`, which rejects non-finite loss before backward and optional non-finite gradients before optimizer stepping. |
| `weight_averaging.py` | `EMAWeightAveraging`, which tracks resumable exponential moving averages and evaluates under averaged parameters. |
| `lr_monitor.py` | `LearningRateMonitor`, which records per-group learning rates. |
| `progress_bar.py` | `ProgressBarCallback`, which reports stage progress. |
| `model_summary/` | The `ModelSummary` callback, layer and model inspection report, and optional Rich renderer. |
| `throughput_monitor.py` | `ThroughputMonitor`, which records synchronized window and epoch throughput over the Strategy's data-parallel sample axis. |
| `device_stats_monitor.py` | `DeviceStatsMonitor`, which records device utilization. |

## Related Components

Callbacks are composed by `../trainer/connectors/callback_connector.py`, dispatched by `../trainer/trainer.py`, and persisted by the Trainer checkpoint connector.
