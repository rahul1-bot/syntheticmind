# syntheticmind.trainer.connectors.logger_connector: Experiment Observation

The logger connector owns the distinct metric views consumed by callbacks, external loggers, and progress rendering. It also owns logger selection, destination identity, hyperparameter emission, stage reset boundaries, and transfer of observable state between launched workers and the parent process.

Modules and loops remain responsible for producing and reducing metrics at their established lifecycle points. Concrete loggers remain responsible for backend persistence. The connector routes those results without introducing another logging API or changing metric computation.

`logger_connector.py` owns routing and views. `metric_accumulator.py` owns scalar validation, nested-batch size inference, weighted step-to-epoch reduction, TorchMetrics windows, and distributed scalar synchronization as one reusable reduction lifecycle. Ambiguous nested leading dimensions require an explicit `batch_size` at the logging call.

| Component | Ownership |
|---|---|
| `_LoggerConnector` | Logger selection, callback/logged/progress views, rank-zero emission, log directory, and hyperparameters. |
| `MetricAccumulator` | Scalar or TorchMetrics epoch window, weighting, reduction, synchronization, reset, and checkpoint state. |

## Metric views

The observation pipeline has distinct representations:

1. Module produces `LoggedMetric` values.
2. Loops compute step observations and update epoch accumulators.
3. `callback_metrics` retains tensors for monitoring and clients.
4. `logged_metrics` retains logger-routed tensors for inspection.
5. Per-stage progress mappings contain Python floats.
6. Concrete Logger implementations receive detached Python scalar mappings.

Sanity validation may expose temporary callback values while executing, but it does not persist ordinary logged or progress views into the real run.

## Scalar reduction

Mean observations infer batch size only when every reachable tensor agrees on its leading dimension. Nested mappings, dataclass instances, lists, and tuples are inspected. Ambiguous or tensor-free batches require `self.log(..., batch_size=...)`.

Distributed mean synchronizes numerator and denominator. It never averages already averaged rank values. Sum, minimum, and maximum synchronize their corresponding accumulated state. `rank_zero_only=True` values do not enter collectives.

TorchMetrics objects retain their own state and compute semantics and must remain one stable instance for one logged identity during an epoch.

## Logger selection

`logger=None` or `True` selects the next versioned default logger. TensorBoard is preferred when installed; CSV is the fallback. `False` disables durable logging. One Logger or an iterable of distinct instances receives every logger-visible value. `fast_dev_run` disables concrete loggers.

The primary logger controls `log_dir`; when no logger exists, Trainer's root directory is used. The selected path is broadcast so every rank observes one run identity.

## Hyperparameters

Module and DataModule initial hyperparameters emit only when each owner requested logging. Shared keys must have equal values. Internal `_class_path` metadata is removed. Emission happens once per run across sequential and multiprocessing-owned stages.

## Partial-epoch state

Scalar accumulator checkpoints validate reduction, count, extrema, sum, weighted numerator, and total weight. TorchMetrics accumulators cannot contain scalar state. Live tensors restore onto the model device only after strict validation.

## Related components

`../../../core/module.py` owns the public log API. `../../../core/metric.py` owns one observation. `../../../loops/` owns cadence. `../../../loggers/` owns backend persistence.
