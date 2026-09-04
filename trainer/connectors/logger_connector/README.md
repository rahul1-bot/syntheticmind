# syntheticmind.trainer.connectors.logger_connector: Experiment Observation

The logger connector owns the distinct metric views consumed by callbacks, external loggers, and progress rendering. It also owns logger selection, destination identity, hyperparameter emission, stage reset boundaries, and transfer of observable state between launched workers and the parent process.

Modules and loops remain responsible for producing and reducing metrics at their established lifecycle points. Concrete loggers remain responsible for backend persistence. The connector routes those results without introducing another logging API or changing metric computation.

`logger_connector.py` owns routing and views. `metric_accumulator.py` owns scalar validation, nested-batch size inference, weighted step-to-epoch reduction, TorchMetrics windows, and distributed scalar synchronization as one reusable reduction lifecycle. Ambiguous nested leading dimensions require an explicit `batch_size` at the logging call.
