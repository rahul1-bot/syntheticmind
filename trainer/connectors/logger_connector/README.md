# syntheticmind.trainer.connectors.logger_connector: Experiment Observation

The logger connector owns the distinct metric views consumed by callbacks, external loggers, and progress rendering. It also owns logger selection, destination identity, hyperparameter emission, stage reset boundaries, and transfer of observable state between launched workers and the parent process.

Modules and loops remain responsible for producing and reducing metrics at their established lifecycle points. Concrete loggers remain responsible for backend persistence. The connector routes those results without introducing another logging API or changing metric computation.

`logger_connector.py` owns routing and views. `metric_accumulator.py` owns weighted step-to-epoch reduction and distributed scalar synchronization as the logger lifecycle's reusable reduction mechanism.
