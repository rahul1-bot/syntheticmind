# syntheticmind.loggers: Experiment Records

The Trainer logger connector owns metric reduction, scalar conversion, rank-zero routing, default selection, and multi-logger fan-out. Logger implementations receive completed scalar observations and portable hyperparameters; they own only destination identity, persistence, buffering, and finalization.

| Module | Contents |
|---|---|
| `logger.py` | Stateful logger inheritance contract, run-directory identity, metric validation, and portable hyperparameter normalization. |
| `csv_logger.py` | Resumable CSV metrics with transactional schema widening plus deterministic JSON hyperparameters on local or fsspec storage. |
| `tensorboard.py` | Local TensorBoard scalar events and hyperparameter records through a lazily acquired writer. |

Disabled logging is represented by an empty logger collection. No fake experiment accepts arbitrary attributes or silently absorbs misspelled backend calls.
