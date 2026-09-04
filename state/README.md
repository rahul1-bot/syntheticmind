# syntheticmind.state: Transitional Trainer State

`state/` temporarily contains only `TrainerState`, the live stage, status, epoch, global-step, interruption, and stopping state advanced by Trainer and loops.

Checkpoint payload assembly and restoration are not state-container responsibilities. They now belong to `trainer/connectors/checkpoint_connector.py`, which coordinates Strategy-specific model and optimizer state, schedulers, callbacks, DataModule state, precision state, loop and loader continuation, and random streams.

`TrainerState` will move to `trainer/states.py` when the canonical Trainer module leaves `core/`; the top-level `state/` package then disappears.
