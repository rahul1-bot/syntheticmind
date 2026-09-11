# syntheticmind.rng: Stochastic Streams

`rng/` owns the process-wide random streams that determine a SyntheticMind run's stochastic trajectory. Clients normally interact through `Trainer(seed=...)`; checkpointing, launchers, and model inspection consume the internal stream owner directly.

| Module | Contents |
|---|---|
| `streams.py` | Seeding plus immutable capture and restoration of Python, NumPy, torch CPU, CUDA, and MPS random streams. |

## Stream ownership

One `_Randomness` instance is created for each Trainer from the selected accelerator device families. It applies an explicit seed consistently, captures the active streams into `_RandomStateSnapshot`, and restores them without consulting inactive accelerator runtimes.

Capturing only active backends is a correctness boundary. Initializing MPS or CUDA solely to inspect a generator can violate fork safety, consume device resources, or alter process initialization order.

```python
from syntheticmind import Trainer

trainer: Trainer = Trainer(
    accelerator="auto",
    max_epochs=20,
    seed=731
)
```

The seed controls Python, NumPy, torch CPU, and the selected accelerator streams. `PYTHONHASHSEED` is exported for child interpreters launched afterward; the current interpreter's hash seed was fixed at process startup.

## Checkpoint continuation

A resumable checkpoint stores the current stream snapshot and, when fitting stops inside an epoch, the epoch-start snapshot required by exact loader replay. Restoration occurs at the lifecycle boundary selected by the training loop so model initialization, dataloader progress, and metric windows consume randomness in the same order as the uninterrupted trajectory.

The snapshot serializes torch generator tensors as integer tuples. This keeps the Pydantic transport schema strict and compatible with process transfer while reconstruction returns the exact `uint8` format torch expects.

## Observation isolation

Model-summary forwarding is observational and must not change the run. It captures the shared streams before the example input executes and restores them in `finally`, together with every nested module training mode.

## Randomness versus determinism

Controlled streams and deterministic kernels are separate mechanisms. A seeded run may still use nondeterministic kernels, and deterministic execution may intentionally consume different seeds. `../trainer/determinism.py` owns algorithm selection, warning mode, cuDNN flags, benchmarking, and CUBLAS policy.

## Related components

`../trainer/trainer.py` constructs and seeds the owner. `../trainer/connectors/checkpoint_connector.py` persists it. `../strategies/launchers/state_transfer.py` transports it between worker and parent processes. `../callbacks/model_summary/report.py` borrows it transactionally.
