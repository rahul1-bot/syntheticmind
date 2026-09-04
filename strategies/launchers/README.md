# syntheticmind.strategies.launchers: Process Ownership

`launchers/` creates local distributed workers when the selected `ParallelStrategy` is not already running inside an externally created process. It also reconciles worker-owned execution state into multiprocessing parents and propagates process failure and termination.

| Module | Contents |
|---|---|
| `launcher.py` | Start-method selection and common launch policy. |
| `subprocess.py` | Importable module or script re-execution for strategy `ddp`. |
| `multiprocessing.py` | Spawn, fork, and forkserver worker creation and parent reconciliation. |
| `environment.py` | Reversible distributed environment construction and rank specialization. |
| `contracts.py` | Strict stage, process-global, and worker-output transport schemas. |
| `state_transfer.py` | Complete worker-to-parent state capture and restoration. |
| `process_observer.py` | Nonzero-rank subprocess failure detection and job termination. |
| `protocols.py` | Narrow Strategy topology required by launchers. |

## Start methods

| Method | Execution model | Primary constraint |
|---|---|---|
| `popen` | Re-executes the current module or script for nonzero ranks. | Requires an importable module or real script path. |
| `spawn` | Pickles Trainer and stage state into fresh interpreters. | Requires guarded program entry and complete pickleability. |
| `fork` | Forks the current interpreter. | Interactive-compatible but prohibited with CUDA. |
| `forkserver` | Requests workers through a clean fork server. | Requires guarded entry and complete pickleability. |

Use `strategy="ddp"` for ordinary scripts and modules. Use `ddp_fork` only where interactive execution requires it and the active backend is fork-safe. Multiprocessing prediction requires `return_predictions=False` because arbitrary output collections are not transferred to the parent.

## External launch

When `ClusterEnvironment.creates_processes_externally` is true, ProcessLauncher creates no children. Each torchrun or scheduler worker enters the Trainer stage directly using its exported rank and world state. Multi-node execution belongs to this lane; SyntheticMind local launch is intentionally restricted to one node.

## Worker state

Spawn and fork workers own the live stage. Local rank zero publishes one `_WorkerOutput` carrying model, Trainer, callback, DataModule, precision, metrics, data totals, loader cursor, optimizer, scheduler, signal, and stochastic state. The dormant parent restores this snapshot rather than inventing live worker objects.

CPU forked parameters and buffers are cloned before worker mutation so shared storage cannot alter the parent prematurely. Metric tensors detach and move to CPU because autograd graphs cannot cross process ownership.

## Failure and signals

The subprocess observer polls nonzero ranks while rank zero may be blocked in a collective. The first nonzero exit terminates sibling workers and signals the parent. SIGTERM remains SIGTERM semantics; unrelated failures escalate to kill where available. Environment variables, torch thread count, observers, process handles, and temporary state are restored or cleared during every failure path.

## Related components

`../parallel.py` owns process-group and collective behavior. `../../trainer/connectors/signal_connector.py` owns signal registration. `../../rng/streams.py` and `../../trainer/determinism.py` own process-global state transported by launchers.
