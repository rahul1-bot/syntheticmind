# Framework Topology

SyntheticMind is organized around machine-learning concepts and runtime
ownership, not generic software layers or line-count splitting. A module exists
when it owns a coherent state machine, algorithm, resource, failure boundary, or
public extension contract. Files do not multiply merely to make implementations
look small.

## Stable public surface

The package root exports five framework entry concepts:

| Public type | Meaning |
|---|---|
| `Trainer` | Complete fit, validation, test, prediction, continuation, and teardown orchestration. |
| `Module` | Trainable client model contract built on `torch.nn.Module`. |
| `DataModule` | Client data preparation, construction, transfer, persistence, and teardown lifecycle. |
| `DatasetDataModule` | Direct dataset-backed DataModule implementation. |
| `CombinedLoader` | Nested multiple-dataloader composition and continuation boundary. |

Callbacks, loggers, profilers, Strategies, accelerators, and plugins are imported
from their semantic packages. This keeps optional dependencies and extension
ownership visible rather than flattening the entire framework into one root
namespace.

## Dependency direction

The conceptual dependency flow is inward to vocabulary and outward to
orchestration:

1. `primitives/` defines recursive value relationships, structural protocols,
   and framework failure categories. It owns no algorithm or runtime resource.
2. `storage/` and `rng/` own external artifact resources and stochastic streams.
3. `core/` and `data/` define the client-facing model and data contracts.
4. `accelerators/`, `plugins/`, and `strategies/` define device, numerical,
   process, collective, wrapping, and checkpoint-representation policy.
5. `loops/` own iterative ML execution and continuation coordinates.
6. `callbacks/`, `loggers/`, and `profilers/` observe lifecycle, persist
   experiment records, and measure runtime actions without taking over loops.
7. `trainer/` composes every selected concept and owns public stage transactions.

The Trainer is an orchestrator, not a universal implementation owner. It may
sequence checkpoint assembly, but Strategy owns topology-specific state and
CheckpointIO owns transport. It may route metrics, but Module owns production,
loops own cadence, Strategy owns reduction, and Logger owns persistence.

## Ownership matrix

| Concern | Sole or primary owner | Explicit non-owners |
|---|---|---|
| Model computation and research steps | `Module` | Callback, Logger, Profiler |
| Dataset preparation and loader construction | `DataModule` or Module data hooks | Loop, Strategy |
| Nested loader policy | `CombinedLoader` | Trainer entrypoint |
| Batch transfer precedence | Trainer transaction using DataModule or Module hooks | Individual loops |
| Epoch and batch iteration | `loops/` | Callback, Module |
| Optimizer and scheduler declaration | `Module.configure_optimizers` | Callback, Trainer constructor |
| Managed optimizer cadence | Training epoch and optimization loops | Logger, DataModule |
| Device discovery and binding | `Accelerator` | Strategy, Callback |
| Model placement, wrapping, collectives, and process topology | `Strategy` | Accelerator, Logger |
| Numerical execution | Precision plugin | Callback, DataModule |
| Full checkpoint trajectory | Trainer checkpoint connector | ModelCheckpoint, CheckpointIO |
| Model-parallel checkpoint representation | `ModelParallelStrategy` checkpoint owner | ModelCheckpoint |
| Artifact transport | `CheckpointIO` | Strategy selection, retention callback |
| Checkpoint cadence and retention | `ModelCheckpoint` | Trainer connector, CheckpointIO |
| Metric observation | `Module.log` and `LoggedMetric` | Logger backend |
| Metric aggregation and views | loops and logger connector | Callback implementation |
| Experiment persistence | `Logger` | Metric accumulator, Strategy |
| Runtime measurement | `Profiler` | Callback, Logger |
| Random streams | Trainer-owned `_Randomness` | Model-summary forward pass |

These boundaries prevent double ownership. A custom checkpoint backend does not
choose top-k retention. A custom callback does not perform a second distributed
metric reduction. A custom Accelerator does not launch processes. A custom
Strategy does not invent experiment logging.

## Composition boundaries

Trainer uses connectors only for configuration families that cross several
runtime concepts and require one compatibility or ordering decision:

| Connector | Vertical boundary |
|---|---|
| Accelerator | Accelerator, devices, nodes, Strategy, precision, environment, checkpoint transport, and layer synchronization. |
| Callback | Defaults, model-provided callbacks, ordering, and state identities. |
| Checkpoint | Path selection, schema validation, trajectory assembly, and restoration ordering. |
| Data | Source precedence, loader construction, limits, sampler injection, reload, and continuation. |
| Logger | Metric accumulation, consumer views, rank policy, fan-out, and hyperparameters. |
| Signal | SIGTERM registration, propagation, chaining, and restoration. |

A connector is private. It does not become an alternate client API and is not a
quota inviting one file per constructor argument.

## Inheritance and structural contracts

Stateful ML behavior uses regular inheritance where subclasses reuse real base
implementation: Module, DataModule, Strategy, Callback, Logger, Profiler, and
precision policies. `@override` marks genuine method replacements.

Protocols are reserved for narrow external substitution boundaries whose
implementations do not share framework state, such as Accelerator,
CheckpointIO, ClusterEnvironment, LayerSync, and device-transferable client
values. A Protocol is not used to hide circular imports or to manufacture a
second interface for an existing class hierarchy.

## Concept-first admission test

Before adding a class, module, folder, or public name, answer all of the
following:

1. Which ML or distributed-systems concept does it represent?
2. What state, algorithm, resource, lifecycle, or failure boundary does it own?
3. Which component constructs it and which independent components consume it?
4. Why does the current owner become less coherent if this behavior remains
   there?
5. Is the new boundary a recognized client extension or only a forwarding layer?
6. Does the name match established PyTorch research vocabulary?

If the answer is only reuse, file length, tidiness, or future possibility, the
new abstraction is rejected. Shared code stays on the concept that owns its
meaning. A real concept may exceed a generic size heuristic when splitting it
would create fake managers, stores, factories, helpers, or pass-through objects.

## Non-goals

SyntheticMind does not promise source compatibility with PyTorch Lightning and
does not copy vendor branding, internal decomposition, historical aliases, or
unsupported backends. Established ML vocabulary and lifecycle meaning remain
the behavioral reference where SyntheticMind implements the same concept.

The current framework does not claim public release readiness, universal
hardware coverage, Fabric parity, XLA execution, or support for every upstream
Trainer option. Capabilities enter only with an owned runtime path and executable
evidence.

## Navigation

`training_lifecycle.md` explains how these owners execute together.
`checkpoint_trajectory.md` explains persistence and continuation.
`distributed_execution.md` explains ranks, launch, device axes, and collectives.
Package-level detail remains in each top-level subpackage README.
