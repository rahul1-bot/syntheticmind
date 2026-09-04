# SyntheticMind Documentation

This directory explains contracts that cross semantic package boundaries. A
package README remains authoritative for the concept it owns; these documents
describe how those concepts compose into one training system.

## Architecture

| Document | Scope |
|---|---|
| [`architecture/framework_topology.md`](architecture/framework_topology.md) | Public surface, dependency direction, component ownership, and admission rules. |
| [`architecture/training_lifecycle.md`](architecture/training_lifecycle.md) | Fit, evaluation, prediction, hook order, durable boundaries, and teardown. |
| [`architecture/checkpoint_trajectory.md`](architecture/checkpoint_trajectory.md) | Schema-v2 state, save and restore ordering, exact continuation, and artifact publication. |
| [`architecture/distributed_execution.md`](architecture/distributed_execution.md) | Launch modes, rank and device ownership, data versus tensor parallel axes, collectives, and failure behavior. |

## Client guides

| Document | Scope |
|---|---|
| [`guides/model_and_data_integration.md`](guides/model_and_data_integration.md) | Building a Module and DataModule that enter the complete Trainer lifecycle. |
| [`guides/trainer_configuration.md`](guides/trainer_configuration.md) | Configuration domains, resolved policies, execution recipes, and conflicting settings. |
| [`guides/extension_points.md`](guides/extension_points.md) | Choosing and implementing the correct callback, logger, profiler, accelerator, Strategy, or plugin boundary. |

## Project documentation policy

[`project_guidelines/documentation_style.md`](project_guidelines/documentation_style.md)
defines the authorized VOCODE line-comment system used throughout the Python
source. The repository contains no generated API dump. Documentation follows
semantic ownership: source comments explain local invariants, package READMEs
explain one concept family, and this directory explains cross-package behavior.

Historical decisions and verified completion state remain indexed by `GHOST.md`,
`LYRA.md`, `memory.md`, and `progress.md`. Those ledgers are not substitutes for
current client or architecture documentation.
