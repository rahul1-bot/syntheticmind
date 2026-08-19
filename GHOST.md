| Strategic Direction | SyntheticMind Harness Hardening | Date: 19 August 2026 | Time: 04:31 AM | Name: Ghost |

    1. SyntheticMind is a private machine-learning harness created from the ground up for Rahul's FAU study work, independent research, experimentation, and the enjoyment of controlling the complete workflow directly.
    2. The project exists to provide end-to-end ownership of training, validation, testing, prediction, optimization, checkpointing, logging, callbacks, devices, precision, and distributed execution without surrendering the runtime to a larger framework.
    3. The harness is intentionally incomplete. Its current status must not be represented as production-ready, release-ready, or capability-complete.
    4. The active phase is code hardening: confirm runtime correctness, close test gaps, strengthen invariants, and add only capabilities justified by real machine-learning workflows.
    5. Public-release ceremony, promotional work, and documentation expansion are outside the current mission unless Rahul explicitly reopens them. Code behavior and operational capability control the work queue.
    6. PyTorch Lightning is a comparative reference for established Trainer terminology, parameter semantics, lifecycle behavior, and failure handling; SyntheticMind is not required to reproduce every Lightning feature.

| Architecture Decision | Public Terminology and Python Standards | Date: 19 August 2026 | Time: 04:31 AM | Name: Ghost |

    1. Established machine-learning terminology is a compatibility boundary. SyntheticMind must use recognizable industry-standard names and meanings when a stable convention already exists.
    2. New public terms are prohibited when they merely rename an established concept. A project-specific term is acceptable only when it represents genuinely different semantics and that difference is enforced by code and tests.
    3. The generic Python engineering rules governing this codebase are stored at `/Users/rahulsawhney/LocalCode/AI/syntheticmind/docs/references/guidelines/code-guidelines.md`.
    4. Public API terminology follows machine-learning conventions, while internal structure follows the generic Python guidelines. Any conflict must be resolved explicitly rather than allowing accidental architectural drift.
    5. The local PyTorch Lightning checkout under `docs/references/code/pytorch_lightning/pytorch-lightning` is read-only comparative material and is excluded from SyntheticMind's Git history with the complete `docs/` tree.

| Repository Policy | Main and Development Branches | Date: 19 August 2026 | Time: 04:31 AM | Name: Ghost |

    1. The GitHub repository is private and uses `origin` as the canonical remote.
    2. `main` holds the stable project baseline, including all four root ledgers. Direct feature and hardening work does not begin on `main`.
    3. `dev` is the default working branch for Trainer hardening and future implementation work.
    4. Every accepted development commit must be pushed to `origin/dev` before the work is treated as synchronized or complete.
    5. Local `dev` and `origin/dev` must remain at the same commit after each completed commit-and-push cycle. Divergence is a defect to resolve before beginning unrelated work.
    6. Changes reach `main` only through an intentional stabilization decision after relevant tests and review pass; ordinary development commits are not mirrored directly onto `main`.
