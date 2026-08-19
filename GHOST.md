| Strategic Direction | SyntheticMind Harness Hardening | Date: 19 August 2026 | Time: 04:31 AM | Name: Ghost |

    1. SyntheticMind is a private machine-learning harness created from the ground up for Rahul's FAU study work, independent research, experimentation, and the enjoyment of controlling the complete workflow directly.
    2. The project exists to provide end-to-end ownership of training, validation, testing, prediction, optimization, checkpointing, logging, callbacks, devices, precision, and distributed execution without surrendering the runtime to a larger framework.
    3. The harness is intentionally incomplete. Its current status must not be represented as production-ready, release-ready, or capability-complete.
    4. The active phase is code hardening: confirm runtime correctness, close test gaps, strengthen invariants, and add only capabilities justified by real machine-learning workflows.
    5. Public-release ceremony, promotional work, and documentation expansion are outside the current mission unless Rahul explicitly reopens them. Code behavior and operational capability control the work queue.
    6. PyTorch Lightning is the controlling reference for user-facing capability parity, established Trainer terminology, parameter semantics, lifecycle behavior, extension hooks, and failure handling; SyntheticMind retains its own guideline-compliant internal architecture.

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

| Architecture Decision | PyTorch Lightning Functional Parity End State | Date: 19 August 2026 | Time: 05:19 AM | Name: Ghost |

    1. The required end state is full user-facing capability parity with the core PyTorch Lightning framework: workflows available to a PyTorch Lightning user must become executable through SyntheticMind.
    2. Fabric is excluded from the current parity boundary. Porting PyTorch Lightning's documentation, release machinery, and upstream test suite is also outside the current work phase because these are not harness capabilities.
    3. Public variables, methods, parameters, accepted values, defaults, meanings, state counters, errors, hook signatures, hook order, and lifecycle effects must follow established machine-learning and PyTorch Lightning conventions wherever those conventions describe the same concept.
    4. Inheritance and method overriding are compatibility-critical behavior. Stable extension points must retain recognizable names and contracts so experienced machine-learning users do not need to translate an invented SyntheticMind vocabulary.
    5. Divergence is permitted only when SyntheticMind implements a genuinely different algorithm or semantic contract. Every divergence requires an explicit technical rationale and must not silently reuse an established name with a different meaning.
    6. Functional parity does not require copying PyTorch Lightning's internal architecture. SyntheticMind internals remain governed by the generic Python guidelines at `/Users/rahulsawhney/LocalCode/AI/syntheticmind/docs/references/guidelines/code-guidelines.md`.
    7. Existing source comments remain untouched during the current phase. Comment cleanup and new source-code comments are deferred unless Rahul explicitly changes this constraint.

| Execution Doctrine | Sequential Capability Reconstruction | Date: 19 August 2026 | Time: 05:19 AM | Name: Ghost |

    1. The codebase must not be expanded through isolated patches applied in the middle of an unmapped runtime. That approach would compound incompatible state, naming, and lifecycle assumptions.
    2. Work proceeds in dependency order: establish the capability boundary, map the architecture, catalogue concrete issues, define compatibility contracts, harden foundational state and lifecycle behavior, and then add higher-level capabilities.
    3. Each subsystem must be understood end to end before implementation begins, including its public surface, runtime path, inherited hooks, state transitions, failure behavior, and interactions with already hardened components.
    4. Each completed layer must be verified before dependent layers begin. Capability breadth is accumulated deliberately rather than simulated through unused parameters or placeholder methods.
    5. `docs/architecture/` and `docs/issues/` are the local planning surfaces for the architecture map and evidence-backed issue catalogue. They remain excluded from Git history under the repository's complete `docs/` ignore policy.
