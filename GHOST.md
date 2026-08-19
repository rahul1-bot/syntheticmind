| Strategic Direction | SyntheticMind Harness Hardening | Date: 19 August 2026 | Time: 04:31 AM | Name: Ghost |

    1. SyntheticMind is a private machine-learning harness created from the ground up for Rahul's FAU study work, independent research, experimentation, and the enjoyment of controlling the complete workflow directly.
    2. The project exists to provide end-to-end ownership of training, validation, testing, prediction, optimization, checkpointing, logging, callbacks, devices, precision, and distributed execution without surrendering the runtime to a larger framework.
    3. The harness is intentionally incomplete. Its current status must not be represented as production-ready, release-ready, or capability-complete.
    4. The active program is code hardening and capability implementation supported by proportionate behavioral verification. Code behavior and demonstrated operational capability control the work queue.
    5. Public-release ceremony, promotional work, and documentation creation are excluded unless Rahul explicitly reopens them. Architecture notes, issue documents, and documentation parity are not deliverables.
    6. PyTorch Lightning is the controlling reference for user-facing capability parity, established Trainer terminology, parameter semantics, lifecycle behavior, extension hooks, and failure handling; SyntheticMind retains its own guideline-compliant internal architecture.

| Architecture Decision | Public Terminology and Python Standards | Date: 19 August 2026 | Time: 04:31 AM | Name: Ghost |

    1. Established machine-learning terminology is a compatibility boundary. SyntheticMind must use recognizable industry-standard names and meanings when a stable convention already exists.
    2. New public terms are prohibited when they merely rename an established concept. A project-specific term is acceptable only when it represents genuinely different semantics and that difference is enforced by code and tests.
    3. The generic Python engineering rules governing this codebase are stored at `/Users/rahulsawhney/LocalCode/AI/syntheticmind/docs/references/guidelines/code-guidelines.md`.
    4. Public API terminology follows machine-learning conventions, while internal structure follows the generic Python guidelines. Any conflict must be resolved explicitly rather than allowing accidental architectural drift.
    5. The local PyTorch Lightning checkout under `docs/references/code/pytorch_lightning/pytorch-lightning` is read-only comparative material and is excluded from SyntheticMind's Git history with the complete `docs/` tree.
    6. Public base classes are extension contracts. Their inheritance order, method-resolution order, cooperative `super()` behavior, overridable method names, parameter names and order, defaults, return contracts, invocation frequency, and lifecycle semantics must remain recognizable to experienced PyTorch and PyTorch Lightning users.
    7. Industry-standard machine-learning concepts and key terms control the public vocabulary. Cosmetic aliases, renamed hooks, and project-specific substitutes for established concepts are prohibited because downstream users inherit these classes and override methods by name.
    8. The complete `code-guidelines.md` is mandatory implementation policy rather than optional style guidance. New and modified code must be reviewed directly against it while preserving the stronger public compatibility boundary where inheritance and machine-learning conventions apply.

| Architecture Decision | Framework Ownership and Reference Taxonomy | Date: 19 August 2026 | Time: 11:37 AM | Name: Ghost |

    1. SyntheticMind is an independent machine-learning harness. PyTorch Lightning is evidence for mature capabilities and runtime contracts, not a product identity, source template, drop-in namespace, or naming authority.
    2. Public vocabulary must be classified before adoption: universal machine-learning and PyTorch terminology is preserved, descriptive extension contracts are adopted only through an explicit SyntheticMind decision, and Lightning-branded vocabulary is rejected or translated into SyntheticMind-owned language.
    3. `LightningOptimizer` is not an optimizer algorithm or an industry-standard term. It is Lightning's public framework wrapper around a user-supplied `torch.optim.Optimizer` for managed stepping, precision, profiling, progress accounting, closures, and optimizer toggling.
    4. SyntheticMind requires the same managed-optimizer capability under the public name `SyntheticOptimizer`. The underlying algorithm remains a standard `torch.optim.Optimizer`, and the wrapper must remain transparent about that distinction.
    5. The rejected `LightningOptimizer`, `use_pl_optimizer`, and `_lightning_optimizers` identifiers have been removed. The accepted managed surface is `SyntheticOptimizer`, `use_synthetic_optimizer`, and the private `_synthetic_optimizers` collection, while explicit raw access continues to return `torch.optim.Optimizer` instances.
    6. The copied callback parameter name `pl_module` was Lightning-owned branding without a broader machine-learning meaning. Every SyntheticMind callback hook and built-in override now uses `module`, preserving method names, parameter order, defaults, positional invocation, and override compatibility inside the SyntheticMind inheritance contract.
    7. Capability parity means that users can complete the same core workflows with equivalent semantics. It does not require copying Lightning-specific class names, prefixes, internal decomposition, or historical implementation baggage.
    8. Ownership correction is accepted only with executable evidence: the source contains no remaining Lightning-branded Python identifiers, callback override signatures match the corrected base, dynamic wrappers retain their underlying PyTorch optimizer inheritance, and automatic, manual multi-optimizer, LBFGS closure, scheduler, toggling, and raw-access paths execute successfully.

| Repository Policy | Main and Development Branches | Date: 19 August 2026 | Time: 04:31 AM | Name: Ghost |

    1. The GitHub repository is private and uses `origin` as the canonical remote.
    2. `main` holds the stable project baseline, including all four root ledgers. Direct feature and hardening work does not begin on `main`.
    3. `dev` is the default working branch for Trainer hardening and future implementation work.
    4. Every accepted development commit must be pushed to `origin/dev` before the work is treated as synchronized or complete.
    5. Local `dev` and `origin/dev` must remain at the same commit after each completed commit-and-push cycle. Divergence is a defect to resolve before beginning unrelated work.
    6. Changes reach `main` only through an intentional stabilization decision after relevant tests and review pass; ordinary development commits are not mirrored directly onto `main`.
    7. Third-party source repositories used for research belong under the ignored `external/` directory and must never enter SyntheticMind's project history.
    8. The ledgers record current objectives, durable decisions, reusable evidence, and live progress. Git commits, diffs, and logs are the canonical timeline and must not be duplicated into ledger chronology.

| Architecture Decision | PyTorch Lightning Functional Parity End State | Date: 19 August 2026 | Time: 05:19 AM | Name: Ghost |

    1. The required end state is full user-facing capability parity with the core PyTorch Lightning framework: workflows available to a PyTorch Lightning user must become executable through SyntheticMind.
    2. Fabric is excluded from the current parity boundary. PyTorch Lightning documentation, release machinery, and replication of its upstream test suite are excluded because they are not harness capabilities; SyntheticMind still requires focused verification of its own behavior.
    3. Public variables, methods, parameters, accepted values, defaults, meanings, state counters, errors, hook signatures, hook order, and lifecycle effects must follow established machine-learning and PyTorch Lightning conventions wherever those conventions describe the same concept.
    4. Inheritance and method overriding are compatibility-critical behavior. Stable extension points must retain recognizable names and contracts so experienced machine-learning users do not need to translate an invented SyntheticMind vocabulary.
    5. Divergence is permitted only when SyntheticMind implements a genuinely different algorithm or semantic contract. Every divergence requires an explicit technical rationale and must not silently reuse an established name with a different meaning.
    6. Functional parity does not require copying PyTorch Lightning's internal architecture. SyntheticMind internals remain governed by the generic Python guidelines at `/Users/rahulsawhney/LocalCode/AI/syntheticmind/docs/references/guidelines/code-guidelines.md`.
    7. Existing source comments remain untouched during the current phase. Comment cleanup and new source-code comments are deferred unless Rahul explicitly changes this constraint.

| Execution Doctrine | Sequential Capability Reconstruction | Date: 19 August 2026 | Time: 05:19 AM | Name: Ghost |

    1. The codebase must not be expanded through isolated patches applied in the middle of an unmapped runtime. That approach would compound incompatible state, naming, and lifecycle assumptions.
    2. Work proceeds in dependency order: establish the capability boundary from source evidence, define compatibility contracts, harden foundational state and lifecycle behavior, verify the completed layer, and then add higher-level capabilities.
    3. Each subsystem must be understood end to end before implementation begins, including its public surface, runtime path, inherited hooks, state transitions, failure behavior, and interactions with already hardened components.
    4. Each completed layer must be verified before dependent layers begin. Capability breadth is accumulated deliberately rather than simulated through unused parameters or placeholder methods.
    5. Documentation artifacts are not created for this program. Necessary reasoning is expressed through code structure, focused tests, concise durable ledger state, and reviewable Git commits.

| Operational Doctrine | Evidence Escalation and Persistence | Date: 19 August 2026 | Time: 08:21 AM | Name: Ghost |

    1. Uncertainty is a research trigger rather than a stopping condition. The first evidence sources are SyntheticMind's complete runtime path, the pinned local PyTorch Lightning source, installed package behavior, and focused executable probes on the MacBook.
    2. If local evidence is insufficient, current official documentation, primary technical sources, repository history, and carefully selected external implementations must be investigated before a design decision is made.
    3. Useful third-party repositories may be cloned under `external/` as ignored read-only references. Selection must be driven by a concrete unresolved capability or implementation question rather than indiscriminate collection.
    4. `git log`, `git show`, and diffs provide the authoritative development timeline and earlier decision evidence. Ledgers must remain concise indexes rather than accumulating session logs.
    5. A phase is complete only after its code is implemented, relevant behavior is verified, the diff is reviewed, and the commit is pushed so `dev` and `origin/dev` resolve to the same commit.
    6. Persistence does not permit speculative interfaces, copied baggage, or false completion. When evidence exposes a weak assumption, the implementation direction must be corrected and work must continue from the strongest supported design.
    7. Python dependencies may be installed when verification requires them, including global installation through `python -m pip install --break-system-packages`. The existing dependency graph must be inspected first, and targeted or isolated installation is preferred whenever a global change could destabilize unrelated workloads.

| Verification Policy | Executable Evidence Without Linters | Date: 19 August 2026 | Time: 09:03 AM | Name: Ghost |

    1. Ruff and other linting tools are excluded from the active hardening workflow and must not be invoked.
    2. Compliance with `code-guidelines.md` is established through deliberate source review rather than automated lint output.
    3. Capability claims require focused runtime assertions, end-to-end execution, exception-path probes, signature and method-resolution inspection, source-backed contract comparison, and Python compilation where relevant.
    4. Verification must prioritize observable behavior, public compatibility, and inheritance safety. Passing a stylistic tool could never substitute for correct lifecycle semantics, parameter meaning, or override behavior.

| Architecture Decision | Optimization Extension Contract | Date: 19 August 2026 | Time: 11:37 AM | Name: Ghost |

    1. `configure_optimizers` currently accepts an optimizer, an optimizer sequence, the paired optimizer-and-scheduler sequence form, optimizer dictionaries, optimizer-plus-scheduler dictionaries, dictionary sequences, and `None`. These structures are deliberately supported capability contracts and must not be described as universal terminology merely because Lightning also supports them.
    2. `torch.optim.Optimizer` is the canonical optimizer-algorithm contract. SyntheticMind's framework-managed public adapter is named `SyntheticOptimizer`; scheduler configuration types remain subject to the same ownership and terminology audit before their public status is finalized.
    3. `Module` exposes optimizer and scheduler accessors and override methods whose behavior remains compatibility-critical. Automatic and manual optimization may receive a `SyntheticOptimizer` where Trainer-owned stepping is required, while raw optimizer access must remain explicit and return the underlying PyTorch optimizer.
    4. Optimizer closures own training-step execution, zeroing, backward, pre-step hooks, gradient clipping, and optimizer execution in the required order. This preserves LBFGS re-entry, gradient accumulation, mixed-precision scaling, manual backward arguments, and optimizer-step-based global progress.
    5. Scheduler configuration preserves interval, frequency, plateau-monitor, strictness, name, optimizer attachment, custom-scheduler override, and checkpoint-state contracts. Optimizer state is placed on the execution device while the top-level optimizer step counter remains on its required device.
    6. Parameter toggling preserves originally frozen parameters, shared optimizer parameters, unmanaged parameters, and exception-safe restoration. The standard context managers yield no value and remain safe when invoked through compiled model code.
    7. The optimization foundation is accepted under SyntheticMind ownership. Existing behavioral verification covers accepted and rejected configuration shapes, public signatures and runtime annotations, inheritance order, CPU automatic and manual training, LBFGS closure repetition, scaler ordering, clipping conflicts, scheduler cadence, checkpoint resume, prebuilt optimizer state on MPS, managed-versus-raw access, and callback signature consistency without invoking a linter.

| Architecture Decision | Minimal Terminal Presentation Contract | Date: 19 August 2026 | Time: 11:28 AM | Name: Ghost |

    1. SyntheticMind adopts a restrained status-line terminal experience inspired by the progressive-disclosure model used by current coding agents. It does not adopt a dense monitoring dashboard, boxed control panel, or interactive agent command interface.
    2. An interactive terminal displays one live line for the active stage. The default information budget is stage, progress, at most two user-selected metrics, and an estimated remaining duration only when the progress horizon is knowable.
    3. Validation replaces the active training line rather than opening a competing bar. Each completed epoch leaves one compact permanent result line, and each completed fit, validation, test, or prediction stage leaves one compact final summary.
    4. Parameter counts, device utilization, memory, throughput, learning rate, logger paths, profiler counters, and other diagnostics remain available through structured experiment logging and profiling but are excluded from default terminal output.
    5. Actionable warnings may interrupt the compact presentation; routine Loguru timestamps, module names, source locations, and diagnostic chatter must not pollute the interactive progress surface.
    6. Rendering occurs only on global rank zero. Non-interactive and redirected execution emits stable plain-text epoch and stage lines without ANSI control sequences, spinners, carriage-return rewriting, or fabricated progress percentages for unsized dataloaders.
    7. Presentation depends on truthful stage-isolated measurements. Training, validation, testing, prediction, and sanity-check timing must not overwrite or contaminate one another, and metric routing must prevent stale values from crossing stage boundaries.
