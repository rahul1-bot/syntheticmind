| Memory | SyntheticMind Mission and Boundaries | Date: 19 August 2026 | Time: 04:31 AM | Name: Ghost |

    1. SyntheticMind is personal infrastructure rather than a public framework product. Its value is direct control over machine-learning execution for FAU work, research, experiments, and creative enjoyment.
    2. The codebase was built from the ground up so training behavior can be inspected, changed, and reasoned about end to end.
    3. The end state requires PyTorch Lightning-scale user capability, but breadth must be added only through correct sequential components. A broad interface containing ceremonial or misleading parameters is not capability parity.
    4. The harness remains incomplete, and future assessments must distinguish implemented behavior from intended behavior or README claims.
    5. Documentation creation and release work are excluded from the active objective. Engineering time belongs to executable correctness, focused verification, and user-visible capability.

| Memory | Terminology Compatibility Rule | Date: 19 August 2026 | Time: 04:31 AM | Name: Ghost |

    1. Machine-learning users carry established expectations for terms such as `model`, `max_steps`, `val_check_interval`, `ckpt_path`, `devices`, `precision`, and `global_step`.
    2. Reusing a standard term with different semantics is as dangerous as inventing a replacement term because both create silent operational confusion.
    3. Trainer comparisons must therefore inspect parameter names, accepted values, defaults, validation, lifecycle effects, and checkpoint behavior rather than comparing signatures mechanically.
    4. Core PyTorch Lightning user capabilities are the target except for explicit exclusions such as Fabric in the current phase. Internal framework baggage, public-product ceremony, documentation machinery, and release infrastructure do not need to be copied merely to achieve functional parity.
    5. The generic Python guidelines at `/Users/rahulsawhney/LocalCode/AI/syntheticmind/docs/references/guidelines/code-guidelines.md` remain the implementation baseline for typing, boundaries, validation, composition, and maintainability.

| Memory | Trainer Audit Evidence | Date: 19 August 2026 | Time: 04:48 AM | Name: Ghost |

    1. The controlling Lightning comparison revision is `fcef404516264798b6c582ef556eaa17e5c29bfb`; comparison claims must not be attributed to another checkout or release.
    2. SyntheticMind currently defines `global_step` as an optimizer-step count in state documentation but increments it once per manual training batch, irrespective of how many optimizer steps actually execute.
    3. Mid-fit validation changes the Trainer running stage to validation and restores only the module stage, causing later training callbacks to observe stale state.
    4. Global-step validation cadence can trigger repeatedly on accumulation microbatches because the validation predicate does not prove that a new optimizer step occurred.
    5. Unsized iterable datasets can finish inside a distributed `no_sync()` accumulation window; the subsequent tail optimizer step uses rank-local gradients and produces divergent model parameters.
    6. The distributed dataloader reconstruction path does not preserve custom batch samplers, loader subclasses, or every current PyTorch DataLoader constructor field.
    7. Several parameter names collide with established Lightning terminology while implementing different values or edge-case semantics. Hardening must correct existing semantics before adding more parameters.
    8. The preferred boundary is a small Trainer facade backed by immutable validated configuration and focused collaborators. Lightning is a semantic reference, not an architectural template to copy wholesale.

| Memory | Compatibility Hierarchy | Date: 19 August 2026 | Time: 05:19 AM | Name: Ghost |

    1. Compatibility begins with meaning. A familiar name is valid only when its accepted values, defaults, validation, runtime effect, state transitions, errors, and checkpoint consequences preserve the established contract.
    2. Method names and hook contracts are especially important because machine-learning users commonly inherit framework classes and override lifecycle behavior. Signature shape, invocation order, frequency, and execution context must therefore be treated as public behavior.
    3. End-to-end workflow parity is the controlling acceptance standard. Matching names without enabling the corresponding workflow is a false interface and must not be counted as implemented capability.
    4. SyntheticMind's internal decomposition may differ from PyTorch Lightning when the result remains semantically compatible and conforms to the generic Python guidelines.
    5. The comparison and reconstruction process must move sequentially through dependency layers. Mid-runtime feature insertion before foundational state and lifecycle contracts are stable creates compounding defects.
    6. Existing source comments are temporarily grandfathered. The current phase neither cleans them nor introduces new source-code comments.
    7. Inheritance order and method-resolution order are public behavior whenever users subclass the harness. Cooperative initialization, `super()` chains, override discovery, signatures, defaults, and return types must be verified together rather than reviewing method names in isolation.
    8. Industry-standard machine-learning key terms and naming conventions are mandatory at every public boundary. An internal abstraction may be cleaner than Lightning's implementation, but it may not force users to translate an invented vocabulary for an established concept.

| Memory | Evidence Escalation and Ledger Discipline | Date: 19 August 2026 | Time: 08:21 AM | Name: Ghost |

    1. When implementation knowledge is incomplete, inspect the local runtime path, pinned PyTorch Lightning source, installed package behavior, and focused probes before expanding the search.
    2. If those sources are insufficient, use current official or primary sources and carefully chosen external repositories. External checkouts belong under ignored `external/` and serve as evidence, not code to copy blindly.
    3. The MacBook environment is part of verification evidence because the required dependencies are installed locally; claims about behavior should be executed where proportionate rather than inferred from signatures alone.
    4. Git history is the complete chronological record. Ledgers contain only the current objective, durable decisions, reusable findings, and actionable state, so completed-session narration must be consolidated instead of appended indefinitely.
    5. Operational persistence means exhausting safe evidence paths, correcting weak assumptions, and continuing until the capability is genuinely complete. It does not justify placeholders, speculative APIs, or declaring parity from surface resemblance.
    6. Package installation is authorized through `python -m pip install --break-system-packages` when a verification dependency is genuinely required. Diagnose the existing package graph first and avoid broad global repairs when a targeted pin or isolated environment answers the engineering question safely.
    7. Ruff and other linters are not part of this program. Guideline compliance is reviewed directly, while behavior is proven through runtime assertions, end-to-end flows, source-backed signature and inheritance inspection, exception probes, and compilation.
