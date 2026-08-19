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
    4. Core PyTorch Lightning user capabilities are the target except for explicit exclusions such as Fabric in the current phase. Lightning-branded names, internal framework baggage, public-product ceremony, documentation machinery, and release infrastructure do not become SyntheticMind requirements merely because the reference contains them.
    5. The generic Python guidelines at `/Users/rahulsawhney/LocalCode/AI/syntheticmind/docs/references/guidelines/code-guidelines.md` remain the implementation baseline for typing, boundaries, validation, composition, and maintainability.
    6. Vocabulary has three categories: universal machine-learning or PyTorch terms that should be preserved, descriptive extension contracts that require deliberate adoption, and framework-owned branding that must be renamed or excluded. Reference parity never collapses these categories.

| Memory | Trainer Audit Evidence | Date: 19 August 2026 | Time: 04:48 AM | Name: Ghost |

    1. The controlling Lightning comparison revision is `fcef404516264798b6c582ef556eaa17e5c29bfb`; comparison claims must not be attributed to another checkout or release.
    2. SyntheticMind currently defines `global_step` as an optimizer-step count in state documentation but increments it once per manual training batch, irrespective of how many optimizer steps actually execute.
    3. Mid-fit and epoch-end validation must restore both Trainer and Module running-stage state to training before subsequent training hooks execute; the hook-dispatch layer now enforces this paired transition.
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

| Memory | Callback Extension Contract | Date: 19 August 2026 | Time: 11:37 AM | Name: Ghost |

    1. The current SyntheticMind `Callback` method surface contains every method in the pinned PyTorch Lightning base contract with equivalent positional shapes and defaults, while its public parameter vocabulary is owned by SyntheticMind rather than copied mechanically.
    2. Every public callback hook and built-in override uses `module`. The previous `pl_module` spelling was Lightning-owned branding rather than universal machine-learning terminology; removing it preserves the SyntheticMind override contract because all runtime hook dispatch is positional and every built-in override matches the corrected base signature.
    3. Evaluation publishes reduced metrics before ordinary epoch-end callbacks, merges metrics logged by the module epoch-end hook, and republishes them before monitoring callbacks. This preserves the standard signature while allowing EarlyStopping and ModelCheckpoint to observe module-contributed values.
    4. Callback state identity follows the qualified-class-name convention, parameterized keys use the standard class-qualified dictionary representation, and `_legacy_state_key` preserves the established compatibility surface.
    5. Python 3.14 callback modules require deferred annotations because `Trainer` is imported only during type checking to avoid circular imports. Without deferred annotations, standard `inspect.signature()` calls evaluate the missing runtime name and fail, obstructing extension discovery and override tooling.
    6. Source comparison checked the complete base surface, structural inspection checked every built-in override, and CPU executions covered fit, validation, test, prediction, EarlyStopping, ModelCheckpoint, callback metrics, state keys, runtime signature inspection, and callback dispatch after the ownership correction.

| Memory | Evidence Escalation and Ledger Discipline | Date: 19 August 2026 | Time: 08:21 AM | Name: Ghost |

    1. When implementation knowledge is incomplete, inspect the local runtime path, pinned PyTorch Lightning source, installed package behavior, and focused probes before expanding the search.
    2. If those sources are insufficient, use current official or primary sources and carefully chosen external repositories. External checkouts belong under ignored `external/` and serve as evidence, not code to copy blindly.
    3. The MacBook environment is part of verification evidence because the required dependencies are installed locally; claims about behavior should be executed where proportionate rather than inferred from signatures alone.
    4. Git history is the complete chronological record. Ledgers contain only the current objective, durable decisions, reusable findings, and actionable state, so completed-session narration must be consolidated instead of appended indefinitely.
    5. Operational persistence means exhausting safe evidence paths, correcting weak assumptions, and continuing until the capability is genuinely complete. It does not justify placeholders, speculative APIs, or declaring parity from surface resemblance.
    6. Package installation is authorized through `python -m pip install --break-system-packages` when a verification dependency is genuinely required. Diagnose the existing package graph first and avoid broad global repairs when a targeted pin or isolated environment answers the engineering question safely.
    7. Ruff and other linters are not part of this program. Guideline compliance is reviewed directly, while behavior is proven through runtime assertions, end-to-end flows, source-backed signature and inheritance inspection, exception probes, and compilation.

| Memory | Optimizer and Scheduler Compatibility | Date: 19 August 2026 | Time: 11:37 AM | Name: Ghost |

    1. Optimizer compatibility is an execution contract rather than a parser contract. Correctness requires the closure to compose training-step execution, gradient clearing, backward, pre-step inspection, clipping, scaler handling, and the raw optimizer step in the expected order.
    2. `global_step` now advances once for each real managed-optimizer `step()` invocation in manual optimization and once for each loop-owned automatic optimizer step. It no longer advances merely because a manual-optimization batch completed.
    3. Step schedulers use the training batch index and configured frequency, while epoch schedulers split non-plateau updates before validation from plateau updates after monitored metrics are available. Scheduler stepping remains overridable through `lr_scheduler_step`.
    4. Python 3.14 evaluates deferred class annotations during runtime signature inspection. A public annotation using bare `float` inside a `torch.nn.Module` subclass can resolve to the inherited `Module.float` method; `builtins.float` is required at that boundary to preserve introspection.
    5. A preconstructed optimizer may already contain CPU state before its model is moved. Recursive state movement must preserve the top-level `step` state while moving tensors such as Adagrad accumulators to the selected device.
    6. Dynamic optimizer wrapping is required for managed extension behavior and `isinstance` compatibility. The public wrapper is the SyntheticMind-owned `SyntheticOptimizer`; runtime subclasses retain the underlying PyTorch optimizer family, expose the raw `torch.optim.Optimizer` through `optimizer`, delegate ordinary attributes, and route `step` through Trainer-owned precision and progress handling. `Module.optimizers()` returns managed wrappers by default and `use_synthetic_optimizer=False` returns raw PyTorch optimizers explicitly.
    7. The distributed unsized-dataloader accumulation tail remains unresolved: a final batch executed under `no_sync()` can still reach the tail optimizer step with rank-local gradients. It must be corrected in the distributed execution phase rather than concealed by optimizer-layer success.

| Memory | Console Experience Contract | Date: 19 August 2026 | Time: 11:28 AM | Name: Ghost |

    1. The current human-facing console is a collection of independent `tqdm` bars and Loguru sentences. Training, validation, and testing have bars; prediction has no progress surface, and stage summaries are not presented through one coherent renderer.
    2. Useful telemetry already exists below the presentation layer: stage and batch duration, dataloader wait and iterator startup, throughput, CPU utilization, CUDA and MPS memory, CUDA utilization and power, optimizer progress, logged metrics, checkpoints, and estimated stepping batches.
    3. The all-stage smoke exposed terminal collisions between Loguru output and live bars, stale training metrics appearing in validation postfixes, unbounded metric strings destroying narrow layouts, and raw logger metadata overwhelming the information hierarchy.
    4. `TimerCallback` currently reuses one timestamp for training and validation. Epoch-end validation overwrites the training start timestamp, causing false near-zero epoch training durations; sanity validation is also included in its validation total. Presentation work must not beautify these incorrect measurements.
    5. Test timing is not printed by `TimerCallback`, prediction has neither a progress bar nor a console duration, and final summaries do not report batch latency, throughput, best metrics, device utilization, checkpoint destination, or termination reason together.
    6. The approved design is one mutable TTY status line, one immutable line per completed epoch, and one immutable final line per completed stage. Default content is limited to stage, progress, at most two selected metrics, and ETA when it can be computed honestly.
    7. Detailed telemetry belongs in CSV, TensorBoard, profiler artifacts, and programmatic Trainer state. The terminal renderer is a presentation subscriber and must not become an experiment logger, profiler, or second source of runtime truth.
    8. Redirected and non-interactive output must remain stable plain text suitable for files, schedulers, continuous integration, and remote cluster logs. The harness borrows progressive disclosure from coding agents without pretending that a long-running training process is an interactive coding-agent session.
