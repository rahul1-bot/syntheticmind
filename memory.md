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
    2. `global_step` is now an optimizer-step count in both automatic and manual optimization. Manual steps advance only through managed `SyntheticOptimizer.step()` execution rather than once per completed training batch.
    3. Mid-fit and epoch-end validation must restore both Trainer and Module running-stage state to training before subsequent training hooks execute; the hook-dispatch layer now enforces this paired transition.
    4. Global-step validation now proves that the current batch advanced optimizer progress before checking the interval, so accumulation microbatches cannot retrigger validation at an unchanged step.
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

    1. The human-facing console is now one SyntheticMind-owned renderer rather than independent `tqdm` bars and routine Loguru sentences. Fit, standalone validation, testing, and prediction share the same output grammar and duration formatting.
    2. Useful telemetry already exists below the presentation layer: stage and batch duration, dataloader wait and iterator startup, throughput, CPU utilization, CUDA and MPS memory, CUDA utilization and power, optimizer progress, logged metrics, checkpoints, and estimated stepping batches.
    3. Terminal collisions, cross-stage metric leakage, unbounded metric strings, prediction silence, and routine raw logger metadata were removed from the default path. Interactive lines are width-bounded and redirected output contains no terminal control sequences.
    4. The former shared Timer timestamp allowed validation to overwrite the training start time and produced false near-zero epoch durations. Independent monotonic stage clocks now pause training around validation, exclude sanity validation, reset standalone stages, and close active segments on failure.
    5. The standard `Timer` separates public stage timing from internal presentation measurements. Trainer-installed instances remain quiet so one renderer owns the human-facing terminal, while an explicitly constructed timer retains its standard `verbose` control.
    6. The approved design is one mutable TTY status line, one immutable line per completed epoch, and one immutable final line per completed stage. Default content is limited to stage, progress, at most two selected metrics, and ETA when it can be computed honestly.
    7. Detailed telemetry belongs in CSV, TensorBoard, profiler artifacts, and programmatic Trainer state. The terminal renderer is a presentation subscriber and must not become an experiment logger, profiler, or second source of runtime truth.
    8. Redirected and non-interactive output must remain stable plain text suitable for files, schedulers, continuous integration, and remote cluster logs. The harness borrows progressive disclosure from coding agents without pretending that a long-running training process is an interactive coding-agent session.
    9. Progress-bar values now travel through stage-isolated mappings populated only by `progress_bar=True`; callback monitoring and experiment logging retain separate routing. Run, epoch, pass, and sanity boundaries clear the appropriate mapping so stale metrics cannot cross stages.
    10. Unsized loaders retain an unknown horizon and prediction now records its own loader length when available. This distinction is required for honest percentages and estimated completion times.
    11. The concrete renderer is separated from callback lifecycle orchestration. This keeps terminal I/O and formatting independently testable and prevents `ProgressBarCallback` from becoming a combined experiment logger, profiler, timer, and terminal driver.
    12. Successful epoch lines are held pending until a later safe boundary. Failures inside epoch-end and fit-end hooks therefore cannot leave a false completed-epoch or completed-fit claim.

| Memory | Exception-Safe Stage Teardown | Date: 19 August 2026 | Time: 12:27 PM | Name: Ghost |

    1. Teardown cannot be implemented as a sequence of unguarded calls because the first failure would skip every later resource owner and can leak distributed, data, callback, module, or logger state.
    2. Runtime failure and cleanup failure require different ownership. A runtime, setup, interruption, or exit exception remains primary; a teardown-originated failure becomes primary only when stage execution had otherwise succeeded.
    3. Exception hooks are untrusted extension points and must be isolated individually. One callback, datamodule, or strategy exception hook may fail without preventing the remaining hooks or replacing the original exception.
    4. Python exception notes preserve secondary failure evidence without fabricating an exception group as the public failure contract. This keeps the first causal exception catchable by its original type while retaining every cleanup diagnosis.
    5. Presentation cleanup must execute after all resource cleanup because a stage summary is a transactional claim. The terminal may report completion only when strategy, datamodule, ordinary callback, module, and logger teardown have all succeeded.
    6. Standard module `setup(stage)` and `teardown(stage)` hooks are now real runtime extension points for fit, validation, test, and prediction rather than inherited methods that were present but never dispatched.
    7. Stage markers must clear after cleanup irrespective of outcome, while interrupted status must survive that clearing. This distinction permits postmortem truth and later Trainer reuse without leaving a live stage attached.

| Memory | Training Limits and Cooperative Stop Semantics | Date: 19 August 2026 | Time: 12:41 PM | Name: Ghost |

    1. Epoch limits and optimizer-step limits are independent hard boundaries. `-1` disables the corresponding hard limit, while zero performs no training and still completes the surrounding fit lifecycle cleanly.
    2. When `max_epochs` is unspecified, a finite `max_steps` must remove the epoch ceiling; otherwise a derived epoch default could terminate iteration-based training before its explicit optimizer-step budget is reached.
    3. Minimum epochs and steps do not weaken hard maxima. They delay only cooperative stop signals, and both minimum conditions must be satisfied before EarlyStopping or another collaborator can end the fit.
    4. The public `Trainer.should_stop` value must share storage with loop state. Separate flags would let callbacks and loops observe different stop decisions and create non-deterministic termination.
    5. A max-step check belongs before batch fetch and after completed batch processing. The first prevents consuming work after a restored checkpoint has already reached its limit; the second preserves hooks and metrics for the step that reaches the boundary.
    6. A closure may execute `training_step` several times while one optimizer invocation still represents one global step. LBFGS verification must therefore assert optimizer progress independently from closure-call count.
    7. Unlimited training requires honest presentation and estimation. A negative epoch denominator is not a user-facing horizon, and an unknown horizon remains infinity unless a finite step cap supplies the bound.
    8. Checkpoint persistence is part of termination safety. Interruption during atomic save must remove the temporary file, and normal save or restore must remain silent unless an actionable problem occurs.

| Memory | Validation Cadence Semantics | Date: 19 August 2026 | Time: 12:55 PM | Name: Ghost |

    1. Validation cadence counts raw batches, not optimizer steps. Gradient accumulation, manual optimization, skipped optimizer transitions, and closure repetition must not alter an integer `val_check_interval`.
    2. `check_val_every_n_epoch=None` changes the integer counter from epoch-local to run-global. Exact checkpoint restoration therefore requires cumulative raw-batch state in addition to `global_step` and within-epoch replay coordinates.
    3. A final-batch validation is an epoch-boundary checkpoint even though it executes inside the training epoch loop. A mid-epoch validation or validation failure remains a replayable mid-epoch checkpoint.
    4. Float cadence requires a knowable effective training horizon. An unsized loader supports `1.0` at exhaustion, an integer interval, or monotonic time; any other fraction would claim a position that cannot be computed.
    5. Time cadence is not converted into batches. Its clock resets after sanity and real validation, elapsed time carries through epochs excluded by `check_val_every_n_epoch`, and duration representations normalize once at construction.
    6. Cooperative stopping may require validation before exit so monitoring state is coherent, but hard limits do not imply an extra validation pass. These termination causes must remain distinct.

| Memory | Batch Limit Semantics | Date: 19 August 2026 | Time: 01:02 PM | Name: Ghost |

    1. Integer and float batch limits are intentionally not interchangeable at one: `1` selects one batch, while `1.0` selects the complete sized dataloader or unbounded consumption of an unsized loader.
    2. Zero is a genuine opt-out and must be checked before sized-versus-unsized resolution. Coercing a zero fraction through `max(1, ...)` silently enables work the user explicitly disabled.
    3. A positive fraction that floors to zero is a configuration error rather than permission to round up. The minimum valid fraction is `1 / number_of_batches`, while an actually empty loader resolves cleanly to zero.
    4. `None` is an API-level default sentinel and normalizes to `1.0`; the normalized Trainer attributes are the runtime source of truth. This preserves introspection defaults without forcing every loop to understand optional values.
    5. An unsized dataloader supports a finite integer budget or the complete `1.0` stream. Fractional sampling has no truthful denominator and must fail before iteration; zero remains a valid no-work exception.
    6. Sanity validation is bounded by both `num_sanity_val_steps` and the resolved validation limit. A restored fit skips sanity entirely so loaded callback, metric, random, and progress state cannot be perturbed before continuation.

| Memory | Wall-Clock Stop Semantics | Date: 19 August 2026 | Time: 01:15 PM | Name: Ghost |

    1. `max_time` is not a hard maximum analogous to `max_steps`. Its `Timer` callback sets `Trainer.should_stop`, and the fit loop applies the same minimum-epoch and minimum-step gate used for other cooperative stop requests.
    2. Time limiting measures the continuous wall-clock interval from training start through training end, including fit-scheduled validation. The separate presentation measurement may pause during validation without changing the public elapsed-time meaning.
    3. Checkpoint restoration contributes saved training time as an offset. The fit-start check is required because a restored timer can already be depleted before `on_train_start` establishes a new local clock.
    4. Step intervals check after each training batch and epoch intervals check after each training epoch. A zero duration therefore prevents training when no minimum is configured, while configured minima intentionally permit an overrun.
    5. Distributed expiration is a collective control decision with logical-OR semantics. Requiring every rank to expire independently could deadlock when rank timing differs around validation, data loading, or callback work.
    6. The public `start_time`, `end_time`, `time_elapsed`, and `time_remaining` methods report standard stage values. Multiple validation passes replace the public validation start and end pair rather than silently converting the query into an aggregate.

| Memory | Debugging and Overfit Semantics | Date: 19 August 2026 | Time: 01:26 PM | Name: Ghost |

    1. `fast_dev_run=1` normalizes to `True` for standard public inspection, while larger integers retain their exact batch count and zero remains disabled. Negative integers and non-integral values are configuration errors.
    2. Fast development mode overrides ordinary stage limits because its purpose is a predictable end-to-end smoke test. `overfit_batches` is the deliberate exception: when enabled concurrently, it retains the training and validation subset budgets while fast development mode still bounds the other stages and runtime controls.
    3. Disabling a configured experiment logger by assigning `None` is behaviorally incomplete because inherited module code may access `self.logger.experiment`. A private no-operation logger preserves that execution surface while guaranteeing that metrics, hyperparameters, experiment calls, and finalization produce no user persistence.
    4. Overfitting a fixed subset requires sampler control in addition to a batch ceiling. Random training or validation samplers would otherwise select a different subset each epoch and make the feature semantically false.
    5. A custom `BatchSampler` can hide its active sampler behind `batch_sampler.sampler` while `DataLoader.sampler` appears sequential. Shuffle detection and replacement must inspect the batching path that iteration actually consumes.
    6. Distributed overfit execution must use `DistributedSampler(shuffle=False)`. Sequentializing before distributed wrapping is insufficient if the later wrapper silently restores shuffling.
