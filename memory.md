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

| Memory | Device Topology and Distributed Environment | Date: 19 August 2026 | Time: 01:39 PM | Name: Ghost |

    1. A device count and an ordered list of device identifiers represent different information. Counts expand through accelerator-specific availability, while explicit GPU identifiers must survive unchanged because local rank indexes the declared list rather than the physical identifier space directly.
    2. CPU parallel devices intentionally repeat `torch.device("cpu")`; their public `device_ids` are logical process positions. CUDA and MPS devices carry physical identifiers, so their public identifiers come from the concrete device objects.
    3. Distributed world size is a topology invariant equal to node count multiplied by processes per node. Both a launcher-provided `WORLD_SIZE` and an already initialized process group must be checked against that invariant or a stale group can execute with silently incorrect rank assumptions.
    4. Communication backend selection follows execution hardware: Gloo is the safe CPU default and NCCL is the CUDA default. Requesting NCCL without CUDA must fail before rendezvous rather than hanging or surfacing a remote backend error.
    5. Externally launched DDP is now executable and parameter-synchronized, but requiring users to invoke `torchrun` manually is not Trainer capability parity. Local subprocess creation, worker re-entry, failure propagation, and parent-state reconciliation remain the next distributed foundation.
    6. A real CUDA runtime is unavailable on the current MacBook. CUDA topology claims must therefore remain bounded to mocked availability, parser, signature, and rank-to-device probes until execution is verified on CUDA hardware.

| Memory | Distributed Launcher Lifecycle | Date: 19 August 2026 | Time: 02:17 PM | Name: Ghost |

    1. A script-reexecution DDP launcher must create child processes once and let every rank continue through the same user program. Waiting for child termination at the end of `fit()` deadlocks when the child advances into a later Trainer stage while rank zero waits for process exit.
    2. Destroying and immediately recreating a process group after each successful stage creates a rendezvous-generation race. A fast rank can connect to the previous TCP store while another rank is still tearing it down, leaving both ranks alive in incompatible groups and permanently blocked at the same apparent barrier.
    3. Successful stage teardown must therefore unwrap DDP while retaining the process group for the process lifetime. Exception handling must destroy the group immediately because a failed collective context is unsafe to reuse, and process-exit cleanup handles the successful path.
    4. Script launch and multiprocessing launch have different parent semantics. Script workers continue user code independently, while spawn and fork terminate after one launched stage and must return rank-zero weights, state, metrics, collaborator state, batch horizons, and random state to the dormant parent.
    5. A child-process observer must treat any nonzero worker exit as a job failure and terminate peers blocked in collectives. The exit path must still inspect worker return codes because a worker can fail after the final Trainer stage while the parent is already unwinding.
    6. Unsized iterable loaders cannot identify the last accumulation batch in advance. If exhaustion occurs after DDP `no_sync()`, the tail path must coordinate gradient presence in deterministic parameter order, reduce dense gradients, materialize missing rank-local gradients, average by world size, and only then perform the optimizer step.
    7. A partial distributed environment is more dangerous than no distributed environment because accidental local relaunch can create nested workers or inconsistent ranks. Rank variables, rendezvous address and port, topology, backend, node rank, local rank, thread count, and world size require validation before execution.
    8. Multiprocessing prediction results cannot be returned truthfully without an explicit cross-rank aggregation and ordering contract. Returning an empty or rank-zero-only prediction list by accident is corruption; the current launcher rejects retained predictions and permits `return_predictions=False` until that capability is implemented.

| Memory | Precision Runtime Semantics | Date: 19 August 2026 | Time: 03:01 PM | Name: Ghost |

    1. Precision is a lifecycle collaborator, not a loop flag. Module conversion, input conversion, forward context, backward preparation, optimizer stepping, gradient clipping, state persistence, and teardown must be delegated through one plugin instance or custom overrides will be bypassed inconsistently.
    2. `Module.backward` is an industry extension point. Calling `Tensor.backward` directly inside the precision plugin silently bypasses user-defined backward behavior even when the surrounding strategy and hook names appear correct.
    3. Float64 module conversion must use `Module.double()` rather than `Module.to(dtype=torch.float64)`. The latter converts complex tensors to real values and destroys their imaginary components, while the standard double conversion preserves existing complex buffers and parameters.
    4. Persistent autocast can cache a weight cast created under nested `no_grad` or `inference_mode`; on current PyTorch 2.9 that cached cast may be reused by the next gradient-enabled forward and produce an output without a gradient function. Clearing the autocast cache when the nested gradient-disabled context exits preserves both differentiability and the normal cached autocast path.
    5. A fused optimizer that performs AMP unscaling internally is valid when gradient clipping is disabled. It must bypass framework unscaling and proceed through `GradScaler.step`; it becomes incompatible only when the framework must inspect unscaled gradients for clipping.
    6. Precision-connected wrappers and the underlying user `Module` have different identities. The strategy-managed wrapper owns forward execution, while hook dispatch, backward overrides, optimizer ownership, checkpoint weights, and public module access remain attached to the underlying `Module`.
    7. Precision-plugin state must travel through ordinary checkpoints and multiprocessing worker reconciliation. A working scaler in a worker process is not sufficient if its scale or custom plugin state disappears when control returns to the parent.
    8. CPU and MPS execution can validate universal float and bfloat behavior, mixed-precision ordering, and process integration. CUDA AMP, NVIDIA fused kernels, and Transformer Engine FP8 require later NVIDIA hardware evidence; configuration mocks cannot establish numerical execution correctness.

| Memory | PyTorch Execution Context Semantics | Date: 19 August 2026 | Time: 03:10 PM | Name: Ghost |

    1. `deterministic=None` is not equivalent to `False`. `None` preserves the process's current deterministic-algorithm state, while explicit `False` disables deterministic algorithms.
    2. `benchmark=None` preserves the active cuDNN benchmark value unless a truthy deterministic mode requires an unspecified benchmark setting to resolve to `False`. Explicit `benchmark=True` remains explicit and therefore produces a conflict warning rather than being silently overwritten.
    3. A seed controls random streams; it does not logically control algorithm selection. Coupling the two causes a harmless-looking `seed` parameter to rewrite global performance and correctness policy.
    4. `torch.inference_mode` and `torch.no_grad` are not cosmetic aliases. No-gradient mode can be locally overridden with `torch.enable_grad`, while inference-mode tensors impose stronger restrictions that are unsuitable for fit validation and current Gloo execution.
    5. Evaluation context must cover lifecycle hooks, batch transfer, precision contexts, step execution, metric reduction, and end hooks. Wrapping only the step body creates inconsistent extension behavior that users cannot reason about.
    6. Autograd anomaly detection must surround the closure-driven fit loop because the backward call may occur inside an optimizer closure. Wrapping only `training_step` would miss the operation that exposes the faulty gradient.
    7. Deterministic flags and RNG state are process-global execution inputs. Spawned workers require explicit restoration, while environment variables such as `CUBLAS_WORKSPACE_CONFIG` propagate through the child process environment.

| Memory | Logger Configuration and Distributed Ownership | Date: 19 August 2026 | Time: 03:26 PM | Name: Ghost |

    1. `logger=None` and `logger=False` are different public contracts: the former requests the default backend and the latter disables experiment logging. Treating both as absence silently violates standard Trainer behavior.
    2. Multiple logger support requires one central dispatch authority. Replacing only the main training-loop call leaves evaluation, checkpoint, profiler, learning-rate, device, and throughput callbacks writing to the first backend and creates inconsistent experiment records.
    3. The first configured logger owns singular compatibility surfaces such as `Trainer.logger` and `Trainer.log_dir`; this does not grant it exclusive metric ownership. All configured backends must receive equivalent event payloads in the same order.
    4. Default version discovery must not create directories during logger construction. In distributed execution every rank constructs the Trainer independently; eager allocation allows later ranks to observe rank zero's new directory and select different versions before process coordination begins.
    5. `log_every_n_steps=0` disables only step-cadence logging. Epoch reductions remain semantically distinct events and must continue reaching experiment backends.
    6. Teardown is a collection-wide operation. Every logger must receive `finalize`, with the first failure retaining causal priority and later failures attached rather than swallowed or allowed to prevent subsequent cleanup.
    7. A no-operation logger during fast development preserves user code that accesses logger integrations while enforcing the stronger requirement that no experiment artifacts are written.

| Memory | Checkpoint State and Exact Resume | Date: 19 August 2026 | Time: 04:11 PM | Name: Ghost |

    1. Distributed checkpoint correctness includes callback state, not only file ownership. If rank zero alone evaluates top-k or last-path decisions, the model files may be valid while every other rank carries divergent `best_model_path`, score, ranking, and last-path state.
    2. Exact mid-epoch continuation needs two random-state coordinates. The epoch-start state reconstructs a shuffled dataloader and permits deterministic skipping, while the interruption-time state restores dropout and all model-side stochastic operations immediately before the first unfinished batch.
    3. A checkpoint epoch value is a lifecycle coordinate rather than the current loop integer. It advances only after the actual epoch-end extension boundary; incrementing it during arbitrary public saves creates a phantom completed epoch.
    4. Standard checkpoint keys should be the primary interoperability surface, but removing historical aliases immediately would turn a hardening phase into silent data loss. Writing or reading deliberate compatibility aliases is safer while the schema transition remains explicit.
    5. Public inheritance audits must compare methods as well as constructor signatures. The missing `on_validation_end` override would have bypassed expected subclass behavior even though checkpoint artifacts appeared to save successfully through adjacent hooks.
    6. On this MPS runtime, recursive tensor transfer with `non_blocking=True` produced zeroed or corrupted batches and non-finite training loss. Asynchronous host-to-device transfer is therefore restricted to CUDA until another backend has executable evidence for safe semantics.
    7. Distributed saves must build state on all participating ranks before rank-zero persistence when checkpoint hooks or callback decisions contain collectives. Making only the final filesystem write rank-zero owned avoids deadlock while preserving one authoritative artifact.

| Memory | Model Summary as an Observational Forward Pass | Date: 19 August 2026 | Time: 04:26 PM | Name: Ghost |

    1. A model summary with an example input is a real forward execution. Without explicit restoration it can consume dropout or custom random streams, alter heterogeneous submodule modes, leave forward hooks attached after failure, and change the training run before its first batch.
    2. Model configuration and precision conversion must precede summarization. Counting or forwarding at Trainer construction would miss layers created by `configure_model`, report the wrong parameter dtype and size, and execute on the wrong device.
    3. Rank-zero presentation and rank-zero computation are different decisions. Every distributed rank must execute the same summary-side model path when extension behavior may depend on rank or process state, while only one rank should write human-facing output.
    4. Callback override compatibility includes invocation form. Passing `total_flops` by keyword preserves subclasses that receive newer summary fields through `**summarize_kwargs`; passing it positionally would break an otherwise compatible established override.
    5. Hierarchical module rows may recursively include child parameters, while direct root parameters belong in a separate leftover row. The total model count remains authoritative and must not be reconstructed by summing overlapping hierarchy rows.
    6. Optional presentation dependencies must not silently seize the default terminal. The plain renderer provides stable redirected output, while Rich formatting remains an explicit user-selected capability with a clear dependency boundary.
    7. Uninitialized parameters require a cached warning because table construction queries totals, trainable counts, row counts, size, and formatting repeatedly. Emitting once preserves the defect signal without turning one lazy layer into a warning avalanche.

| Memory | Profiler Lifecycle and Action Semantics | Date: 19 August 2026 | Time: 05:07 PM | Name: Ghost |

    1. A profiler is a stage lifecycle collaborator rather than a timing utility wrapped around `training_step`. Correct observability requires the same centralized dispatch boundary used by module, datamodule, callback, and strategy extension hooks.
    2. Action names are an inspection interface. Universal execution terms such as `run_training_batch`, `run_training_epoch`, `save_checkpoint`, `backward`, and optimizer-step names should remain stable, while framework-owned prefixes must identify SyntheticMind rather than Lightning.
    3. Nested profiling is required because a batch action contains strategy, module, precision, optimizer, callback, and checkpoint work. cProfile parents must pause and resume around children, while PyTorch record-function ranges must preserve the nesting directly.
    4. Profiler teardown belongs after ordinary and presentation teardown hooks so their cost is observable, but its own failure must not mask an earlier runtime or cleanup failure. A failure reported after profiler closure must not restart the profiler through exception hooks.
    5. Artifact ownership follows the first logger only as a directory authority; profiler reports are not terminal progress metrics or experiment-log events. Rank suffixes prevent distributed writers from colliding, and the disabled pass-through path must not create directories.
    6. PyTorch profiler schedules advance on standard step-hook boundaries, module-name recording follows the actual strategy-connected module, and repeated Trainer stages require complete recording and trace reset without discarding the reusable profiler object.
    7. Hardware evidence remains explicit: MPS validates PyTorch operator recording on this MacBook, Gloo validates rank-separated artifacts, and CUDA NVTX plus XLA traces require later execution on the corresponding runtime.

| Memory | Barebones Contract Boundaries | Date: 19 August 2026 | Time: 05:20 PM | Name: Ghost |

    1. Barebones mode means measuring Trainer overhead with nonessential services removed; it does not mean bypassing optimization, lifecycle hooks, precision, distributed coordination, or user callbacks.
    2. Constructor normalization and runtime suppression are both required. Disabling default loggers is insufficient when `Module.log` can still allocate, reduce, or dispatch metrics inside a batch loop.
    3. Explicit enablement of a prohibited service must fail rather than be silently discarded because silent normalization conceals a contradictory benchmark configuration.
    4. User callback ownership remains precise: checkpoint and progress callbacks conflict with their disabled Trainer controls, an explicit model-summary callback remains legal, and a `Timer` introduced by `max_time` remains a termination capability rather than presentation overhead.
    5. Logging suppression must occur before ordinary hook-context and prediction validation so inherited model code becomes a warning no-operation consistently across every execution context.
    6. A missing progress bar requires a minimal run signal to distinguish deliberate quiet execution from a hung process. The signal must remain rank-zero, stable in redirected output, and independent of experiment logging.
    7. The standard plugin boundary is broader than precision alone. Its next audit must distinguish universal extension concepts from framework-owned branding before adding checkpoint-I/O, cluster-environment, or layer-synchronization capabilities.

| Memory | Plugin Composition and Lifecycle | Date: 19 August 2026 | Time: 05:46 PM | Name: Ghost |

    1. `CheckpointIO`, `ClusterEnvironment`, and `LayerSync` are descriptive extension contracts rather than Lightning-owned terminology. The default environment must therefore carry SyntheticMind-owned language while retaining the standard rank and rendezvous meanings.
    2. Plugin duplication is defined by capability family, not concrete subclass. Strategy construction records explicitly configured families separately from lazily supplied defaults so Trainer conflict detection does not reject ordinary default composition.
    3. Checkpoint I/O belongs behind the strategy because automatic callbacks, public saves, restoration, deletion, distributed rank ownership, storage options, and future storage backends must share one extension boundary.
    4. An asynchronous save must clone mutable tensors and arrays before returning, serialize writes, wait before load or removal, recreate its executor after stage teardown, and report the original write failure even when base teardown also fails.
    5. Layer synchronization precedes precision conversion and distributed wrapping, but reversal must target the post-conversion model. Retaining only the pre-conversion identity breaks custom precision plugins that return a replacement `Module`.
    6. Cluster environments are runtime topology authorities. Local launch may set rank and world size, while externally managed execution must retain its supplied ranks and rendezvous values and be checked against the configured device and node topology.
    7. The standard map-location contract includes callable and mapping forms in addition to device and string values. Narrowing this parameter would silently break custom checkpoint consumers even when ordinary CPU restoration passes.
    8. Default remote protocol support and concrete scheduler environments are separate integrations above the core extension contract and must remain explicit gaps until executable behavior exists.

| Memory | Dataloader Reload and Resume Semantics | Date: 19 August 2026 | Time: 06:04 PM | Name: Ghost |

    1. Reload cadence must be measured from the epoch in which a loader was actually requested. Absolute epoch modulo fails after checkpoint continuation and incorrectly couples validation construction to training cadence.
    2. Training and validation require separate reload coordinates because validation may run less frequently, may run several times within one epoch, and may execute once during sanity checking before real training begins.
    3. A sanity loader counts as the validation reload anchor only when the current epoch is validation-eligible. Otherwise the first scheduled validation must request a fresh loader even when the sanity loader object still exists.
    4. Checkpoint progress and datamodule state must be available before `train_dataloader()` executes. Loading the payload once, exposing its progress coordinate early, restoring data state after setup, and completing model and optimizer restoration later avoids duplicate I/O and stale loader construction.
    5. `DataModule.trainer` is a public extension dependency rather than a convenience attribute. It must be attached before data setup and loader hooks and restored in the multiprocessing parent alongside the module's Trainer reference.
    6. Every new loader must pass through the same validation, overfit sampling, distributed sampling, worker warning, rank synchronization, and effective-length path. Reloading through a smaller duplicate branch causes sampler and horizon drift.
    7. Disabled training and validation paths must not request their loaders. Lazy construction prevents unused hooks from raising or performing expensive work and aligns capability detection with overridden step and dataloader methods.
    8. Real execution matched the pinned infrequent and frequent validation matrices, relative checkpoint cadence, one-read custom checkpoint I/O, dynamic lengths, custom batch samplers, failure teardown, MPS, Gloo, and every supported local launch model.

| Memory | Artifact Paths and Remote Filesystem Semantics | Date: 19 August 2026 | Time: 06:38 PM | Name: Ghost |

    1. Converting a protocol URI through `Path` is data corruption rather than normalization because strings such as `memory://`, `s3://`, and `gs://` lose their protocol meaning. Local normalization and remote preservation require an explicit protocol boundary.
    2. Logger version discovery must remain observational. Creating a run directory while constructing independently initialized distributed ranks allows later ranks to select a different version before process coordination begins.
    3. `Trainer.log_dir` performs a strategy broadcast and is therefore an all-rank collective. A rank-zero guard may control filesystem mutation only after every rank has resolved the directory; reversing that order creates a deterministic distributed stall.
    4. Generic object storage does not guarantee append mode. Durable CSV and line-oriented profiler output must rewrite preserved content or use a storage-specific transactional mechanism instead of assuming local-file semantics from a permissive in-memory test backend.
    5. Native profilers that accept only local filenames require temporary local staging followed by protocol-aware upload. Passing a remote URI directly can create a malformed local directory or fail inside third-party code that knows nothing about fsspec.
    6. Symbolic-link checkpoint aliases are meaningful only on local filesystems. Remote `save_last="link"` must fail explicitly, while ordinary remote last checkpoints remain regular persisted objects.
    7. TensorBoard's current writer directly supports the exercised fsspec memory URI. Optional-backend behavior must still be executed after dependency installation because fallback CSV success does not prove the preferred backend.
    8. Fast development suppression includes auxiliary runtime telemetry. Replacing the experiment logger with a no-operation logger is insufficient if a callback interprets that placeholder as permission to create its own artifact directory.

| Memory | Hyperparameter Capture and Publication Semantics | Date: 19 August 2026 | Time: 06:57 PM | Name: Ghost |

    1. Automatic hyperparameter logging means publishing user-saved model and datamodule construction state. Logging Trainer controls such as epoch limits, precision, accelerator, and strategy under this contract is semantically false even when the resulting record appears useful.
    2. `save_hyperparameters()` is an inheritance-sensitive capture operation. Constructor frames across the active `super()` chain must be collected in order, while explicit names, one mapping or namespace, dataclass initialization fields, ignore lists, and empty-container calls retain distinct meanings.
    3. `hparams` is mutable runtime state, while `hparams_initial` is a defensive deep copy of the state recorded by the latest capture operation. Automatic experiment publication consumes the initial mapping so later ordinary mutation cannot rewrite the run's declared construction inputs.
    4. `save_hyperparameters(logger=False)` suppresses automatic experiment publication for that source but does not suppress checkpoint persistence. The logger flag must change even when the supplied container is empty.
    5. Model and datamodule mappings merge only when shared keys have the same concrete type and value. Tensors require identity rather than element equality, and `_class_path` is excluded because it is configuration-instantiation metadata rather than an experiment hyperparameter.
    6. OmegaConf must remain a live container at the user boundary or interpolation semantics are lost. Checkpoints therefore retain the container and its type metadata, while experiment loggers receive backend-safe normalized values.
    7. Rank-zero persistence does not remove the need for all ranks to own equivalent captured state. Real two-process execution verified that every rank participates in the stage while exactly one rank invokes the logger.
    8. OmegaConf `2.3.1` was installed through the authorized Python package path to exercise preservation and checkpoint round trips directly rather than claiming optional-configuration support from structural inspection.

| Memory | Trainer Runtime State and Registry Classification | Date: 19 August 2026 | Time: 07:11 PM | Name: Ghost |

    1. Entry-point state and running-stage state are different coordinates. A fit entry point may move through sanity checking, training, and validation, so public state properties must derive from the fine-grained running stage rather than the outer function.
    2. Sanity checking cannot coexist with a private boolean and a separately mutated validation stage without drift. One authoritative running-stage value lets callbacks, modules, timers, checkpointing, profiling, and presentation observe the same lifecycle truth.
    3. The strategy-managed `model` may be a distributed or precision wrapper, while the user `module` remains the unwrapped extension identity. Conflating them breaks both forward execution inspection and inherited hook expectations.
    4. Distributed sampler construction should receive explicit `num_replicas` and `rank` from the strategy contract. Depending only on ambient process-group discovery hides topology ownership and weakens custom strategy compatibility.
    5. A parameter present on a mature Trainer is not automatically an industry convention. The pinned `model_registry` and `suggest_integrations` parameters are respectively a Lightning Cloud product integration and promotional surface, so mechanical adoption would violate SyntheticMind ownership without adding universal core capability.
    6. Last-batch inspection for unsized iterables requires fetcher lookahead or equivalent progress state, and signal receipt requires process-lifecycle ownership. Neither can be implemented truthfully as a derived convenience boolean.

| Memory | Module Initialization Semantics | Date: 19 August 2026 | Time: 07:23 PM | Name: Ghost |

    1. Device, dtype, and empty allocation are module-construction lifecycle concerns and therefore belong behind the strategy rather than as independent Trainer-side global mutations.
    2. Mixed precision is an execution contract rather than a reduced-precision parameter-storage contract. Its initialization context retains float32, while true precision changes the floating default dtype during construction.
    3. PyTorch 2.9 does not route every `torch.nn.init` operation through a single dispatch shape. `ones_`, `zeros_`, `eye_`, `dirac_`, sparse initialization, output arguments, indexed assignment, and private helper calls require behavior-based suppression.
    4. Empty initialization must also preserve the random stream. Preventing final tensor mutation while allowing an initializer's intermediate random permutation still changes subsequent model behavior and is not a correct no-initialization contract.
    5. Current single-device and distributed data-parallel strategies should allocate uninitialized real-device storage. Meta tensors and sharded construction have different materialization obligations and belong to future specialized strategy overrides.
    6. Initialization contexts must restore process-global device and dtype state on exceptions. A successful allocation probe alone cannot establish safe reusable context behavior.
    7. Multiprocessing state reconciliation must update every public identity backed by worker state. Restoring only `Trainer.module` leaves `Trainer.model` stale because the latter delegates to strategy ownership.

| Memory | Signal and Final-Batch Semantics | Date: 19 August 2026 | Time: 07:42 PM | Name: Ghost |

    1. A signal handler should perform minimal process-safe work: record receipt, notify locally owned workers once, and preserve any pre-existing callable handler. Framework exceptions belong at controlled loop boundaries where cleanup and distributed coordination remain valid.
    2. The conventional terminated-process status is `128 + SIGTERM`, which is `143` on the supported runtime. Preserving this value through a `SystemExit` subtype lets callers distinguish requested termination from an ordinary training defect.
    3. Signal registration is process-global but Trainer stages and launch contexts may nest. A registration depth prevents an inner teardown from restoring the caller's handler while an outer lifecycle still depends on SyntheticMind ownership.
    4. Distributed logical-OR decisions are safe only when every rank reaches the collective in the same order. Unrepeated prediction partitions can contain different batch counts, so per-batch signal reduction deadlocks even though fit, validation, and test currently iterate equally.
    5. Truthful last-batch state for an unsized iterable requires one-batch lookahead. Catching `StopIteration` after processing the previous batch is too late for gradient accumulation, `no_sync`, validation cadence, or batch hooks that require the final boundary during execution.
    6. A finite batch limit must make the limited batch final without reading one additional source item. This matters for stateful streams, expensive generators, remote readers, and exact checkpoint continuation even when the extra item would otherwise be discarded.
    7. `is_last_batch` is runtime control state rather than presentation metadata. It must be set before `on_train_batch_start`, forced by the standard `-1` early-exit response, retained after fit for inspection, and transferred from multiprocessing workers to their parent.
    8. Process launchers require bidirectional signal semantics: the dormant parent must forward termination to children, and a child signal exit must return termination state rather than being mislabeled as a generic distributed worker crash.

| Memory | Multiple-Dataloader Runtime Semantics | Date: 19 August 2026 | Time: 08:40 PM | Name: Ghost |

    1. A loader collection is not equivalent to a sequence of independent stages. Simultaneous modes produce one structured training batch and one global batch index, while sequential evaluation and prediction reset the batch index and expose the producing `dataloader_idx` for each loader.
    2. Per-loader limits in simultaneous modes determine the combined stopping horizon through minimum or maximum composition; applying each limit inside the child iterator changes cycling and padding behavior and therefore silently changes the sampled data.
    3. Multiple evaluation metrics require loader-local accumulation before flattening into callback and logger state. The standard `/dataloader_idx_N` suffix prevents collisions, and disabling it transfers collision responsibility to the user-selected metric names.
    4. Prediction retention defaults to enabled for single-device and subprocess execution and disabled for spawn and fork, where arbitrary output transfer lacks a truthful aggregation contract. Unequal distributed prediction uses unrepeated samplers and cannot contain per-batch collectives.
    5. An outer subprocess launcher must not perform a synchronized signal check after rank zero returns from a stage while peer ranks continue into the next user-program stage. That unmatched collective can pair with an unrelated later collective and deadlock the process lifetime.
    6. Persistent PyTorch dataloader workers must be released by dropping the multiprocessing iterator reference and resetting the loader iterator rather than explicitly invoking the private shutdown method; the pinned implementation treats the latter as a regression path.
    7. The inherited `test_step` fallback must forward `dataloader_idx` when the overridden `validation_step` accepts it, while retaining compatibility with legacy two-argument validation overrides. Discarding the index produces valid-looking but incorrectly attributed results.
    8. Stateful loader cursor restoration and deterministic prefix replay are competing resume mechanisms. Integrating both without an explicit ownership rule can skip the completed prefix twice, so Trainer-level state loading remains open until exact continuation is proven.

| Memory | Direct Data Sources and Transfer Ordering | Date: 19 August 2026 | Time: 08:58 PM | Name: Ghost |

    1. Direct dataloaders, a datamodule, and model dataloader hooks are alternative data owners rather than interchangeable objects. One explicit precedence and conflict rule prevents setup, checkpoint, and inheritance behavior from depending on accidental truthiness.
    2. Positional datamodule compatibility must be resolved before validating the direct-loader argument because the standard entry points historically accept a `DataModule` in that position.
    3. Model-owned loaders may depend on state created by `configure_model` or `setup`; requesting them before those hooks silently narrows the established lifecycle contract.
    4. `prepare_data` on the model remains meaningful without a datamodule. Treating all data preparation as datamodule-owned bypasses a public model extension point for direct and model-owned sources.
    5. Fabricating a datamodule around direct loaders corrupts observable identity and may trigger lifecycle, hyperparameter, state-dictionary, and teardown behavior that the caller never requested.
    6. Distributed entry-point invocation must pass the optional datamodule by keyword after direct loader parameters enter the public signature; positional forwarding otherwise reclassifies a datamodule as data.
    7. Batch-start hooks and step argument tuples must be formed after precision conversion and device transfer. Constructing either beforehand retains references to the CPU batch even if a later local variable is replaced with its device-resident form.
    8. Cursor-aware checkpoint continuation must distinguish stateful loader restoration from stateless deterministic prefix replay at the same structural granularity as nested loader leaves.
