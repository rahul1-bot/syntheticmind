| Memory | Public Machine-Learning Compatibility | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. Public compatibility is semantic rather than cosmetic. Method names, parameter names and order, defaults, accepted values, return meanings, hook timing, state counters, and failure behavior jointly define the contract users inherit.
    2. PyTorch Lightning is a mature source of capability evidence, but its branded types and internal decomposition are not industry standards. SyntheticMind adopts universal concepts and owns its implementation language.
    3. An inherited extension point is incomplete until cooperative initialization, method-resolution order, override discovery, launcher transport, exception behavior, and every runtime dispatch site agree with the public signature.
    4. Surface comparison cannot establish capability. A parameter that is accepted but ignored, a hook that is defined but never dispatched, or a property backed by stale state is a defect rather than partial parity.
    5. Divergence from an established concept requires genuinely different semantics and explicit runtime enforcement; renaming a familiar concept only creates translation debt.

| Memory | Sequential Runtime Hardening | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. Runtime work must proceed from ownership and lifecycle foundations toward higher-level features. Isolated changes inside an unmapped loop create contradictory counters, duplicated cleanup, and unmatched distributed collectives.
    2. Every phase begins with the complete local execution path and pinned reference behavior, then defines observable acceptance criteria before implementation.
    3. Cross-cutting changes require regression execution against earlier persistence, launch, callback, metric, and device paths rather than validation of the new method alone.
    4. Git preserves chronology and exact diffs. Ledgers preserve only reusable facts, current architecture, and actionable direction.
    5. Uncertainty must trigger additional evidence from installed behavior, repository history, official primary sources, or a focused ignored reference checkout rather than a placeholder contract.

| Memory | Lifecycle and Failure Ownership | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. Stage state, module modes, hook context, progress state, and active exceptions require exception-safe restoration on success, early exit, and every failure path.
    2. Cleanup is a transaction across Strategy, datamodule, callbacks, module, logger, profiler, and progress presentation. The first causal failure remains primary while later cleanup failures become attached evidence.
    3. A completion record may be rendered only after failure-prone end hooks and teardown have succeeded; otherwise the console states a lie immediately before the run fails.
    4. Zero-batch stages, sanity validation, fit-owned validation, standalone evaluation, prediction, interruption, and setup failure each expose different lifecycle edges and require direct execution.
    5. Nested hook contexts must restore the previous hook name rather than clearing it unconditionally, because hooks can invoke nested framework operations.

| Memory | Optimization and Progress Accounting | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. Optimization compatibility is an execution contract. A closure owns training-step execution, zeroing, backward, pre-step hooks, clipping, precision handling, and the raw optimizer step in their required order.
    2. `global_step` advances on real managed optimizer steps, not batches. This distinction controls maximum steps, schedulers, checkpoint names, validation timing, and manual multi-optimizer training.
    3. The underlying algorithm remains `torch.optim.Optimizer`; `SyntheticOptimizer` is a transparent SyntheticMind-managed adapter for runtime orchestration and explicit raw access remains available.
    4. Parameter toggling must restore originally frozen, shared, unmanaged, and optimizer-owned parameter states even when the enclosed operation fails.
    5. Step schedulers follow optimizer progress and configured frequency, while epoch plateau schedulers run only after their monitored metric exists.
    6. Preconstructed optimizer state may exist on CPU before model placement. Recursive movement must place algorithm state on the execution device without corrupting state such as optimizer step counters.

| Memory | Distributed Execution and Collective Safety | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. Every collective must be reached by exactly the participating ranks in matching order. A rank-zero return before another rank enters a later signal or metric collective produces a deadlock rather than a local error.
    2. Parent-side and worker-side configuration must be separated deliberately. Spawn and fork workers own live execution, while the dormant parent requires explicit, serialization-safe state reconciliation.
    3. Distributed sampler reconstruction must preserve concrete dataloader subclasses and available constructor state; replacing every loader with a plain PyTorch DataLoader destroys stateful behavior.
    4. Rank-zero rendering, logging, checkpoint writing, and model printing are output ownership rules, while metrics and control decisions that require agreement remain collective operations.
    5. External two-rank Gloo execution is the minimum proof for collective semantics because two local objects without a process group cannot expose ordering or gradient synchronization defects.

| Memory | Precision, Device, and Process Boundaries | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. Precision affects parameter construction, autocast, loss scaling, optimizer stepping, clipping, checkpoint state, and evaluation; changing only an autocast context is not precision support.
    2. Device placement must propagate through model parameters, buffers, optimizer state, batches, checkpoint map location, metrics, and multiprocessing reconciliation.
    3. On the current MPS runtime, recursive tensor transfer with `non_blocking=True` produced corrupted batches and non-finite loss. Non-blocking host transfer therefore remains CUDA-specific until another backend has executable safety evidence.
    4. Optional ecosystem imports can initialize process-sensitive systems. Lazy TorchMetrics discovery prevents ordinary numeric logging from destabilizing macOS fork execution.
    5. Python 3.14 runtime behavior is authoritative. Unsupported installed-runtime features, including unavailable compilation paths, must be reported as environment bounds rather than simulated success.

| Memory | Terminal Presentation and Timing Truth | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. Terminal presentation is a subscriber to runtime state, not a logger, profiler, scheduler, or second source of truth.
    2. One active TTY line, one completed-epoch record, and one final stage record provide sufficient default visibility; detailed telemetry belongs in structured logger and profiler outputs.
    3. Stage clocks must be independent, monotonic, pause training around validation, exclude sanity work where appropriate, and close active segments during failure cleanup.
    4. Unknown loader horizons must remain unknown. Counts are truthful; fabricated percentages and estimated completion times are not.
    5. Interactive output may use in-place rendering, while redirected, subprocess, scheduler, and continuous-integration output must remain stable plain text without control sequences.
    6. Progress-selected metrics, callback-monitoring metrics, and experiment-logger metrics require separate routing and reset boundaries to prevent cross-stage leakage.

| Memory | Data Ownership and Multiple Loaders | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. Model, datamodule, and direct Trainer dataloaders are distinct ownership paths. Fabricating an adapter changes observable lifecycle, hyperparameter, state, and teardown identity.
    2. Nested loaders require one structural owner for flattening, combination mode, length, reset, iterator cleanup, loader indices, transfer hooks, and prediction reconstruction.
    3. Metric identity includes the originating hook and applicable dataloader index; public-name-only storage causes collisions between hooks and sequential loaders.
    4. Unsized streams require lookahead-aware boundaries because the loop cannot infer final-batch status from length. Prefetch ownership must not advance a saved cursor beyond the last batch actually returned.
    5. Prediction uses unrepeated distributed sampling because evaluation padding duplicates samples and corrupts returned prediction cardinality.

| Memory | Stateful Continuation and Checkpoint Truth | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. A restored stateful cursor owns physical data position, while checkpoint loop progress owns logical batch, accumulation, validation, and optimizer coordinates. Applying deterministic prefix replay after restoring the cursor skips data twice.
    2. Mixed stateful and stateless loader collections legitimately continue different physical paths; their shared logical progress must still remain coherent.
    3. Unsized lookahead checkpoints must snapshot the cursor after the returned batch but before the next unseen prefetch, including the initial cursor when the first batch fails.
    4. Natural epoch exhaustion clears active cursor continuation, while premature step termination retains it for the next Trainer or multiprocessing parent.
    5. Random continuation includes Python, NumPy, torch, and accelerator state. Restricted checkpoint loading requires safe serializable representations without surrendering exact stochastic continuation.
    6. Atomic checkpoint writes must expose either the prior complete file or the new complete file and remove failed temporary artifacts.

| Memory | Metric Routing and Tensor State | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. Logger, callback, progress, evaluation-result, checkpoint-monitor, scheduler, and early-stopping consumers have distinct metric contracts and cannot be reconstructed safely from one public-name dictionary.
    2. Epoch means synchronize numerator and denominator, not already averaged rank values. Sum, minimum, and maximum retain their own collective operations and optional process-group ownership.
    3. Public callback and logged metric inspection remains tensor-valued so dtype, device, and requested graph state survive; scalar conversion belongs only at logger, terminal, and result-record boundaries.
    4. `enable_graph=True` must reach cached step computation and weighted epoch reduction even when evaluation runs inside a no-gradient context. The default path must detach inside the reducer itself.
    5. TorchMetrics owns its stateful reduction and synchronization. SyntheticMind owns registration discovery, routing, reset timing, and lazy optional loading.
    6. Multiprocessing reconciliation detaches metric tensors because a live autograd graph cannot cross process ownership meaningfully.

| Memory | Model Extension and Runtime Access | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. Model-owned callbacks must compose before setup and checkpoint restoration, validate transactionally, remain stable across repeated stages, and preserve checkpoint callbacks at the final ordering boundary.
    2. Nested SyntheticMind modules require recursive Trainer attachment, while serialization must remove the live Trainer back-reference to avoid capturing the complete runtime graph.
    3. `Module.print` is globally rank-zero-owned, while `Trainer.print` is local-rank-zero-owned per node. Rank suppression occurs before progress dispatch and file output.
    4. Python print compatibility requires exact `args`, `sep`, `end`, `file`, and `flush` meanings; progress-safe output belongs to the configured callback rather than model-specific terminal code.
    5. `Module.loggers` is a live view of Trainer ownership, while detached access returns an independent empty collection.

| Memory | Collective Structure and Autograd Ownership | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. The model collective wrapper owns recursive scalar, NumPy, tensor, and collection conversion; the Strategy owns process-group communication and gradient behavior.
    2. World-size-one gathering is an identity and does not add a leading dimension. Distributed gathering stacks rank results according to the selected group's size and order.
    3. A collective is already a synchronization boundary. Adding a barrier increases cost and deadlock surface without improving correctness.
    4. Graph-free gathering uses ordinary collective buffers, while `sync_grads=True` requires PyTorch's differentiable distributed collective so backward contributions cross rank boundaries.
    5. Equal tensor shapes and participation by every selected rank are preconditions; uneven data requires a separate explicit padding or object-gather protocol.
    6. Named tuple reconstruction preserves concrete user collection contracts after recursive tensor transformation.

| Memory | Checkpoint Reconstruction Ownership | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. Class checkpoint loading is constructor reconstruction followed by state restoration, not a state-dictionary shortcut. Hyperparameters must create the correct subclass architecture before weights can be applied.
    2. Explicit constructor values override saved values, unknown values are filtered when unsupported, and wrapper-style `hparams` constructors receive overrides inside the wrapped mapping.
    3. `configure_model` runs before `on_load_checkpoint`, and the load hook runs before `load_state_dict`, enabling deferred layers and checkpoint payload mutation.
    4. `strict=False` permits missing and unexpected keys but never tensor shape mismatches. Relaxed differences remain visible through warnings.
    5. Mapped state tensors determine returned model placement; automatic mapping retains available accelerator tags and sends unavailable accelerator storage to CPU.
    6. Trainer restoration and class reconstruction share hook and strictness meanings even though only Trainer restoration includes optimizer, scheduler, callback, datamodule, precision, loop, loader, and random state.
    7. Checkpoint-controlled import paths are executable behavior. SyntheticMind rejects external instantiators instead of inheriting a vendor CLI allowlist.

| Memory | Verification and Environment Discipline | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. `ruff`, `mypy`, `pyright`, every other linter, and every static type checker are prohibited.
    2. Guideline compliance is reviewed directly, while behavior is proven through focused assertions, complete stage execution, exception paths, signatures, inheritance inspection, compilation, and diff review.
    3. CPU-only success cannot establish accelerator or distributed capability; MPS, launcher-specific, and direct process-group paths are selected according to real ownership boundaries.
    4. Ignored verification programs under `external/` are disposable evidence and never enter the tracked project history.
    5. A completed phase is not synchronized until the accepted commit exists on both local `dev` and `origin/dev` at the same revision.
