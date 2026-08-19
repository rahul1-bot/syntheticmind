| Strategic Note | SyntheticMind Development Intent | Date: 19 August 2026 | Time: 04:31 AM | Name: Ghost |

    1. SyntheticMind is maintained as Rahul's private machine-learning control layer for study, research, experimentation, and creative enjoyment.
    2. The architecture is valuable because it exposes the complete workflow rather than hiding consequential behavior behind a framework boundary.
    3. That control is useful only when the behavior is correct. The current implementation is incomplete and must be hardened sequentially while the missing capability surface is implemented and verified.
    4. Review decisions must favor deterministic execution, explicit state transitions, recoverable checkpoints, correct optimization accounting, and truthful interfaces while progressing toward full user-facing PyTorch Lightning capability parity.
    5. The public API must speak established machine-learning language. Internally invented aliases for standard concepts are prohibited unless a real semantic distinction requires them.

| Strategic Note | Working Branch Discipline | Date: 19 August 2026 | Time: 04:31 AM | Name: Ghost |

    1. The stable baseline and all root ledgers are retained on `main` and `origin/main`.
    2. Active implementation work occurs on `dev` and is published to `origin/dev` after every completed development commit.
    3. A commit is not operationally complete while local `dev` and `origin/dev` differ.
    4. Promotion from `dev` to `main` requires an explicit stabilization decision supported by relevant verification.
    5. The repository remains private because it is personal research and study infrastructure, not a public framework release.

| Strategic Note | Capability Parity and Ordered Hardening | Date: 19 August 2026 | Time: 05:19 AM | Name: Ghost |

    1. SyntheticMind's end state is full user-facing capability parity with core PyTorch Lightning, with Fabric, public documentation, release work, and replication of the upstream PyTorch Lightning test suite excluded.
    2. Established machine-learning and PyTorch variable names, method names, parameter contracts, state meanings, and inherited extension hooks are compatibility requirements rather than cosmetic preferences. Lightning-specific branding is reference ownership, not an industry convention.
    3. SyntheticMind will preserve its own internal architecture and apply the generic Python guidelines while matching the external semantics that machine-learning practitioners already understand.
    4. Exceptions require genuinely different semantics or algorithms and an explicit rationale; private terminology must never disguise an established concept.
    5. Hardening proceeds sequentially from source-backed compatibility contracts through foundational runtime layers. No isolated feature will be inserted into the middle of an unmapped execution path.
    6. Existing code comments remain in place, and new source-code comments are deferred during the current phase.
    7. Each capability requires focused local verification even though upstream test-suite replication is excluded; an unexercised interface is not accepted as implemented.

| Strategic Note | Research Persistence and Operational Memory | Date: 19 August 2026 | Time: 08:21 AM | Name: Ghost |

    1. Unknown implementation details must be resolved through source inspection, executable probes, installed-environment checks, current primary sources, and repository history before accepting a compromise.
    2. Carefully selected external repositories may be cloned into the ignored `external/` tree when they answer a concrete engineering question that the pinned PyTorch Lightning reference does not resolve.
    3. The ledgers preserve current direction and reusable knowledge only. Git history preserves chronological implementation detail, and duplication between those systems is prohibited.
    4. Every completed phase must leave local `dev` and `origin/dev` synchronized after verification and review.

| Strategic Note | Extension Contracts and Verification Boundary | Date: 19 August 2026 | Time: 09:03 AM | Name: Ghost |

    1. SyntheticMind's base classes must preserve standard machine-learning terminology, inheritance chains, method-resolution behavior, override names, signatures, parameter meanings, defaults, and lifecycle effects because downstream models and integrations extend these contracts directly.
    2. `code-guidelines.md` governs every implementation decision, while PyTorch and PyTorch Lightning conventions govern public concepts that users already understand.
    3. Ruff and other linters are excluded. Verification proceeds through focused executable behavior, signature and inheritance inspection, source comparison, failure-path probes, and compilation.

| Strategic Note | Optimization Foundation | Date: 19 August 2026 | Time: 11:37 AM | Name: Ghost |

    1. The optimization boundary must preserve meaningful optimizer access, scheduler access, manual backward, optimizer stepping, zeroing, clipping, and scheduler stepping contracts without importing Lightning's product identity.
    2. The accepted `configure_optimizers` structures are normalized behind a private parser, allowing SyntheticMind to retain a smaller internal architecture without changing the public machine-learning contract.
    3. Closure execution, manual multi-optimizer behavior, parameter toggling, scheduler cadence, mixed-precision ordering, checkpoint restoration, and optimizer-state device placement are executable capabilities rather than declared placeholders.
    4. `LightningOptimizer` is a Lightning-owned managed wrapper rather than an industry optimizer term. SyntheticMind's equivalent public wrapper is `SyntheticOptimizer`, with the actual algorithm remaining a `torch.optim.Optimizer`.
    5. ✅ The ownership correction removed `LightningOptimizer`, `use_pl_optimizer`, `_lightning_optimizers`, `LRSchedulerPLType`, and `pl_module`. The resulting `SyntheticOptimizer`, explicit raw-access flag, generic scheduler union, and `module` callback parameter were reverified across extension and execution paths.
    6. Existing comments remain untouched, new code comments remain prohibited, and no linter may substitute for direct guideline review and behavioral evidence.

| Strategic Note | Console Experience Boundary | Date: 19 August 2026 | Time: 11:28 AM | Name: Ghost |

    1. SyntheticMind's terminal experience should operate as a coherent live run console rather than independent progress bars interrupted by raw logger sentences.
    2. Visual polish is subordinate to measurement truth. Independent stage timers, sanity exclusion, stage-isolated progress metrics, boundary resets, and prediction batch totals now provide the verified substrate required by the renderer.
    3. Existing runtime telemetry should feed the console through structured events so terminal rendering, CSV and TensorBoard persistence, callbacks, and distributed execution remain separate concerns.
    4. The approved renderer uses one active TTY line, one completed-epoch line, and one final stage line. It shows only stage, progress, at most two selected metrics, and honest ETA by default; all detailed telemetry stays in structured experiment outputs.
    5. Non-interactive output remains stable plain text, distributed rendering remains rank-zero only, and unknown dataloader lengths produce counts rather than false percentages or ETA values.
    6. Callback monitoring metrics, experiment logging, profiling telemetry, and terminal-selected metrics remain separate channels. The renderer consumes runtime truth but does not become another authority for it.
    7. ✅ One renderer now covers fit, validation, testing, and prediction; routine framework information no longer competes with the active line, and completed epochs are deferred across failure-prone hook boundaries before becoming permanent output.
    8. ✅ Trainer cleanup is now one exception-safe transaction. Strategy, datamodule, ordinary callbacks, module, and logger complete their cleanup attempts before the progress callback is permitted to declare a successful stage outcome.
    9. Runtime, setup, interruption, exit, exception-hook, and teardown failures preserve the first causal exception while retaining secondary failures as notes; module setup and teardown are now dispatched for every public stage and all stage markers clear after failure.

| Strategic Note | Standard Training Limits | Date: 19 August 2026 | Time: 12:41 PM | Name: Ghost |

    1. ✅ SyntheticMind now distinguishes finite epoch budgets, finite optimizer-step budgets, and explicit unlimited horizons using the standard `max_epochs` and `max_steps` meanings.
    2. Minimum epoch and step requirements constrain only cooperative stopping. EarlyStopping may request termination immediately, but the fit continues until both minima are satisfied unless a hard maximum is reached first.
    3. Automatic optimization, accumulation, manual managed optimizers, LBFGS closures, schedulers, and checkpoint continuation share one authoritative optimizer-step counter and termination boundary.
    4. `Trainer.should_stop` is the public stop contract; loop state, EarlyStopping, and downstream callbacks no longer require a project-specific state access path.
    5. Unlimited execution is represented honestly in step estimates and terminal epoch labels, while checkpoint operations remain quiet and atomic-save interruption leaves no temporary artifact.
    6. The next parameter correction is validation scheduling: replace project-specific scope and end-validation controls with the established `val_check_interval` and `check_val_every_n_epoch` contract before expanding further.

| Strategic Note | Standard Validation Cadence | Date: 19 August 2026 | Time: 12:55 PM | Name: Ghost |

    1. ✅ Validation cadence is now expressed exclusively through the standard `val_check_interval` and `check_val_every_n_epoch` contract rather than SyntheticMind-specific scope and epoch-end switches.
    2. Batch cadence follows raw training progress, float cadence follows the effective epoch horizon, and duration cadence follows monotonic elapsed time. Optimizer accumulation cannot distort any of these meanings.
    3. Epoch eligibility, sanity checks, cooperative stopping, unsized dataloaders, dataloader reloads, and disabled validation share one normalized scheduling authority.
    4. Cumulative raw-batch checkpoint state preserves cross-epoch integer cadence during exact mid-epoch restoration, while final-batch checkpoints remain true epoch boundaries.
    5. The next sequential boundary is the standard `limit_train_batches`, `limit_val_batches`, `limit_test_batches`, `limit_predict_batches`, and sanity-step semantics on which later Trainer capabilities depend.

| Strategic Note | Standard Data-Pass Limits | Date: 19 August 2026 | Time: 01:02 PM | Name: Ghost |

    1. ✅ Every stage now resolves its batch budget through one standard contract owned by the Trainer rather than a separate loop-local interpretation.
    2. Integer counts, fractional counts, full-loader values, zero disabling, empty loaders, unsized loaders, and impossible sub-batch fractions now retain distinct and verified meanings.
    3. Sanity validation defaults to two batches, clips to the real validation budget, supports all-batch execution through `-1`, and never replays during checkpoint continuation.
    4. Effective training, validation, sanity, testing, and prediction batch counts are available through the established public Trainer inspection properties.
    5. The next foundational termination capability is the standard `max_time` contract, whose elapsed-time signal must remain cooperative rather than being mislabeled as a hard boundary.

| Strategic Note | Standard Wall-Clock Control | Date: 19 August 2026 | Time: 01:15 PM | Name: Ghost |

    1. ✅ `Trainer(max_time=...)` and the public `Timer` callback now provide standard wall-clock termination without importing Lightning branding or internal architecture.
    2. Duration expiration raises one synchronized cooperative stop request. Minimum epoch and step requirements may delay the exit, while the existing hard epoch and step limits retain precedence.
    3. The timer persists elapsed training time through checkpoints, prevents depleted resumes from consuming another batch, and counts validation wall time after training has begun.
    4. Public timing queries retain the latest stage semantics expected by users, while separate internal measurements provide truthful epoch presentation without redefining the public contract.
    5. The terminal remains restrained: Trainer-installed timers do not emit raw logger chatter, and a completed fit records `max_time reached` in the existing compact summary.
    6. The next sequential Trainer boundary must be selected from source-backed capability dependencies rather than inserting an isolated convenience in the middle of the runtime.

| Strategic Note | Standard Debugging Controls | Date: 19 August 2026 | Time: 01:26 PM | Name: Ghost |

    1. ✅ `fast_dev_run` is now a real all-stage smoke-test contract rather than a decorative constructor flag: it bounds execution, removes sanity work, restores standard validation cadence, and suppresses persistence without removing ordinary user callbacks.
    2. ✅ `overfit_batches` now selects a stable training and validation subset and disables shuffling across initial, reloaded, standalone, and distributed dataloader paths.
    3. The implementation remains SyntheticMind-owned. Its private no-operation logger and centralized sampler reconstruction provide the required behavior without copying Lightning namespaces or connector architecture.
    4. The next dependency-ordered boundary is hardware topology: `accelerator`, `strategy`, `devices`, and `num_nodes` must become one validated execution contract before precision and higher distributed capabilities are expanded.

| Strategic Note | Standard Hardware Topology | Date: 19 August 2026 | Time: 01:39 PM | Name: Ghost |

    1. ✅ Accelerator choice, ordered parallel devices, node count, strategy selection, process ranks, world size, and root-device placement now form one validated runtime contract rather than independent constructor hints.
    2. The public surface preserves established hardware terminology and defaults. Internal parsing and topology composition remain SyntheticMind-owned, with no Lightning-branded class or connector vocabulary introduced.
    3. CPU multi-process execution defaults to Gloo, CUDA execution defaults to NCCL, explicit GPU identifiers retain their declared order, and unsupported or contradictory configurations fail before model placement.
    4. Real CPU and MPS execution plus a two-process CPU `torchrun` fit verify the supported local hardware paths; CUDA parsing and rank mapping are verified through controlled availability probes because the current MacBook exposes no CUDA runtime.
    5. Topology alone is not distributed user parity. The next sequential phase is a local subprocess launcher that converts a multi-device Trainer request into the required worker processes while preserving entry-point arguments, failures, teardown, and parent-process results.

| Strategic Note | Distributed Process Ownership | Date: 19 August 2026 | Time: 02:17 PM | Name: Ghost |

    1. ✅ A multi-device Trainer can now create its own local DDP workers through standard `ddp`, `ddp_spawn`, `ddp_fork`, and `ddp_notebook` strategy names while externally launched `torchrun` execution remains supported.
    2. The default script launcher owns one process lifetime, not one Trainer stage. Consecutive stages reuse the same healthy process group, eliminating stale-rendezvous races without copying Lightning's connector architecture or product terminology.
    3. Spawn and fork workers reconcile rank-zero execution state into the parent; subprocess workers instead continue the same user program. These are intentionally different standard process models and must not be disguised behind one false state-transfer promise.
    4. Distributed failures are terminal to the affected local job, broken process groups are destroyed, launch mutations are restored when control returns, and worker observers prevent blocked collectives from leaving zombie processes.
    5. The unsized gradient-accumulation tail now synchronizes dense gradients, including parameters absent on an individual rank, before stepping. Sparse tail synchronization remains an explicit limitation rather than silent corruption.
    6. The next dependency-ordered boundary is the standard precision contract and its interaction with optimizers, strategies, devices, checkpoint restoration, and user-visible Trainer semantics.

| Strategic Note | Precision Ownership and Execution | Date: 19 August 2026 | Time: 03:01 PM | Name: Ghost |

    1. ✅ The universal Trainer precision contract is now one executable subsystem rather than precision strings interpreted independently by loops, optimizers, and devices.
    2. SyntheticMind retains descriptive, framework-owned precision classes while preserving the standard method, parameter, default, hook-order, inheritance, and state semantics required by downstream overrides.
    3. True and mixed precision operate across all public stages; custom backward methods, precision-connected wrappers, automatic and manual optimization, scaler checkpointing, clipping, skipped steps, fused optimizers, and distributed parent-state recovery share the same authority.
    4. The current PyTorch autocast cache defect around nested `no_grad` and `inference_mode` contexts is guarded explicitly, preventing a cached gradient-disabled cast from poisoning a later gradient-enabled forward in the same precision context.
    5. FP8 Transformer Engine configuration and optional dependency boundaries are represented truthfully, while CUDA and real FP8 execution remain hardware-unverified and may not be claimed from CPU, MPS, or mocked evidence.
    6. The next dependency-ordered boundary is Trainer execution context: standard inference-mode selection, anomaly detection, benchmarking, and deterministic-algorithm semantics must be reconciled before higher strategy and integration layers expand.

| Strategic Note | Execution Context Integrity | Date: 19 August 2026 | Time: 03:10 PM | Name: Ghost |

    1. ✅ Standard deterministic, benchmark, inference-mode, and anomaly-detection controls are now executable Trainer behavior rather than inert configuration vocabulary.
    2. Random seeding and deterministic execution are separate authorities. Reproducing random streams no longer changes performance or deterministic-algorithm policy behind the user's back.
    3. Evaluation context selection preserves the meaningful distinction between inference mode and no-gradient mode, including gradient re-entry for specialized validation algorithms and the Gloo compatibility boundary.
    4. Anomaly detection reaches the real backward execution path and restores process state after failure, while deterministic and benchmark settings survive spawned worker transfer.
    5. CPU, MPS, forked Gloo, and spawned DDP evidence establishes the supported local contract; CUDA-specific deterministic kernels remain bounded by the unavailable NVIDIA runtime.
    6. The next phase must resume the pinned Trainer capability surface in dependency order and select the earliest unresolved execution collaborator rather than introducing isolated convenience flags.

| Strategic Note | Experiment Logger Composition | Date: 19 August 2026 | Time: 03:26 PM | Name: Ghost |

    1. ✅ The Trainer now distinguishes default logging, explicit enablement, explicit disablement, one backend, and multiple backends through the established public configuration meanings.
    2. Every configured backend receives the same hyperparameters and metric events, while the first logger remains the standard singular integration surface and rank zero remains the sole persistent writer.
    3. Default run directories are SyntheticMind-owned, versioned lazily, and consistent across external, forked, and spawned distributed execution. Fast development runs preserve callable logger integrations without producing experiment artifacts.
    4. Step-cadence zero suppresses only step traffic, preserving epoch metrics. Cleanup attempts every backend even when one finalizer fails, preventing an early failure from abandoning later buffers or file handles.
    5. CPU, MPS, CSV, multiple custom backends, callback singleton, malformed configuration, teardown failure, and distributed ownership paths are verified; all modified files are type-clean and compile successfully without invoking a linter.
    6. The next sequential boundary is Trainer-owned default checkpoint composition. It must preserve standard callback ownership, directory semantics, disabling behavior, user conflicts, and fast-development suppression before model-summary or profiler composition is considered.

| Strategic Note | Checkpoint Integrity and Continuation | Date: 19 August 2026 | Time: 04:11 PM | Name: Ghost |

    1. ✅ Trainer-owned checkpoint composition now preserves standard enablement, disabling, default callback, generic extension, ordering, root-directory, fast-development, and public save and restore meanings.
    2. ModelCheckpoint is a SyntheticMind implementation of established checkpoint capability rather than a Lightning clone. Its public parameters and override contracts match the mature reference where the concepts are universal, while framework branding and unavailable integration promises remain excluded.
    3. Full checkpoints restore the complete training transaction and weights-only checkpoints restore the model boundary without stale optimizer or callback mutation. Standard state names are primary and earlier SyntheticMind state names remain load-compatible.
    4. Exact stochastic continuation is verified across shuffled data and dropout, including interruption inside an epoch. Correctness depends on preserving both dataloader reconstruction state and the random state at the unfinished computation boundary.
    5. Filesystem mutation remains rank-zero owned while checkpoint callback decisions and state remain synchronized across forked, spawned, and externally launched Gloo workers. CPU and MPS paths are executable; CUDA remains explicitly hardware-unverified.
    6. The next dependency-ordered phase is standard model-summary composition through `enable_model_summary`, followed only after its public callback and parameter-count behavior are understood end to end.

| Strategic Note | Model Summary Ownership | Date: 19 August 2026 | Time: 04:26 PM | Name: Ghost |

    1. ✅ Trainer-owned model summary composition now preserves standard enablement, explicit disabling, custom callback precedence, depth selection, callback inheritance, and fit-start timing.
    2. Summary data is computed from the configured, restored, device-placed, precision-connected model and includes nested structure, shapes when an example input exists, parameter ownership, train and evaluation modes, estimated storage, and FLOPs.
    3. The summary forward leaves no training mutation behind: temporary hooks, submodule modes, and all supported process and device random states return to their prior values even when the model raises.
    4. Distributed workers all execute the summary contract, but only global rank zero owns presentation. This behavior is verified under external Gloo, fork, and spawn rather than inferred from a rank check.
    5. The formal plain renderer remains the Trainer default under the approved restrained console policy; the standard Rich renderer is an explicit opt-in instead of an optional dependency silently taking terminal ownership.
    6. The next dependency-ordered boundary is the `profiler` parameter and its complete stage, action, artifact, logger, teardown, exception, and distributed ownership semantics.

| Strategic Note | Profiling Ownership and Observability | Date: 19 August 2026 | Time: 05:07 PM | Name: Ghost |

    1. ✅ SyntheticMind now has a standard profiler hierarchy and Trainer parameter contract without becoming a PyTorch Lightning clone or emitting Lightning-owned names.
    2. Profiling observes the real extension and execution boundaries across modules, datamodules, callbacks, strategies, batches, epochs, dataloaders, optimization, checkpoint persistence, and every public stage.
    3. The default pass-through path remains artifact-free, while selected profilers resolve output ownership through an explicit directory, the first logger, or the default root and isolate distributed ranks by filename.
    4. Exception and teardown handling preserve the causal runtime failure and close profiler state deterministically, including repeated stages and failures arising after the report has been written.
    5. CPU, MPS, and all supported local Gloo launch paths establish the available runtime contract; CUDA and XLA execution remain bounded by unavailable hardware or optional dependencies rather than being falsely claimed.
    6. The next dependency-ordered phase is the standard `barebones` Trainer mode and its coordinated disabling of nonessential runtime features.

| Strategic Note | Barebones Overhead Isolation | Date: 19 August 2026 | Time: 05:20 PM | Name: Ghost |

    1. ✅ Barebones mode now provides an executable framework-overhead baseline instead of a cosmetic flag, disabling every standard nonessential Trainer service through one validated constructor contract.
    2. Model logging is suppressed at its public entry point, preventing inherited `log` and `log_dict` calls from rebuilding metric work while preserving an actionable rank-zero warning.
    3. The mode does not amputate ordinary training semantics: callbacks, optimization, precision, distribution, validation, and wall-clock termination remain available unless their specific service is part of the standard exclusion set.
    4. Minimal stage announcements retain operational visibility without reviving progress rendering or dense framework chatter, and distributed execution emits the announcement from global rank zero only.
    5. Artifact isolation, exception propagation, teardown, explicit callback conflicts, CPU, MPS, and Gloo execution are verified. The next dependency-ordered boundary is the standard `plugins` parameter and the extension families it composes with strategies.

| Strategic Note | Plugin Extension Ownership | Date: 19 August 2026 | Time: 05:46 PM | Name: Ghost |

    1. ✅ SyntheticMind now composes the standard precision, checkpoint-I/O, cluster-environment, and layer-synchronization extension families without adopting Lightning-branded class names or connector architecture.
    2. Plugin ownership is singular and explicit: a family may be configured through the Trainer or strategy, never both, and every accepted method remains overridable under its established parameter and lifecycle contract.
    3. Checkpoint persistence now crosses one strategy-owned I/O boundary, including asynchronous snapshots and transactional failure handling; distributed rendezvous and ranks cross one cluster-environment boundary; synchronized layers cross one reversible fit-only boundary.
    4. Local, MPS, forked, spawned, subprocess, and externally launched execution verify that plugin composition survives the supported process models rather than merely passing constructor inspection.
    5. Remote default storage and managed scheduler environments remain truthful future work. The next dependency-ordered Trainer boundary is `reload_dataloaders_every_n_epochs`.
