| Strategic Direction | SyntheticMind Harness Hardening | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. SyntheticMind is Rahul's private machine-learning harness for FAU study, independent research, experimentation, and direct end-to-end control of machine-learning workflows.
    2. The harness is intentionally incomplete and must not be represented as production-ready, release-ready, or capability-complete.
    3. The active objective is full user-facing capability parity with core PyTorch Lightning while retaining an independent SyntheticMind architecture.
    4. Fabric, public documentation, release machinery, promotional work, and replication of the upstream PyTorch Lightning test suite are excluded.
    5. Code behavior, inherited extension compatibility, failure handling, and proportionate executable verification control the work queue.
    6. New source-code comments and docstrings remain prohibited. The active ground-up refactor explicitly targets rewritten legacy modules, so stale or misleading source prose is removed with the code it described rather than preserved as false architecture.

| Architecture Decision | Public Terminology and Framework Ownership | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. Established machine-learning and PyTorch terminology is a public compatibility boundary. Standard concepts retain their recognized names, parameter names, defaults, meanings, and lifecycle effects.
    2. Public base classes are inheritance contracts. Method-resolution order, cooperative initialization, overridable names, signatures, invocation frequency, return meanings, and state transitions require executable preservation.
    3. PyTorch Lightning is the pinned capability and behavior reference, not a namespace, product identity, implementation template, or source of vendor branding for SyntheticMind.
    4. New public terms are prohibited when they merely rename an established concept. Project-specific terminology requires genuinely different semantics enforced by implementation and verification.
    5. `LightningOptimizer` is vendor-owned wrapper terminology rather than an optimizer algorithm. SyntheticMind exposes `SyntheticOptimizer` for its managed adapter while the underlying algorithm remains `torch.optim.Optimizer`.
    6. Callback parameters use the SyntheticMind-owned name `module`; Lightning-branded `pl_module`, `use_pl_optimizer`, and equivalent copied identifiers remain prohibited.
    7. Internal architecture follows `/Users/rahulsawhney/Library/CloudStorage/OneDrive-Personal/Documents/StudyDocuments/Rahul/code-guidelines.md` without blindly reproducing Lightning internals. Historical repository-local guideline copies are not authoritative.

| Repository Policy | Private Development and Reference Boundaries | Date: 04 September 2026 | Time: 06:28 PM | Name: Ghost |

    1. The GitHub repository is private, `origin` is canonical, `main` is the stable baseline, and active implementation occurs on `dev`.
    2. Every completed development phase must be committed locally, pushed to `origin/dev`, and verified to leave local `dev` and `origin/dev` at the same commit.
    3. Promotion to `main` requires an explicit stabilization decision; ordinary hardening commits do not enter `main` automatically.
    4. Comparative source checkouts live only under ignored `external/`; the active PyTorch Lightning reference is pinned there at revision `21ac84116633b8c629da03c41e6d6ec2cfb15241` and remains read-only evidence.
    5. The obsolete two-gigabyte ignored `docs/` reference clone, stale optimization cache, and generated parent-scope probe artifacts were moved to recoverable Trash; additional source references may enter only `external/` for a concrete unresolved question.
    6. The four ledgers contain current direction, durable decisions, reusable knowledge, and live progress. Git commits, diffs, and logs are the authoritative chronology.

| Execution Doctrine | Sequential Capability Reconstruction | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. Hardening proceeds in dependency order from source-backed contracts through foundational state, lifecycle, execution, and higher-level capabilities.
    2. Each subsystem is traced end to end before modification, including public surfaces, inherited hooks, data flow, state ownership, failure paths, and interactions with completed layers.
    3. A phase is complete only after implementation, focused runtime evidence, failure probes, source and diff review, Python compilation, ledger maintenance, commit, push, and branch synchronization.
    4. Uncertainty triggers inspection of SyntheticMind, the pinned reference, installed runtime behavior, repository history, official primary sources, and carefully selected external implementations as needed.
    5. Placeholders, unused compatibility parameters, speculative interfaces, copied framework baggage, and surface-only parity claims are prohibited.
    6. Python packages may be installed with `python -m pip install --break-system-packages` when a verified dependency gap requires them and the existing environment has been inspected first.

| Verification Policy | Executable Evidence Without Static Analysis | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. `ruff`, `mypy`, `pyright`, every other linter, and every static type checker are prohibited and must not be invoked.
    2. `code-guidelines.md` compliance is established through deliberate direct source review rather than automated style or type analysis.
    3. Capability claims require focused runtime assertions, end-to-end execution, exception probes, signature and method-resolution inspection, source comparison, Python compilation, and diff validation where relevant.
    4. CPU, MPS, distributed, subprocess, spawn, fork, notebook, persistence, and optional-dependency paths are exercised in proportion to the capability's real execution surface.
    5. Historical references to static-analysis evidence describe work performed before the prohibition and do not authorize any future invocation.

| Architecture State | Verified Framework Runtime Baseline | Date: 04 September 2026 | Time: 05:23 PM | Name: Ghost |

    1. ✅ Fit, validation, test, prediction, sanity, failure, and teardown lifecycles preserve standard hook order, module modes, stage state, causal exceptions, and run controls.
    2. ✅ CPU, CUDA, MPS, precision, automatic and manual optimization, scheduler cadence, distributed launchers, samplers, signals, and parent-state reconciliation form an executable training runtime.
    3. ✅ Model, DataModule, direct, multiple, nested, sized, unsized, stateful, and distributed data paths preserve ownership, identity, limits, ordering, and exact continuation.
    4. ✅ Metric production, graph state, reduction, TorchMetrics, callback views, progress views, logger output, scheduler monitoring, early stopping, and checkpoint ranking remain distinct and coherent.
    5. ✅ Model, data, checkpoint, and callback hooks; model-provided callbacks; logger access; rank-safe printing; nested gathering; and model checkpoint reconstruction preserve established extension semantics.
    6. ✅ ConsoleRenderer owns terminal mechanics and owner-bound metric ordering without a lambda, while one ProgressBarCallback owns rank-zero stage presentation; the hook-rich callback retains an explicit concept-integrity size exception without annotation, prose, static, or pass-through debt.

| Architecture Decision | Atomic Sharded Checkpoint Generations | Date: 04 September 2026 | Time: 05:44 AM | Name: Ghost |

    1. Sharded checkpoints now store immutable distributed generations beneath one checkpoint directory and publish visibility through a strict versioned `active_generation` record rather than deleting the readable checkpoint before writing its replacement.
    2. The pointer records the active and immediately previous published UUIDv7 generation identifiers; readers that resolved the former active generation remain valid across one publication, while failed and unpublished generations are removed after the next successful commit.
    3. Local pointer replacement uses a same-directory temporary file, file synchronization, atomic replacement, and directory synchronization; non-local fsspec publication runs within the filesystem transaction boundary.
    4. The storage owner rejects incomplete generations, invalid or future pointer schemas, optimizer-incomplete resume state, and destructive full-to-sharded or sharded-to-full path collisions before replacing the existing artifact.
    5. Model-parallel checkpoint internals now occupy the semantic `strategies/model_parallel_checkpointing/` subpackage, preserving the established `ModelParallelStrategy` import and class identity while reducing the top-level strategy package from the topology-review threshold to five direct modules.
    6. Full and sharded uninterrupted-versus-resumed runs produce bit-identical three-step model state, and distributed failure injection confirms that shard, metadata, and pointer failures preserve the previously published generation on both ranks.

| Architecture Decision | Concept-First Framework Topology | Date: 04 September 2026 | Time: 09:00 AM | Name: Ghost |

    1. Rahul established a project-specific authority rule for SyntheticMind: machine-learning and distributed-systems concepts control module, class, and package decomposition before generic line-count or package-width guidance.
    2. Structural budgets remain review and subtraction signals, but they cannot force one coherent training concept into pass-through modules, one-consumer classes, or implementation-oriented peer abstractions.
    3. The current PyTorch Lightning source is the reference for recognized training vocabulary and conceptual placement, while its ABCs, free functions, broad annotations, historical compatibility layers, and oversized implementation details remain non-authoritative.
    4. Trainer connectors are admitted only for complete composition boundaries: accelerator topology, data attachment, callback configuration, full-trajectory checkpointing, metric and logger routing, and signal lifecycle.
    5. `Strategy` is a stateful execution-policy hierarchy rather than a behaviorless Protocol. `ParallelStrategy` is the shared distributed execution concept; DDP and model-parallel strategies inherit it and contain only algorithm-specific deltas.
    6. This decision supersedes the Protocol-only Strategy conclusion recorded at 07:05 AM. The earlier runtime evidence remains valid, but its decomposition judgment was rejected after source-level comparison exposed duplicated concrete behavior.

| Architecture Decision | Stateful Strategy and Model Initialization | Date: 04 September 2026 | Time: 06:23 PM | Name: Ghost |

    1. `Strategy` now owns universal model connection, wrapped-step dispatch, precision-mediated backward and optimizer stepping, single-process collective defaults, initialization context, and ordinary checkpoint collaboration without using an ABC.
    2. `ParallelStrategy` now owns the cluster environment, process-group lifecycle, rank topology, process launcher, distributed collectives, sampler coordinates, and distributed tail-gradient synchronization directly.
    3. `SingleDeviceStrategy` is a named execution policy with inherited single-device semantics; `DDPStrategy` contains only DDP wrapping, layer synchronization, and manual-optimization forward behavior; `ModelParallelStrategy` exposes named data and tensor mesh axes for researcher-owned tensor-parallel, composable FSDP2, or hybrid plans and rejects legacy FSDP1.
    4. The strategy-owned ModuleInitializer now intercepts the dynamic PyTorch override contract through a named recursive torch-function value schema, validates mode state, preserves RNG under empty initialization, and materializes meta parameters through explicit reset operations without `object` escape hatches.
    5. Direct source comparison found thirty-one method names repeated across all three former concrete strategies, nine identical method bodies across all three, and twenty-three identical bodies between DDP and model parallelism; the new hierarchy removes that duplication through true semantic inheritance.
    6. Empty and normal initialization RNG behavior, precision-directed construction, meta materialization, CPU and MPS execution, two-rank DDP, spawn and fork launch, nonzero-node multiprocessing rank derivation, tensor-parallel resume, and composable FSDP2 distributed-checkpoint resume pass.

| Architecture Decision | Accelerator, Precision, and Connector Ownership | Date: 04 September 2026 | Time: 06:37 PM | Name: Ghost |

    1. Runtime-checkable Accelerator and ClusterEnvironment contracts define type-level discovery, parsing, topology, availability, naming, and detection through `@classmethod`; concrete implementations explicitly inherit the Protocols and mark every implementation with `@override`.
    2. Trainer accelerator composition belongs to one internal `AcceleratorConnector` covering backend and device parsing, nodes, Strategy, precision, cluster environment, checkpoint I/O, and layer synchronization.
    3. The former temporary topology, plugin-configuration, and resolver objects were removed; a stateful GPU parser owns count, identifier, uniqueness, and availability invariants.
    4. Accelerators own backend discovery, binding, and device-local questions; Strategies own execution semantics; the connector validates their compatibility and injects plugins in deterministic order.
    5. Precision construction covers CPU float16 fallback, model-parallel restrictions, recursive conversion, and the NVIDIA Transformer Engine 2.18 autocast and strict delayed-scaling boundary.
    6. Trainer remains the sole public composition surface. Neither invalid overrides nor ABC inheritance remain, dtype context rollback passes under failure, CPU float64 and bfloat16 plus live MPS float16 true and mixed training pass, and unavailable CUDA, layer synchronization, DDP, model parallelism, and Transformer Engine boundaries remain explicit.

| Architecture Decision | Callback Contract, Monitoring, and Connector Ownership | Date: 04 September 2026 | Time: 06:44 PM | Name: Ghost |

    1. Callback defines the complete established setup, fit, training, validation, test, prediction, optimization, exception, checkpoint, and state hook vocabulary; all forty-three call signatures remain unchanged and concrete implementations mark every actual override.
    2. Model-provided callbacks are configured once per stage, replace compatible Trainer callbacks through the established inheritance rule, and remain ordered before checkpoint callbacks.
    3. The connector accepts a bound configuration operation rather than a concrete Trainer reference, preserving module hook context and profiler ownership without introducing a circular import or synthetic Trainer-host Protocol.
    4. Trainer and loops own hook timing, HookDispatcher owns profiler-aware invocation, and the connector owns only callback composition and configuration state; the base contract now uses Python 3.14 lazy package annotations without a future import, conditional type import, source prose, or circular dependency workaround.
    5. Early stopping, finite-value protection, learning-rate telemetry, data-axis throughput, and accelerator statistics retain separate ML meanings; obsolete imports, the unused legacy state-key property, static pseudo-utilities, and nine callback initializers terminating at `object` were removed while real overrides remain explicit.
    6. Fit, validation, test, prediction, invalid configuration, non-finite failure, optimizer telemetry, distributed sample accounting, filtered device statistics, checkpoint participation, serialization, DDP, spawn, and fork verification pass.

| Architecture Decision | Signal Connector Placement | Date: 04 September 2026 | Time: 09:15 AM | Name: Ghost |

    1. Process-signal registration, nested registration depth, launched-worker termination, original-handler chaining, serialization cleanup, and teardown now live in `trainer/connectors/signal_connector.py`.
    2. The connector depends only on the selected Strategy because worker termination is a launcher responsibility; the former full-Trainer dependency and conditional type import were removed.
    3. Signal handler state now uses a closed union covering Strategy, handlers, booleans, integers, and absence rather than a generic object dictionary.
    4. DDP spawn serialization and live SIGTERM propagation pass, the originating process exits through the expected interruption path, and no owned child remains attached after completion.

| Architecture Decision | Data Connector and Batch-Limit Ownership | Date: 04 September 2026 | Time: 03:28 PM | Name: Ghost |

    1. Dataloader source ownership, module or DataModule factory invocation, direct and nested loaders, combination modes, limits, reload cadence, overfit determinism, sampler injection, and prepared-loader state now belong to `trainer/connectors/data_connector.py`.
    2. Trainer retains stage sequencing, data preparation and setup hook order, while loops consume the connector-owned prepared loaders and batch totals rather than parallel Trainer fields.
    3. Distributed training uses padded `DistributedSampler` semantics where equal work is required, while prediction uses an unrepeated strided sampler so the union of rank partitions contains every sample exactly once without overlap.
    4. DataLoader subclass reconstruction preserves standard constructor values and current subclass-required attributes during sampler replacement, including stateful loader continuation and worker configuration.
    5. Trainer construction owns normalization of fast-development, sanity, epoch, step, overfit, and raw batch-limit inputs; the data connector applies its resolved stage limits because sized versus unsized loader semantics depend on the attached data source.
    6. The former `loops/batch_limits.py` function module and Trainer's module-level training-limit resolver were deleted without a wrapper class, factory, public alias, or duplicate implementation; the loop package now contains only class-owned behavior.
    7. Direct, module-owned, DataModule-owned, sized, unsized, nested, spawned, two-rank DDP, stateful continuation, overfit, reload, fast-development, persistent-worker, and unrepeated-prediction limits pass through the reconstructed owners.

| Architecture Decision | Metric Accumulation, Routing, and Persistence | Date: 04 September 2026 | Time: 04:33 PM | Name: Ghost |

    1. Module hooks produce metric observations, loops own reduction cadence, MetricAccumulator owns scalar validation, batch weighting, TorchMetrics windows, and distributed reduction, the logger connector owns view routing and rank policy, and concrete loggers own persistence.
    2. MetricAccumulator now consumes the batch only when loop-driven mean inference is required, retains its established direct `update(value, batch_size=...)` API, rejects inconsistent nested leading dimensions, and contains no free function, fake Protocol, generic `object`, or source prose.
    3. Logger names and versions are validated single path segments, integer versions retain standard `version_N` layout, custom loggers contribute their own `log_dir`, duplicate instances are rejected, and public logger collections are defensive copies.
    4. Shared hyperparameter normalization accepts portable deterministic values, rejects live modules and non-scalar tensors or arrays, and emits once per run across sequential and spawned stages through explicit launcher-state transfer.
    5. CSV persistence validates existing headers, reconciles worker-written schema before later stages, widens columns transactionally, restores in-memory schema on failed promotion, resumes existing rows, and supports transactional fsspec replacement.
    6. TensorBoard persistence validates local-only storage, records normalized hyperparameters and step-keyed scalar events, acquires its optional writer lazily, and releases it deterministically even when flushing fails.
    7. Weighted, unweighted, dataclass, ambiguous-batch, TorchMetrics, tensor-state, CPU, MPS, standard distributed reduction, local and remote CSV, TensorBoard, external DDP, and internal spawn paths pass; the connector alone gates durable output.

| Architecture Decision | Checkpoint Schema V2 and Trajectory Ownership | Date: 04 September 2026 | Time: 06:52 PM | Name: Ghost |

    1. The Trainer checkpoint connector remains the sole trajectory coordinator; stage `ckpt_path` and mutable `trainer.ckpt_path` are the explicit selection controls, while the unused constructor `resume_from_checkpoint` alias and connector accessor were removed.
    2. Schema version 2 records explicit `training` or `weights` artifact identity, resolved Trainer configuration, strict integer progress, typed state collections, metadata, and one canonical random-state snapshot; malformed structure fails before model or optimizer mutation where runtime construction permits.
    3. The foundational Randomness owner and immutable Pydantic snapshot now serve seeding, checkpoints, and worker transfer identically across Python, NumPy, torch CPU, and only the active CUDA or MPS backend; the duplicate launcher state and obsolete SeedManager were removed.
    4. Mid-epoch continuation persists completed-batch coordinates and epoch-start randomness independently of loader state, while automatic optimization marks the trajectory non-durable from batch entry through partial accumulation and returns to durability only after optimizer mutation and progress recording complete.
    5. A persisted effective configuration snapshot permits larger run ceilings while rejecting drift in accumulation, training and validation limits, cadence, clipping, determinism, sampler policy, barebones state, synchronization, reload cadence, and precision before restoration; bounded Timer runs record their actual unbounded epoch ceiling.
    6. Resumable saves during incomplete automatic batches or accumulated gradients fail explicitly and ModelCheckpoint defers its trigger, while weights-only output remains safe, omits trajectory state, retains the model hook, and cannot continue fit.
    7. Strict corruption, configuration mismatch, callback isolation, unsafe direct-save rejection, no step-zero callback artifact, exact local and spawned accumulated resume, stochastic replay, stateful continuation, checkpoint selection, and full or sharded model-parallel resume pass.

| Architecture Decision | Training Epoch and Optimization Boundaries | Date: 04 September 2026 | Time: 06:52 PM | Name: Ghost |

    1. `TrainingEpochLoop.run` now composes explicit epoch start, resumable data iteration, batch advancement, tail optimization, continuation persistence, and epoch-metric completion transitions rather than embedding the complete lifecycle in one 368-line method.
    2. Fetch, transfer, batch hooks, automatic or manual optimization, metric dispatch, scheduled validation, scheduler cadence, and loader continuation remain private transitions on the one epoch owner; no batch-manager, state-container, helper class, or new public API was introduced.
    3. Trainer now owns one DataModule-over-Module batch-transfer hook transaction used by training, validation, testing, and prediction, deleting three duplicated stage implementations while preserving exact hook dispatch and profiling.
    4. Strategy owns deferred backward synchronization. DDP supplies `no_sync`, while loops and `SyntheticOptimizer` request the policy without importing or inspecting the wrapper; explicit automatic and manual step operations replace the former boolean step contract.
    5. Standard and plateau scheduler updates have distinct canonical operations, `estimated_stepping_batches` is the sole horizon calculation, manual clipping no longer mutates loop configuration temporarily, and the unused duplicate scheduler-step resolver plus dead precision and scaler fields were removed.
    6. TrainingEpochLoop remains a large cohesive ML lifecycle under the concept-integrity authority; Trainer, evaluation, and prediction orchestration are already decomposed by lifecycle transition, and conditional annotation imports are absent across production.
    7. CPU and MPS execution, automatic and multi-optimizer manual optimization, two-rank deferred accumulation, manual DDP, transfer precedence across every stage, multiple-loader DDP, exact stateful continuation, plateau cadence, and sharded model-parallel exact resume pass.

| Architecture Decision | Validated Trainer Configuration Boundary | Date: 04 September 2026 | Time: 06:18 PM | Name: Ghost |

    1. The private `_TrainerConfiguration` Pydantic model represents resolved scalar run policy at Trainer construction and is the sole owner of training-limit validation, batch-limit normalization, debugging-mode rewrites, barebones conflicts, validation cadence inputs, and execution flags.
    2. The model contains no live model, optimizer, scheduler, dataloader, logger, profiler, accelerator, Strategy, callback, plugin, device, path handle, or external resource; those remain with their runtime connectors and owners.
    3. Mutable validation-duration dictionaries become owned immutable tuples inside the frozen configuration and are reconstructed as fresh dictionaries only when handed to ValidationSchedule.
    4. Trainer remains responsible for composition and side effects, while its constructor now delegates in order to accelerator materialization, resolved policy application, connector construction, loop construction, and live-state initialization; these are private lifecycle phases rather than public builders or fabricated service classes.
    5. Checkpoint restoration now separates artifact and participant restoration from application of resumed epoch, step, loader cursor, and random-state coordinates, preserving one trajectory authority while keeping each transition within the method limit.
    6. A bounded Timer with no explicit epoch or step ceiling now makes time the governing stop condition, the presentation-only Timer retains the ordinary fallback, explicit ceilings remain authoritative, and accumulation is inspectable from Trainer; default, invalid, optimizer, loader, DDP, spawn, continuation, checkpoint, and model-parallel probes pass.

| Architecture Decision | Core Module, DataModule, Metric, and Hook Surface | Date: 04 September 2026 | Time: 06:45 PM | Name: Ghost |

    1. Rahul confirmed that established machine-learning concepts and their runtime semantics control SyntheticMind architecture before generic software decomposition rules. The three hook families therefore remain explicit framework contracts rather than being folded into one oversized model implementation.
    2. `ModelHooks`, `DataHooks`, and `CheckpointHooks` preserve the recognized lifecycle vocabulary and no-op extension semantics used by Trainer and loops. Data transfer hooks require explicit loader identity, Module follows the proven cooperative initialization graph, and concrete overrides are marked only where a parent hook or PyTorch method is actually replaced.
    3. ModelHooks owns hook-level logging admissibility, Module owns metric production and request validation, and LoggedMetric lives only in its semantic metric module for independent loop and logger consumers; the historical `core.module` compatibility export was removed.
    4. DataModule and DatasetDataModule retain preparation, setup, teardown, loader construction, batch transfer, state, direct-dataset construction, and the current `load_from_checkpoint` classmethod over constructor hyperparameters plus DataModule state.
    5. Model and DataModule reconstruction share schema-v2 payload reading and hyperparameter precedence but retain distinct private loaders because only models own configure hooks, strict parameter application, and device mapping.
    6. Every production module now contains zero future annotation imports, conditional type imports, `Any`, `object`, source comments, static methods, module functions, lambdas, and unannotated assignments; all fifty-five established public Module method signatures remain unchanged.
    7. The 1,242-line module and 1,158-line Module class retain an explicit concept-integrity exception because splitting the researcher-facing PyTorch model contract through mixins or forwarding services would destroy extension semantics; every method remains within its hard limit.
    8. Package import, hook dispatch, log failures, scalar and TorchMetrics views, differentiable gather, runtime access, CPU, MPS, DDP, spawn, model and DataModule local, file-like, and remote hydration, optimization, exact continuation, and model parallelism pass.

| Architecture Decision | Plugin and Launcher Topology | Date: 04 September 2026 | Time: 11:11 AM | Name: Ghost |

    1. Numerical precision is now located under `plugins/precision/` because full, half, mixed, and Transformer Engine policies are interchangeable Trainer plugins controlling numerical execution rather than a peer training lifecycle.
    2. Checkpoint transport moved from the vague `plugins/io.py` name to `plugins/checkpoint_io.py`. The current plugin family owns only checkpoint persistence, so a generic I/O package would imply unsupported transport concepts and reproduce upstream baggage.
    3. The process launcher moved from `strategies/launcher.py` into `strategies/launchers/launcher.py`, placing start-method selection, process launch delegation, state transfer, environment reconstruction, child observation, and termination under one distributed-launch concept.
    4. Public class identities and Trainer configuration meanings remain unchanged, but obsolete import paths were removed without compatibility façades. Precision now explicitly inherits the checkpoint-hook contract and marks its checkpoint hook implementations as overrides.
    5. Full, half, mixed, and Transformer Engine precision execution, asynchronous checkpoint I/O, local optimizer execution, spawned DDP runtime state, subprocess DDP forward and consensus behavior, and sharded two-rank model parallelism pass through the new canonical paths.

| Architecture Decision | Cross-Cutting Contract, Override, and Storage Ownership | Date: 04 September 2026 | Time: 05:44 PM | Name: Ghost |

    1. Trainer owns run seeding, worker randomness, terminal rendering, deterministic algorithms, benchmark flags, and complete-run configuration; Loop owns evaluation gradient context.
    2. Core data transfer owns recursive batch placement, Module owns its one-consumer gather traversal, Strategy owns topology-sensitive initialization, and the logger connector owns metric accumulation.
    3. Path-bound StoragePath owns one resolved fsspec backend and protocol-aware normalization, joining, containment, metadata, resources, traversal, mutation, copying, and staging; twenty-three module functions and every broad fsspec return annotation were removed without changing ML-facing `str | Path` APIs.
    4. Framework exception and recursive domain type contracts live at the package root; only used FrameworkError, MisconfigurationError, CheckpointError, and SIGTERMException categories remain, with explicit constructor overrides and unchanged message prefixes.
    5. Every method replacing an explicit project or framework parent contract is marked `@override`, including real constructor, representation, and pickle hooks; Protocol-generated or `object`-only constructors remain deliberately undecorated because no declared obligation exists.
    6. Local, `file://`, in-memory remote, transactional CSV, profiler artifact, full checkpoint, remote ModelCheckpoint, schema-v2 model hydration, spawned DDP, and failure-atomic sharded model-parallel paths pass through StoragePath and their assigned artifact owners.

| Architecture Decision | Training Data Composition Ownership | Date: 04 September 2026 | Time: 05:06 PM | Name: Ghost |

    1. `CombinedLoader` moved from the transitional utility package to the canonical `syntheticmind.combined_loader` module and is exported from the package root as a first-class training-data composition concept.
    2. Root placement preserves dependency direction because both model-facing `DatasetDataModule` construction and outward Trainer stage execution consume the concept; placing it beneath Trainer connectors would force core data contracts to import orchestration.
    3. The four private mode iterators remain colocated because minimum-size, maximum-size, cycling, and sequential traversal are real algorithmic variants constructed, reset, serialized, and consumed exclusively by `CombinedLoader`.
    4. Module-level functions, static pseudo-utilities, mutable mode dispatch, generic annotations, and legacy utility imports are absent; tree-leaf classification, safe length, and limit normalization are private operations on the live CombinedLoader owner.
    5. The cohesive 588-line module and 337-line primary class exceed generic structural limits under Rahul's explicit SyntheticMind concept-integrity authority. Splitting tree structure, limits, iteration, state continuation, and worker teardown would create implementation fragments and is therefore rejected.
    6. All four combination modes, finite and unsized limits, nested construction, serialization, state capture and restoration, persistent-worker cleanup, two-rank lifecycle execution, exact stateful resume, and prediction ordering pass.

| Architecture Decision | Strategy-Owned Distributed Reduction | Date: 04 September 2026 | Time: 12:04 PM | Name: Ghost |

    1. The selected Strategy now owns the standard tensor `reduce` operation alongside barrier, broadcast, boolean consensus, gather, ranks, and process-group lifetime; `ParallelStrategy` implements mean, sum, minimum, and maximum collectives with optional process-group selection.
    2. Trainer, loops, and callbacks now gate output and side effects through `trainer.is_global_zero`, preserving selected cluster-environment rank semantics even before a process group is initialized instead of consulting ambient global state.
    3. Logged metrics and metric accumulators receive the Strategy's bound reduction operation at their computation boundary. This supplies only the required capability and avoids a circular core-to-Strategy dependency or a synthetic host Protocol.
    4. The former `DistributedUtils` namespace and its seven duplicate free-function entrypoints were deleted without a replacement utility module. Process-group teardown remains exclusively owned by `ParallelStrategy`.
    5. Local reduction validation, two-rank sum and mean, standard distributed mean, sum, minimum and maximum metrics, TorchMetrics, tensor graph state, progress rendering, checkpoint selection, callback artifacts, spawned metric views, and sharded model parallelism pass through Strategy ownership.

| Architecture Decision | Trainer Stage, Loop, and Hook Ownership | Date: 04 September 2026 | Time: 06:40 PM | Name: Ghost |

    1. Trainer remains the canonical owner of fit, validation, test, and prediction orchestration because it alone composes strategy, connectors, loops, hooks, checkpoint restoration, and teardown; the public entrypoints now delegate to bounded private transitions instead of containing monolithic workflows.
    2. Validation and test continue through one EvaluationLoop because their no-gradient execution, step-output, and metric-reduction contracts align, while prediction retains its separate output-retention and CPU-materialization lifecycle.
    3. Stage beginning now owns state reset, metric reset, data attachment, and source attachment; stage preparation owns environment, profiler, data setup, model setup, precision, and checkpoint restoration; loops own only iterative machine-learning execution.
    4. Fit optimization, managed-optimizer connection, manual validation, checkpoint continuation, sanity validation, and dynamic worker-stage invocation are explicit private Trainer transitions; the stateless one-method StageInvoker class was deleted instead of retaining a second stage authority.
    5. The private HookDispatcher now owns callback, Module, DataModule, and Strategy invocation mechanics, module hook context, profiler regions, callback-group lookup, wrapper unwrapping, and override discovery; its generic Trainer host avoids a circular import, conditional type import, quoted annotation, or synthetic host Protocol.
    6. Profiler owns active-stage truth directly, replacing Trainer's shadow profiler boolean, while stage teardown still treats model-mode restoration and every framework, loader, user-hook, logger, progress, and profiler cleanup as one failure-collecting transaction.
    7. CPU, MPS, true two-rank DDP, spawned continuation, callback configuration, multi-loader evaluation, checkpoint selection, early stopping, logger rank ownership, profiler reports and rollback, failure composition, and sharded model-parallel resume pass; the macOS multi-Trainer spawn shutdown defect remains an environment boundary.

| Architecture Decision | Model Summary Capability Ownership | Date: 04 September 2026 | Time: 04:59 PM | Name: Ghost |

    1. Model-summary lifecycle, inspection, and optional rendering now form the semantic `callbacks/model_summary/` package with separate callback, private report, and Rich adapter modules; the callback remains the canonical public lifecycle entrypoint.
    2. Layer shape collection uses a bound inspector method, parameter count, trainability, and storage bytes belong to layer inspection, and deterministic `finally` cleanup replaces destructor-dependent hook removal.
    3. Model size now sums each realized parameter's actual storage bytes instead of inferring storage from Trainer precision labels, preserving correct mixed-precision and custom-dtype accounting without a lookup table.
    4. Attached example inspection uses Trainer's DataModule-over-Module transfer transaction and shared multi-backend Randomness snapshot; detached inspection uses Module hooks, and every nested mode plus random source restores after success or failure.
    5. Plain and Rich rendering are owner-bound instance behavior, the report contains no static pseudo-utility, and all three modules contain no future import, conditional annotation, broad type, source prose, or nested function.
    6. Plain and Rich rendering, lazy parameters, hook cleanup, failure rollback, canonical DataModule transfer, CPU and MPS randomness, default Trainer integration, callback configuration, and exactly-once two-rank presentation pass.

| Architecture Decision | Profiler Runtime Ownership and Callback Subtraction | Date: 04 September 2026 | Time: 02:11 PM | Name: Ghost |

    1. `DataLeakageGuard`, `WatchdogCallback`, and callback-based `RuntimeProfiler` remain deleted because they lacked general executable semantics or duplicated the canonical profiler action lifecycle.
    2. `Profiler` is now a regular stateful framework base rather than an ABC; it owns scoped actions, stage and rank identity, report output, module attachment, causal failure notes, and deterministic teardown, while `PassThroughProfiler` remains the explicit disabled policy.
    3. `SimpleProfiler` resets wall-clock state per Trainer stage, and `AdvancedProfiler` preserves properly nested mutually exclusive cProfile regions, stage reset, raw-stat export, and launcher-safe reconstruction.
    4. PyTorch operator profiling forms one semantic subpackage whose private schedule tracks training, validation, test, and prediction independently and excludes fit sanity validation, while its private module recorder owns reversible forward-hook instrumentation.
    5. PyTorch profiler configuration rejects misspelled backend options, invalid tables, invalid schedules, ignored NVTX options, and empty flame-graph results instead of silently degrading requested behavior.
    6. XLA profiling was removed because SyntheticMind has no executable XLA accelerator or strategy; backend-specific surface without a backend lifecycle is compatibility theater rather than a training capability.
    7. CPU, local stage reuse, MPS-adjacent timing, remote fsspec output, failure cleanup, custom and stage-aware schedules, external DDP, and spawned DDP reports and traces pass; macOS fork remains bounded by the independently reproduced MPSGraph fork-safety failure.

| Architecture Decision | Resumable EMA Weight Averaging | Date: 04 September 2026 | Time: 04:54 PM | Name: Ghost |

    1. The unvalidated `EMACallback` shadow-model copy was replaced by the standard `EMAWeightAveraging` concept with strict decay, update cadence, starting step, and buffer configuration.
    2. Averaging initializes after model configuration, placement, and checkpoint restoration, advances only when global optimizer-step progress changes, copies the first eligible state exactly, averages floating tensors, and copies non-floating buffers.
    3. Average tensors, processed step, and update count persist through callback state. A full training checkpoint always substitutes captured live tensors under the sole model `state_dict`, while a weights-only artifact intentionally records the averaged in-memory model.
    4. Validation, test, and prediction swap average weights transactionally; exception and teardown restore live tensors, successful training exposes averaged weights after a real update, and a subsequent fit first restores the captured live training state.
    5. DTensor and sharded model state are rejected explicitly because correct distributed average persistence is not implemented; DDP and MPS are supported and verified.
    6. The cohesive 415-line module and 377-line callback exceed the generic class threshold under the concept-integrity authority; splitting one averaging algorithm into a one-consumer manager or store is rejected.
    7. CPU parameter and buffer behavior, accumulation cadence, failure rollback, full raw-state continuation, averaged weights artifacts, exact interrupted resume, MPS placement, two-rank DDP, and model-parallel rejection pass without future imports or static utilities.

| Architecture Decision | Pre-Mutation Finite Value Guard | Date: 04 September 2026 | Time: 12:41 PM | Name: Ghost |

    1. The batch-end `NanInfGuard` was replaced by `FiniteValueGuard`, whose name and configuration match the values it actually inspects.
    2. Loss tensors are rejected in `on_before_backward`, and optional dense or sparse gradient values are rejected in `on_after_backward`; both hooks execute before optimizer mutation through the shared precision and Strategy backward path.
    3. Automatic, manual, accumulated, and distributed optimization therefore share one preventive boundary rather than inspecting detached step output after the optimizer has already run.
    4. Configuration rejects non-boolean gradient checks, and failures report epoch, global step, and the offending parameter without introducing state or checkpoint surface.
    5. Automatic loss, automatic gradient, manual loss, parameter immutability, and spawned DDP failure-cleanup probes pass.

| Architecture Decision | Learning Rate Monitoring Semantics | Date: 04 September 2026 | Time: 12:47 PM | Name: Ghost |

    1. `LearningRateMonitor` now implements the standard forced step, forced epoch, and scheduler-controlled automatic cadence instead of storing and ignoring `logging_interval`.
    2. Step observations execute at `on_before_optimizer_step`, so gradient accumulation, manual multiple optimizers, and repeated closure evaluation follow actual optimizer use rather than batch count. Epoch observations execute at epoch start after prior scheduler updates.
    3. Every optimizer and parameter group receives a deterministic scheduler, optimizer-type, occurrence, or explicit group name; ambiguous scheduler and group names fail instead of overwriting metrics.
    4. Optional momentum, weight decay, and key prefixes are scalar-validated, dynamic parameter-group growth preserves existing history, and callback metrics remain visible on every rank while durable logger emission remains rank-zero-owned.
    5. Last step and epoch emission coordinates persist without serializing unbounded history; forced and automatic cadence, two manual optimizers, repeated types, named and dynamic groups, momentum, weight decay, invalid configuration, missing logger, exact resume, and spawned DDP persistence pass.

| Architecture Decision | Data-Parallel Throughput Accounting | Date: 04 September 2026 | Time: 12:52 PM | Name: Ghost |

    1. Strategy now exposes `data_parallel_size` separately from world size: local execution reports one, DDP reports the complete replica world, and model parallelism reports only the data-parallel mesh dimension.
    2. `ThroughputMonitor` aggregates global samples, batches, and elapsed wall time over optimizer-step windows and complete epochs, including dataloader wait while excluding validation pauses.
    3. CUDA and MPS execution synchronize before timing reads, preventing queued accelerator work from producing fictitious rates.
    4. Recursive batch-cardinality inference requires every reachable tensor to agree; ambiguous structures fail, while the standard optional `batch_size_fn` supports domain-specific batches with a validated non-negative result.
    5. Window and epoch metrics enter callback views and the rank-zero logger, while configuration requires a logger and a positive optimizer-step cadence.
    6. CPU accumulation windows, nested and custom batch cardinality, invalid configuration, MPS synchronization, DDP global sample totals, rank-zero persistence, and tensor-parallel axis semantics pass.

| Architecture Decision | Accelerator-Owned Device Statistics | Date: 04 September 2026 | Time: 12:58 PM | Name: Ghost |

    1. The Accelerator contract now includes device-stat collection because backend-specific counters and units belong with device discovery and binding rather than inside a callback switchboard.
    2. CPU reports process time, logical core count, and available load averages; CUDA reports selected allocator current and peak counters, retries, out-of-memory events, and free or total bytes; MPS reports current, driver, and recommended maximum bytes.
    3. `DeviceStatsMonitor` observes the selected accelerator across training, validation, test, and prediction, validates optional backend keys, follows optimizer-step or evaluation-batch cadence, and requires a logger.
    4. Callback metrics remain device-resident and logger persistence remains rank-zero-owned, while custom accelerators must satisfy the same explicit statistics contract.
    5. CPU and live MPS adapters, source-shaped CUDA behavior, filters, invalid configuration, train and validation stages, MPS Trainer execution, and spawned DDP persistence pass.

| Architecture Decision | Distributed Early Stopping State Machine | Date: 04 September 2026 | Time: 01:03 PM | Name: Ghost |

    1. EarlyStopping now validates monitor identity, finite non-negative delta, non-negative patience, boolean switches, mode, and finite target or divergence thresholds before a run begins.
    2. Global rank zero broadcasts the scalar monitor before every decision, so rank-zero-only metrics, missing values, patience, and stop reasons remain identical across workers.
    3. The state machine distinguishes improvement, patience exhaustion, target attainment, divergence, and non-finite monitoring; every stop decision is synchronized through Strategy and recorded with a stable reason and message.
    4. Best score, validation-check wait count, stopped epoch, and reason persist through the callback-state channel with strict scalar restoration and configuration-qualified identity.
    5. Patience counts validation checks rather than epochs, sanity and fast-development passes remain excluded, and missing metrics either fail on every rank or warn once on global rank zero.
    6. Local patience, target, divergence, non-finite, missing-monitor, invalid-state, and exact-resume paths plus spawned DDP rank-zero-only monitoring pass.

| Architecture Decision | Resumable Wall-Clock Timer | Date: 04 September 2026 | Time: 04:50 PM | Name: Ghost |

    1. Timer now has one continuous resumable training budget clock, one active epoch-training clock paused during validation, and one latest-pass clock for fit, validation, test, and prediction.
    2. Step and epoch interruption cadence remains explicit, while validation completion checks a step-level or epoch-boundary limit so long validation cannot hide an exhausted budget.
    3. Global rank zero evaluates the wall-clock boundary and broadcasts the decision, avoiding drift-dependent all-rank consensus.
    4. Elapsed training time, reached state, configured duration, and check interval persist with strict compatibility; a changed Timer policy fails on restore, a depleted checkpoint stops before another batch, and spawned workers reconcile the outcome to their parent.
    5. Duration parsing and empty clock-map construction are owner-bound Timer operations; strings enforce clock ranges, mappings enforce supported non-negative integer units, and every duration resolves to a positive finite interval.
    6. Stage-clock isolation, validation pause, step stopping, compatible and incompatible restore, invalid duration and state, progress rendering, and spawned two-rank consensus pass without future annotations or static utilities.
    7. The cohesive 489-line module and 470-line class exceed generic structural limits under the concept-first authority; fake clock-store, duration-factory, or stop-manager classes are rejected.

| Architecture Decision | Transactional Model Checkpoint Policy | Date: 04 September 2026 | Time: 05:01 PM | Name: Ghost |

    1. Model checkpoint lifecycle, immutable configuration, trigger schedule, scalar ranking, artifact paths, and save transaction now occupy one semantic callback capability package while Trainer retains trajectory assembly, Strategy retains distributed representation, and checkpoint I/O retains transport.
    2. The local responsibility census accepts each private collaborator because schedule resume, ranking state, path containment, and multi-artifact rollback are distinct state or failure boundaries; line pressure did not create forwarding wrappers, generic helpers, or additional public entrypoints.
    3. Configuration rejects invalid ranking and competing trigger modes before setup, monitored values must be scalar, generated paths cannot escape their checkpoint root, and remote storage rejects local-link semantics explicitly.
    4. Ranking, schedule, last-path, and last-artifact state mutate before serialization and roll back on failed persistence; consumed deferred-validation triggers remain consumed in the saved state, and train-end persistence advances a stale last artifact to the final global step.
    5. Rejected top-k candidates still produce a fresh standalone last artifact instead of linking to an older best checkpoint; the stateful CheckpointPaths owner now performs link publication directly, and the unused ModelCheckpoint forwarding alias was deleted.
    6. CPU ranking and failure probes, remote fsspec persistence, asynchronous snapshot isolation, callback composition, tensor metric state, DDP selection, and two-rank sharded model-parallel exact resume pass after reconstruction.
