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

| Memory | Model Runtime State Authority | Date: 19 August 2026 | Time: 11:37 PM | Name: Ghost |

    1. Epoch, optimizer-step, and rank properties are Trainer state while attached and neutral zero values while detached; cached mirrors are transport details rather than public authorities.
    2. Reading `torch.distributed` directly from a detached model leaks ambient process state into an object that has no Strategy or Trainer ownership.
    3. Fluent `freeze` and `unfreeze` returns matter because inherited user code may chain evaluation, placement, or export operations after changing parameter state.
    4. `on_gpu` is a historical CUDA-specific convenience and intentionally remains false for MPS despite both being accelerators.
    5. Direct process-group and every Trainer launcher path are required to distinguish detached neutrality from attached rank truth.

| Memory | Model-Parallel Evidence and Persistence Ownership | Date: 04 September 2026 | Time: 05:26 AM | Name: Ghost |

    1. A standalone DTensor and distributed-checkpoint probe establishes only PyTorch runtime availability; it does not establish SyntheticMind Trainer wiring, hook order, optimizer restoration, callback participation, progress continuation, or failure cleanup.
    2. Model-parallel verification must execute the complete harness path with a real `Module`, `Trainer`, process launcher, device mesh, tensor-parallel plan, optimizer step, checkpoint callback, and resumed Trainer state on every participating rank.
    3. Full and sharded checkpoint persistence form one cohesive model-parallel storage capability and therefore belong to one persistence owner rather than a collection of module-level functions.
    4. A weights-only model artifact is not resumable training state; optimizer-count validation must occur before live model mutation so a rejected resume leaves the target model unchanged.
    5. Calling `forward` directly from a default step bypasses PyTorch module hooks and compiled-call machinery even when the returned tensor is numerically correct; framework-owned forward computation must enter through the module instance.
    6. Distributed checkpoint shards completing successfully does not prove atomic replacement. Publication requires an explicit generation-commit protocol so readers observe the previous complete generation or the next complete generation, never a deleted or partial path.

| Memory | Atomic Generation Publication | Date: 04 September 2026 | Time: 05:44 AM | Name: Ghost |

    1. A distributed checkpoint directory cannot be atomically overwritten portably because replacing a non-empty directory is not a cross-platform filesystem contract; immutable generation directories plus an atomic pointer provide the required commit point.
    2. The pointer must identify both the active and previous published generations. Selecting the newest inactive directory is unsafe because a complete but unpublished failed generation can be newer than the last valid recovery generation.
    3. Generation metadata and all distributed shards must exist before pointer publication; incomplete data remains unreachable and therefore cannot masquerade as a checkpoint.
    4. The pointer is a durable schema rather than an implementation string, so it carries an explicit schema version, rejects extra fields and unknown versions, and constrains generation identifiers to canonical UUIDv7 values.
    5. Exact resumed progress is insufficient evidence for checkpoint correctness. An uninterrupted trajectory and a resumed trajectory must produce exactly matching model state under identical initialization, data ordering, and optimizer steps.
    6. Cross-format overwrite should fail rather than destroy an existing checkpoint because a portable atomic replacement cannot switch safely between a regular file and a non-empty directory.

| Memory | Strategy Contracts and Process Lifetime | Date: 04 September 2026 | Time: 07:05 AM | Name: Ghost |

    1. Replacing an ABC with a Protocol is correct only when shared state and default lifecycle behavior are removed from the contract; otherwise the syntax changes while the architectural coupling remains.
    2. DDP, tensor parallelism, and composable fully sharded execution share process coordination but not model-wrapping identity. Shared coordination belongs to a composed process resource, while each concrete strategy retains its own placement and checkpoint semantics.
    3. A process group used by several sequential Trainer stages or retained DTensors cannot be destroyed during stage teardown. The correct release boundary is exception recovery or process exit unless a separately owned external group proves a different lifecycle.
    4. `object` in an owned contract erases which values can cross the boundary. Named recursive aliases for steps, optimizer closures, state dictionaries, stage arguments, and storage options expose the actual supported domain and make invalid widening visible during review.
    5. Checkpoint artifact reading and live-state restoration are separate operations. Metadata may be loaded before model construction, while restoration must occur only after the model and optimizers exist; a bounded preload cache preserves this ordering without duplicate disk reads.
    6. Asynchronous checkpointing requires a main-thread snapshot before worker submission. The snapshot and background save belong to the checkpoint plugin itself and do not justify nested functions, module functions, or a callable wrapper object.

| Memory | Device and Precision Boundary Design | Date: 04 September 2026 | Time: 07:21 AM | Name: Ghost |

    1. Static backend queries do not require an abstract base carrying placeholder methods; a structural accelerator contract preserves custom backend substitution while concrete accelerators retain device-specific binding.
    2. Integer GPU configuration means a device count, while a string such as `"2"` means the explicit identifier two and `"-1"` means every available device. Conflating numeric strings with counts silently changes placement semantics.
    3. Frozen resolved configuration requires immutable nested values as well as a frozen outer model; tuple device identifiers prevent mutation that a frozen model containing a list would still permit.
    4. A recursive tensor-conversion algorithm used only by precision policies belongs on the concrete precision lifecycle. A stateless converter class adds an object and dependency edge without adding an independent boundary.
    5. Optional-library integration should isolate dynamic imports and attribute adaptation in one adapter. The owning precision policy should expose stable domain configuration rather than leaking arbitrary mappings or module-shaped Protocol collections.
    6. Current Transformer Engine uses a general `autocast` context with a `recipe` argument, and current FP8 Linear guidance requires both matrix dimensions to be divisible by sixteen. Version checks and adapter tests must encode that current boundary explicitly.

| Memory | Concept Integrity Versus Mechanical Decomposition | Date: 04 September 2026 | Time: 09:00 AM | Name: Ghost |

    1. Removing an ABC does not justify replacing a behavior-rich framework hierarchy with a Protocol. When concrete implementations repeat the same lifecycle, the repeated behavior proves that a stateful base concept exists.
    2. Strategy is the integration boundary for model placement, precision, optimizer execution, distributed communication, and checkpoint representation; extracting each mechanism as a peer module obscures the selected training policy rather than clarifying it.
    3. A Trainer connector is earned when it converts one user-facing configuration family into a connected runtime slice with ordering and compatibility invariants. A stateless resolver or temporary runtime-data model does not become a connector merely by being renamed.
    4. The strongest topology test is whether an ML researcher can identify the concept, its lifecycle, and its algorithmic variants without referring to code size. DDP, FSDP, precision, dataloaders, and checkpoints pass; generic runtime, resolver, forwarding, and state-container names require stronger internal evidence.
    5. Package width is harmless when each file is a selectable recognized implementation, as with callbacks or strategies. The same width is architectural debt when files are fragments created to satisfy numeric budgets.
    6. Concept-first decomposition still requires directed dependencies, bounded public surface, explicit state ownership, and runtime evidence; it overrides mechanical splitting, not correctness or maintainability review.

| Memory | Accelerator Connector Admission Evidence | Date: 04 September 2026 | Time: 09:07 AM | Name: Ghost |

    1. Accelerator, device count, node count, Strategy, precision, cluster environment, checkpoint I/O, and synchronized-layer policy form one compatibility matrix during Trainer construction; resolving them independently permits contradictory intermediate states.
    2. A resolved device list containing live `torch.device` objects is runtime composition state, not a portable Pydantic data model and not an independent topology concept.
    3. Precision selection cannot be owned by a standalone resolver because validity and fallback depend on the chosen accelerator, Strategy, and explicitly supplied plugin.
    4. Strategy instances that already configure accelerators or devices must conflict with explicit Trainer values rather than silently choosing one source, while default `auto` values permit Strategy-owned configuration.
    5. A connector earns its class and module through cross-component ordering and compatibility invariants; deleting several stateless resolvers is evidence that the connector consolidates a real boundary rather than renaming utility code.

| Memory | Callback Composition Without Trainer Coupling | Date: 04 September 2026 | Time: 09:13 AM | Name: Ghost |

    1. Callback configuration and callback execution are separate concepts: the connector owns selection, defaults, replacement, ordering, and identity, while Trainer owns when lifecycle hooks execute.
    2. Passing a bound hook operation into the connector preserves profiling and module hook context without letting the connector reach into the entire Trainer or requiring a circular annotation dependency.
    3. Model callback replacement must remove callbacks by compatible parent type, not merely exact class identity, while preserving unrelated Trainer callbacks and placing checkpoint callbacks last.
    4. Connector state must clear after configuration failure and after every stage so a rejected callback declaration cannot poison later validation, testing, or prediction.

| Memory | Data Composition Is One Training Concept | Date: 04 September 2026 | Time: 09:26 AM | Name: Ghost |

    1. A training data boundary includes source selection, factory invocation, nested loader combination, finite or unknown length, limits, sampler policy, reload state, and worker-preserving reconstruction; separating these by implementation technique creates contradictory owners.
    2. Direct loaders, model-owned factories, and DataModule-owned factories are alternative sources for the same stage input and must converge before combination, limits, sampling, and loop consumption.
    3. Distributed prediction cannot use padded training partitions because duplicated samples corrupt output identity; an unrepeated rank-strided sampler is a distinct evaluation contract.
    4. Sampler replacement must reconstruct DataLoader subclasses with their required constructor attributes. Replacing only the sampler on a generic base DataLoader can silently discard stateful-loader behavior and checkpoint continuation.
    5. Multiprocessing result transfer must serialize connector-owned loader totals rather than reaching for obsolete Trainer fields; worker transfer failures expose hidden ownership remnants after an otherwise successful local refactor.
    6. A cohesive connector may exceed a generic size threshold when every retained method participates in one lifecycle and extraction would produce one-consumer implementation fragments. Runtime breadth and conceptual unity control the exception.

| Memory | Metric Production, Routing, and Persistence | Date: 04 September 2026 | Time: 09:33 AM | Name: Ghost |

    1. Metric production belongs to Module hooks, reduction cadence belongs to loops, view routing belongs to the logger connector, and durable output belongs to concrete loggers; merging these phases creates hidden state and inconsistent consumers.
    2. Callback metrics may include monitoring values that are intentionally absent from external logs, while progress metrics are scalar presentation state and logged metrics retain tensors for programmatic inspection.
    3. Sanity validation must clear every observable metric view after execution so early diagnostic batches cannot alter checkpoint ranking, early stopping, progress output, or experiment records.
    4. Logger selection, default version discovery, and hyperparameter emission share destination identity and therefore belong to the connector rather than a free configuration module.
    5. Multiprocessing launch must transfer all three metric views and the last displayed stage as one observation snapshot; transferring only public logged metrics loses callback and progress semantics.

| Memory | Checkpoint Trajectory Ownership | Date: 04 September 2026 | Time: 09:51 AM | Name: Ghost |

    1. A resumable checkpoint is a trajectory aggregate, not a model-weight file: optimizer, scheduler, callbacks, DataModule, precision, loop progress, loader cursor, random streams, epoch, and global step must share one coordinator.
    2. Strategy owns the representation of model and optimizer state because sharding depends on execution topology; the Trainer connector owns when that representation joins the remaining trajectory and in which order it restores.
    3. `CheckpointIO` owns transport and atomic persistence but must not interpret training lifecycle state. `ModelCheckpoint` owns cadence and retention but must not construct a competing payload.
    4. A stateless class named `CheckpointState` obscures behavior and lifecycle. Path preload caches, user-managed selection, continuation coordinates, and schema validation prove that checkpoint coordination is a live connector responsibility.
    5. Private checkpoints require one explicit schema version and one canonical key per semantic field. Emitting singular and plural aliases increases ambiguity and restoration branches without serving current SyntheticMind consumers.
    6. Full, sharded, asynchronous, weights-only, mid-epoch, and failure-atomic paths must be verified separately because success in one persistence representation does not prove another.

| Memory | Optimizer Versus Optimization Semantics | Date: 04 September 2026 | Time: 10:09 AM | Name: Ghost |

    1. An optimizer is a model-facing PyTorch object and declaration contract; optimization is the loop lifecycle that invokes forward, loss extraction, zeroing, backward, accumulation, clipping, stepping, and scheduler cadence.
    2. Automatic and manual optimization are recognized execution modes and justify a loop subpackage. A `core/optimization/` package mixing parser, wrapper, and execution behavior erases that distinction.
    3. A managed optimizer should adapt hook-aware stepping and parameter toggling through explicit connected operations. Dynamic subclass fabrication and generic attribute forwarding make runtime identity dishonest and hide the supported API.
    4. PyTorch optimizer closures remain ordinary callables, but SyntheticMind can supply a bound `run` method instead of defining a custom callable object or nested function.
    5. Strategy and precision jointly own the actual optimizer step because wrapping, sharding, scaling, unscaling, and overflow behavior depend on execution topology and numerical policy.
    6. Scheduler cadence belongs to the training epoch lifecycle, while `Module.lr_scheduler_step` remains the explicit extension hook for schedulers outside the standard PyTorch API.

| Memory | Canonical Import Paths Express Ownership | Date: 04 September 2026 | Time: 10:16 AM | Name: Ghost |

    1. Moving a class without removing its former import path preserves ambiguous ownership. A ground-up private framework should update consumers directly and leave exactly one canonical implementation path.
    2. Trainer state is not a general state-domain package; its status, stage, counters, interruption, and stop flag change with Trainer lifecycle and therefore belong beside Trainer.
    3. Connector extraction should precede canonical Trainer relocation when it materially shrinks the orchestration module and establishes the package's conceptual contents before imports move.
    4. Subprocess and multiprocessing launchers are mandatory relocation tests because serialized module paths and conditional annotations can remain stale after local imports appear successful.

| Memory | Hook Contracts Are Machine-Learning Concepts | Date: 04 September 2026 | Time: 11:04 AM | Name: Ghost |

    1. Hook families are not arbitrary mixins created to shorten a file. They are the stable extension vocabulary through which research models and data modules participate in deterministic Trainer and loop boundaries.
    2. Lightning's `ModelHooks`, `DataHooks`, and `CheckpointHooks` separation is conceptually sound even though unrelated Lightning implementation details remain unsuitable for SyntheticMind. Removing these families would erase the distinction between model lifecycle, data lifecycle, and checkpoint participation.
    3. Cooperative `super()` inside `DataHooks.__init__` has meaning only inside the complete multiple-inheritance graph. It must not be decorated as an override merely because `object.__init__` exists; the concrete MRO must demonstrate that the call initializes the next real framework participant.
    4. Hook dispatch belongs to Trainer and loops, hook implementations belong to modules, data modules, and callbacks, and hook-name logging policy belongs with the hook vocabulary. A separate rule-model family adds no independent state, boundary, or lifecycle.
    5. Serialized checkpoint field strings are durable schema values, while Python identifiers naming those fields are implementation details. Renaming the identifiers to lower snake case removes style debt without migrating the checkpoint format.

| Memory | Plugin Categories and Process Launch Placement | Date: 04 September 2026 | Time: 11:11 AM | Name: Ghost |

    1. Precision is an interchangeable numerical-execution plugin even though its behavior reaches deeply into module initialization, forward contexts, backward execution, optimizer stepping, gradient scaling, and checkpoint state.
    2. A package name must describe the concepts actually implemented rather than copy the breadth of an upstream framework. `checkpoint_io.py` is correct while checkpoint transport is the only I/O plugin family; `io/` becomes justified only if distinct plugin I/O concepts actually appear.
    3. Launcher policy and launcher mechanisms form one distributed-systems concept. Keeping the policy module beside Strategy while its subprocess and multiprocessing implementations live in `launchers/` makes ownership harder to infer and creates an artificial package boundary.
    4. Relocation verification must include importable spawned workers, subprocess workers, Strategy serialization, optional precision adapters, and checkpoint wrappers because local construction cannot expose stale serialized module paths.

| Memory | Utility Modules Conceal Framework Ownership | Date: 04 September 2026 | Time: 11:14 AM | Name: Ghost |

    1. Randomness configuration changes with Trainer run construction, console rendering changes with Trainer lifecycle observation, metric accumulation changes with logger reduction semantics, and model materialization changes with Strategy topology. None is a generic utility concern.
    2. A one-consumer helper module should be deleted when it duplicates a native framework operation, or moved beside its natural owner when it carries reusable state or a complete mechanism used by that owner.
    3. Import relocation is incomplete until spawned workers and distributed strategies deserialize the moved class successfully. Compilation and local unit behavior alone cannot prove a process-boundary move.

| Memory | Tensor Traversal Belongs With Its Operation | Date: 04 September 2026 | Time: 11:18 AM | Name: Ghost |

    1. Generic recursive traversal is not automatically an independent concept. When one operation is the sole consumer, the traversal remains a private method on that operation's natural owner.
    2. Batch device placement is shared by the data-hook contract and prediction collection, so it earns a precise core data-transfer module. Collective input conversion is used only by `Module.all_gather` and therefore does not earn a parallel module.
    3. Nested collection preservation must be verified across local and distributed execution because named tuples, process groups, scalar tensorization, and worker serialization can fail independently.

| Memory | Execution Context Is Loop State | Date: 04 September 2026 | Time: 11:19 AM | Name: Ghost |

    1. Inference-mode selection is not a generic context-manager utility when its choice depends on evaluation lifecycle and distributed backend constraints. The shared Loop owner is the narrowest concept serving validation, testing, and prediction.
    2. Process-wide PyTorch determinism and benchmark flags are run configuration side effects. Trainer construction must validate and apply them before model execution so there is one visible owner and no alternate function entrypoint.

| Memory | Checkpoint Transport Owns Serialization | Date: 04 September 2026 | Time: 11:22 AM | Name: Ghost |

    1. Atomic file replacement, remote transaction boundaries, map location, weights-only loading, and payload validation are checkpoint transport behavior and belong on `TorchCheckpointIO` rather than free persistence functions.
    2. Model-parallel metadata may reuse the ordinary checkpoint transport while distributed tensor shards retain their specialized store. Composition preserves the difference between metadata serialization and distributed state publication without duplicating either lifecycle.
    3. Failure injection must prove that a metadata write or active-generation publication failure cannot replace the preceding durable sharded generation.

| Memory | Filesystem Is Infrastructure, Not Utility | Date: 04 September 2026 | Time: 11:23 AM | Name: Ghost |

    1. Protocol-aware paths, binary and text resources, artifact staging, and remote transactions constitute one external storage boundary even when many framework concepts consume it.
    2. A shared boundary belongs in a precise top-level package when no single checkpoint, logger, profiler, or Trainer consumer owns its complete lifecycle.

| Memory | Dependency-Neutral Framework Contracts | Date: 04 September 2026 | Time: 11:41 AM | Name: Ghost |

    1. Shared type aliases and framework exception classes are contracts, not utilities: they constrain values and failure categories crossing subsystem boundaries without executing a lifecycle of their own.
    2. Placing these contracts at the package root prevents core, Trainer, loops, strategies, plugins, callbacks, loggers, and profilers from depending on a generic utility layer or on one another merely to name shared domains.
    3. A contract relocation requires multiprocessing and distributed execution evidence because imported module paths participate in worker reconstruction even when local imports compile.
    4. Historical capability notes that describe code deliberately removed from the current scope should be pruned from active ledgers; Git history remains the chronology while ledgers retain current durable knowledge.

| Memory | Combined Loader Is a Data Composition Primitive | Date: 04 September 2026 | Time: 11:53 AM | Name: Ghost |

    1. Nested loader tree shape, sampling mode, length and limit semantics, iteration state, stateful-loader continuation, and worker teardown form one training-data concept rather than unrelated utility mechanisms.
    2. The mode iterator classes pay architectural rent because each implements a distinct recognized traversal algorithm and exists only as a private strategy selected by one public `CombinedLoader` owner.
    3. A core DataModule and outward Trainer loops both consume loader composition, so an inward package-root module preserves acyclic dependency direction better than a Trainer connector subpackage or a one-file data directory.
    4. Worker reconstruction, pickling, direct distributed execution, and exact stateful resume are mandatory relocation evidence because ordinary local iteration cannot expose module-path or cursor-ownership regressions.
    5. Explicit project authority may retain one oversized cohesive concept when every proposed split creates implementation fragments; the exception must name the inseparable responsibilities and remain subject to subtraction review.

| Memory | Distributed State Must Follow the Selected Strategy | Date: 04 September 2026 | Time: 12:04 PM | Name: Ghost |

    1. Ambient `torch.distributed` rank queries are weaker than Strategy state because a launcher or cluster environment can know the correct rank before process-group initialization and a detached object has no execution-policy authority.
    2. Metric reduction is an execution-policy operation: backend, process group, topology, and local identity behavior belong to Strategy, while metric owners retain reduction intent and accumulated values.
    3. Passing one bound reduction method into inward metric code avoids both a circular dependency on concrete Strategy and an oversized host contract containing unrelated Trainer behavior.
    4. Rank-zero presentation and persistence gates should read the Trainer's selected Strategy, while collective participation must remain symmetric across ranks before any rank-specific return.
    5. A utility namespace that duplicates Strategy ranks and collectives creates two authorities and free-function aliases; deletion is safer than relocation once every real consumer already carries Trainer or Strategy.
