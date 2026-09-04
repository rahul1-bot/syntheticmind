| Memory | Public Machine-Learning Compatibility | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. Public compatibility is semantic rather than cosmetic. Method names, parameter names and order, defaults, accepted values, return meanings, hook timing, state counters, and failure behavior jointly define the contract users inherit.
    2. PyTorch Lightning is a mature source of capability evidence, but its branded types and internal decomposition are not industry standards. SyntheticMind adopts universal concepts and owns its implementation language.
    3. An inherited extension point is incomplete until cooperative initialization, method-resolution order, override discovery, launcher transport, exception behavior, and every runtime dispatch site agree with the public signature.
    4. Surface comparison cannot establish capability. A parameter that is accepted but ignored, a hook that is defined but never dispatched, or a property backed by stale state is a defect rather than partial parity.
    5. Divergence from an established concept requires genuinely different semantics and explicit runtime enforcement; renaming a familiar concept only creates translation debt.

| Memory | Lifecycle, Presentation, and Failure Ownership | Date: 04 September 2026 | Time: 04:44 PM | Name: Ghost |

    1. Stage state, module modes, hook context, rank-zero progress state, and active exceptions require exception-safe restoration on success, early exit, and every failure path.
    2. Cleanup is a transaction across Strategy, datamodule, callbacks, module, logger, profiler, and progress presentation. The first causal failure remains primary while later cleanup failures become attached evidence.
    3. A completion record may be rendered only after failure-prone end hooks and teardown have succeeded; stage presentation is one hook-rich state machine, while terminal fitting, throttling, writing, and failure tolerance belong to its renderer.
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

| Memory | Strategy Contracts, Initialization, and Process Lifetime | Date: 04 September 2026 | Time: 04:36 PM | Name: Ghost |

    1. Replacing an ABC with a Protocol is correct only when shared state and default lifecycle behavior are removed from the contract; otherwise the syntax changes while the architectural coupling remains.
    2. DDP, tensor parallelism, and composable fully sharded execution share process coordination but not model-wrapping identity. Shared coordination belongs to a composed process resource, while each concrete strategy retains its own placement and checkpoint semantics.
    3. A process group used by several sequential Trainer stages or retained DTensors cannot be destroyed during stage teardown. The correct release boundary is exception recovery or process exit unless a separately owned external group proves a different lifecycle.
    4. Even PyTorch's dynamic `__torch_function__` boundary can use a named recursive value schema; broad `object` merely conceals tensor, device, dtype, layout, generator, shape, scalar, and nested-container semantics that model initialization actually accepts.
    5. Checkpoint artifact reading and live-state restoration are separate operations. Metadata may be loaded before model construction, while restoration must occur only after the model and optimizers exist; a bounded preload cache preserves this ordering without duplicate disk reads.
    6. Asynchronous checkpointing requires a main-thread snapshot before worker submission. The snapshot and background save belong to the checkpoint plugin itself and do not justify nested functions, module functions, or a callable wrapper object.

| Memory | Accelerator and Precision Boundary Design | Date: 04 September 2026 | Time: 09:07 AM | Name: Ghost |

    1. Static backend queries do not require an abstract implementation base; a structural accelerator contract preserves substitution while concrete accelerators own discovery, parsing, binding, and backend-local observations.
    2. Integer GPU configuration is a device count, numeric strings are explicit identifiers, and `-1` selects every device; one stateful parser must preserve these distinct meanings.
    3. Accelerator, Strategy, precision, environment, checkpoint storage, and layer synchronization form one compatibility matrix and therefore require one Trainer connector rather than independent temporary resolver objects.
    4. Recursive conversion belongs to the concrete precision lifecycle, while optional libraries belong in isolated adapters with stable domain configuration.
    5. Transformer Engine uses the current autocast recipe API and sixteen-by-sixteen FP8 Linear constraint; version and adapter evidence must encode that boundary.

| Memory | Concept Integrity Versus Mechanical Decomposition | Date: 04 September 2026 | Time: 09:00 AM | Name: Ghost |

    1. Removing an ABC does not justify replacing a behavior-rich framework hierarchy with a Protocol. When concrete implementations repeat the same lifecycle, the repeated behavior proves that a stateful base concept exists.
    2. Strategy is the integration boundary for model placement, precision, optimizer execution, distributed communication, and checkpoint representation; extracting each mechanism as a peer module obscures the selected training policy rather than clarifying it.
    3. A Trainer connector is earned when it converts one user-facing configuration family into a connected runtime slice with ordering and compatibility invariants. A stateless resolver or temporary runtime-data model does not become a connector merely by being renamed.
    4. The strongest topology test is whether an ML researcher can identify the concept, its lifecycle, and its algorithmic variants without referring to code size. DDP, FSDP, precision, dataloaders, and checkpoints pass; generic runtime, resolver, forwarding, and state-container names require stronger internal evidence.
    5. Package width is harmless when each file is a selectable recognized implementation, as with callbacks or strategies. The same width is architectural debt when files are fragments created to satisfy numeric budgets.
    6. Concept-first decomposition still requires directed dependencies, bounded public surface, explicit state ownership, and runtime evidence; it overrides mechanical splitting, not correctness or maintainability review.

| Memory | Callback Contracts and Monitoring Ownership | Date: 04 September 2026 | Time: 04:47 PM | Name: Ghost |

    1. Callback vocabulary, callback configuration, hook timing, and hook invocation are separate concepts: Callback defines signatures, the connector owns selection and identity, loops and Trainer choose transitions, and HookDispatcher owns profiled calls.
    2. Python 3.14 lazy annotations can reference `syntheticmind.Trainer` and `syntheticmind.Module` through the package during circular construction without a future import, `TYPE_CHECKING`, quoted annotation, or weakened host Protocol.
    3. Monitoring callbacks remain separate only when each names a real ML observation or control concept; removing static helpers means their validation, scalar conversion, synchronization, and emission operations stay on the callback instance that owns the policy.
    4. Early-stop decisions require rank consensus, finite checks precede mutation, learning rates follow optimizer use, throughput follows the data-parallel sample axis, and device statistics come from the selected Accelerator.

| Memory | Data Composition and Validated Run Policy | Date: 04 September 2026 | Time: 03:53 PM | Name: Ghost |

    1. A training data boundary includes source selection, factory invocation, nested loader combination, finite or unknown length, limits, sampler policy, reload state, and worker-preserving reconstruction; separating these by implementation technique creates contradictory owners.
    2. Direct loaders, model-owned factories, and DataModule-owned factories are alternative sources for the same stage input and must converge before combination, limits, sampling, and loop consumption.
    3. Distributed prediction cannot use padded training partitions because duplicated samples corrupt output identity; an unrepeated rank-strided sampler is a distinct evaluation contract.
    4. Sampler replacement must reconstruct DataLoader subclasses with their required constructor attributes. Replacing only the sampler on a generic base DataLoader can silently discard stateful-loader behavior and checkpoint continuation.
    5. Scalar run-policy validation and resolution belong to one immutable Trainer configuration boundary, while applying a resolved stage limit belongs to DataConnector because only that owner knows whether the loader is sized, unsized, nested, or distributed.
    6. Configuration must not absorb live runtime objects merely to shorten a constructor; accelerators, strategies, loggers, profilers, callbacks, dataloaders, and optimizers retain behavioral owners and are composed after scalar policy resolution.

| Memory | Metric Production, Weighting, Routing, and Persistence | Date: 04 September 2026 | Time: 04:33 PM | Name: Ghost |

    1. Metric production belongs to Module hooks, reduction cadence belongs to loops, scalar and batch-weight reduction belongs to MetricAccumulator, view routing belongs to the logger connector, and durable output belongs to concrete loggers.
    2. Batch-size inference is part of mean-reduction semantics rather than a loop utility; nested tensors with contradictory leading dimensions must fail and request explicit `batch_size` instead of silently selecting the first candidate.
    3. Sanity validation must clear every observable metric view after execution so early diagnostic batches cannot alter checkpoint ranking, early stopping, progress output, or experiment records.
    4. Global-rank persistence belongs in the connector rather than every callback and loop call site; callbacks may retain rank-local observations while one central gate prevents duplicate durable writes.
    5. Multiprocessing launch must transfer metric views, presentation state, and one-time hyperparameter progress; a later worker otherwise repeats run metadata or widens a CSV file from stale parent schema.
    6. CSV schema widening is a commit operation: adopt the existing header, serialize all prior rows under the union, atomically replace the file, and roll back live field state if promotion fails.
    7. TorchMetrics objects retain their own update and reset lifecycle, while scalar observations retain differentiable tensors only when explicitly requested; neither path may hide its value behind `object` or an unearned Protocol.

| Memory | Checkpoint Trajectory, Configuration, and Randomness | Date: 04 September 2026 | Time: 04:26 PM | Name: Ghost |

    1. A resumable checkpoint is a trajectory aggregate, not a model-weight file: optimizer, scheduler, callbacks, DataModule, precision, resolved configuration, loop progress, loader cursor, random streams, epoch, and global step must share one coordinator.
    2. Strategy owns the representation of model and optimizer state because sharding depends on execution topology; the Trainer connector owns when that representation joins the remaining trajectory and in which order it restores.
    3. `CheckpointIO` owns transport, ModelCheckpoint owns cadence and retention, and CheckpointPaths owns formatting, containment, versions, removal safety, and link publication directly; forwarding aliases create duplicate checkpoint verbs without a boundary.
    4. Loader state and training progress are independent: an ordinary stateless loader still requires completed-batch coordinates, epoch-start randomness for deterministic replay, and checkpoint-time randomness before the next model step.
    5. Random state must use one immutable validated schema across checkpoint and process boundaries and capture only active accelerators; probing MPS during a CPU fork can initialize unsafe native runtime state.
    6. Resolved run configuration is provenance and a compatibility boundary; a higher step or epoch ceiling may continue the same run, while trajectory-defining accumulation, data, validation, precision, and synchronization settings must not drift silently.
    7. A weights-only artifact must omit every resumable-state fragment and identify itself explicitly, while full, sharded, remote, asynchronous, mid-epoch, linked, distributed, and failure-atomic paths require separate evidence.

| Memory | Optimizer, Epoch, and Topology Semantics | Date: 04 September 2026 | Time: 02:58 PM | Name: Ghost |

    1. An optimizer is a model-facing PyTorch object and declaration contract; optimization is the loop lifecycle that invokes forward, loss extraction, zeroing, backward, accumulation, clipping, stepping, and scheduler cadence.
    2. A training epoch is not one indivisible method: resume, fetch, transfer, batch hooks, optimization, metrics, validation, scheduler, and continuation are named transitions that remain cohesive under one epoch owner.
    3. Shared batch-transfer precedence belongs beside Trainer hook dispatch; repeating it inside each stage loop creates several authorities that can drift independently.
    4. Gradient synchronization belongs to Strategy because the selected wrapper and topology determine how accumulation or manual toggling suppresses communication.
    5. Managed optimizers require distinct automatic and manual step operations; passing a boolean mode through an otherwise identical callable hides progress semantics and invites double step counting.
    6. Standard scheduler cadence and monitor-dependent plateau cadence are separate operations, while `Module.lr_scheduler_step` remains the explicit extension hook for scheduler-specific behavior.

| Memory | Hook Contracts Are Machine-Learning Concepts | Date: 04 September 2026 | Time: 11:04 AM | Name: Ghost |

    1. Hook families are not arbitrary mixins created to shorten a file. They are the stable extension vocabulary through which research models and data modules participate in deterministic Trainer and loop boundaries.
    2. Lightning's `ModelHooks`, `DataHooks`, and `CheckpointHooks` separation is conceptually sound even though unrelated Lightning implementation details remain unsuitable for SyntheticMind. Removing these families would erase the distinction between model lifecycle, data lifecycle, and checkpoint participation.
    3. Cooperative `super()` inside `DataHooks.__init__` has meaning only inside the complete multiple-inheritance graph. It must not be decorated as an override merely because `object.__init__` exists; the concrete MRO must demonstrate that the call initializes the next real framework participant.
    4. Hook dispatch belongs to Trainer and loops, hook implementations belong to modules, data modules, and callbacks, and hook-name logging policy belongs with the hook vocabulary. A separate rule-model family adds no independent state, boundary, or lifecycle.
    5. Serialized checkpoint field strings are durable schema values, while Python identifiers naming those fields are implementation details. Renaming the identifiers to lower snake case removes style debt without migrating the checkpoint format.

| Memory | Utility Evacuation Ownership Test | Date: 04 September 2026 | Time: 11:23 AM | Name: Ghost |

    1. A generic helper must move to the narrowest existing owner whose lifecycle, state, or external boundary gives the behavior meaning; one-consumer traversals remain private methods and dead native-operation façades are deleted.
    2. Run configuration belongs to Trainer, loop execution contexts belong to Loop, topology-sensitive materialization belongs to Strategy, metric routing belongs to the logger connector, and recursive batch placement belongs to the data-transfer hook boundary.
    3. Checkpoint serialization and atomic transport belong to `CheckpointIO`, distributed shards belong to model-parallel checkpointing, and protocol-aware paths shared by several artifact owners justify the explicit storage infrastructure package.
    4. Compilation and local behavior cannot prove relocation across a process boundary; spawned workers, distributed strategies, persistence reload, and failure recovery are selected according to the moved concept.

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

| Memory | Hook Dispatch and Stage Transactions | Date: 04 September 2026 | Time: 04:01 PM | Name: Ghost |

    1. Recursive model mode and step-signature knowledge are live Module concerns, Trainer owns the outer transaction, and one HookDispatcher owns the mechanics of invoking and profiling callback, Module, DataModule, and Strategy hooks.
    2. A loop owns iterative lifecycle order rather than environment construction or resource cleanup; validation and test share EvaluationLoop only because their inputs, no-gradient steps, metric reduction, and result contracts align.
    3. Public stage methods should resolve user inputs and launcher dispatch, then enter one local implementation whose private transitions name preparation, optimization, restoration, execution, and finalization without inventing a second stage authority.
    4. Failure cleanup is part of stage semantics: mode restoration and every teardown action must continue after a secondary failure, while the original execution exception remains primary and accumulates causal notes.
    5. Dispatch can store its Trainer host through a generic class parameter and depend directly on existing callback and profiler owners; this prevents a circular Trainer import without inventing a Protocol, quoted annotation, conditional import, or forwarding method family.

| Memory | Model Summary Is Inspection Plus Presentation | Date: 04 September 2026 | Time: 04:59 PM | Name: Ghost |

    1. Model-summary callback timing, layer and model inspection, and optional Rich presentation are related but independently changing responsibilities that justify one semantic capability package rather than a utility module or one oversized callback file.
    2. Forward-hook ownership must be explicit and deterministic: a bound method records the first shape observation, layer inspection owns parameter accounting, and report-level `finally` cleanup removes every surviving hook.
    3. Parameter storage size should derive from realized parameter `numel()` and `element_size()` rather than a precision-name table because mixed precision and custom dtypes decouple execution policy from stored parameter dtype.
    4. Example-input inspection is observational: attached models must reuse Trainer's DataModule transfer precedence and shared active-backend Randomness snapshot, while detached models use their own Module hooks.
    5. Optional rendering dependencies belong in their adapter module and must not become mandatory imports for the base callback or Trainer construction.

| Memory | Profiler Ownership Requires Executable Measurement | Date: 04 September 2026 | Time: 02:11 PM | Name: Ghost |

    1. A callback name is not evidence of capability: a watchdog that reports only after recovery does not watch a live stall, and a split guard driven by one optional batch key does not validate dataset integrity.
    2. Profiling has one execution owner. A callback that duplicates action timing, device telemetry, logger emission, and durable artifacts creates a second observability stack and forces unrelated loop state to exist solely for that duplicate.
    3. A profiler base owns shared state and lifecycle behavior and therefore remains a regular class; replacing its ABC with a Protocol would discard inherited implementation while replacing it with another ABC would preserve the original coupling under different syntax.
    4. Simple duration aggregation, Python call attribution, and PyTorch operator tracing are different measurement algorithms; module forward ranges and stage-aware schedules are real private PyTorch-profiler subcomponents because they own independent resources and state transitions.
    5. A supplied schedule must not execute during configuration validation, fit sanity steps must not consume the training trace window, and unsupported schedule outputs must fail where the profiler advances the real step.
    6. Requested trace artifacts must contain their promised evidence. When an installed backend captures no stack frames, explicit flame-graph failure is more truthful than a successful zero-byte file.

| Memory | Weight Averaging Is Optimizer-Progress State | Date: 04 September 2026 | Time: 04:54 PM | Name: Ghost |

    1. EMA updates must follow actual optimizer-step progress rather than batch count because gradient accumulation and manual optimization decouple batches from parameter changes.
    2. Averaging a deep-copied model before `configure_model`, device placement, or sharding creates stale structure and cross-device operations; named live tensors must be captured only after the real model exists.
    3. Callback average state belongs in the callback channel, while a full resumable checkpoint must pair optimizer state with live training tensors even when the post-fit module or an evaluation stage currently exposes averaged tensors.
    4. Floating and complex tensors admit exponential averaging, while integer and boolean buffers must copy the latest value. Shape and key drift are hard failures rather than silent zip truncation.
    5. Evaluation swaps require exception and teardown rollback; full checkpoints substitute live tensors, weights-only artifacts intentionally retain averages, and a later fit restores captured live state before creating new optimizer progress.
    6. Unsupported sharded or DTensor averaging must fail explicitly until average-state representation and distributed checkpoint restoration have executable semantics.

| Memory | Finite Checks Must Precede Mutation | Date: 04 September 2026 | Time: 12:41 PM | Name: Ghost |

    1. Checking loss or gradients at batch end is too late because the optimizer may already have consumed non-finite values and corrupted parameters or state.
    2. Loss finiteness belongs at `on_before_backward`; gradient finiteness belongs at `on_after_backward`, which serves automatic, manual, accumulated, mixed-precision, and distributed backward paths before stepping.
    3. Gradient inspection must examine coalesced sparse values rather than applying dense-only tensor operations to sparse storage.
    4. A numerical guard should name what it checks and expose a strict gradient switch; checking gradients is not checking parameters.
    5. Distributed failure evidence must establish that every worker exits and no optimizer mutation occurred, not merely that one rank printed an exception.

| Memory | Learning Rate Is Optimizer-Use Telemetry | Date: 04 September 2026 | Time: 12:47 PM | Name: Ghost |

    1. Step learning rate belongs immediately before a real optimizer step, not at an arbitrary batch boundary, because accumulation, manual optimization, multiple optimizers, and repeated closures decouple batches from optimizer use.
    2. Epoch learning rate belongs at epoch start after the prior epoch scheduler has advanced; scheduler configuration controls automatic cadence only when the callback does not force one interval.
    3. Metric identity includes scheduler name or optimizer type, repeated optimizer occurrence, and parameter-group name. Ambiguous explicit names must fail rather than overwrite another optimizer's telemetry.
    4. Optimizer parameter groups may grow during fine-tuning; the first historical series must migrate from the unsuffixed key to the first-group key when a second unnamed group appears.
    5. Resume state requires only the last emitted step and epoch coordinates. Persisting the complete learning-rate history would bloat checkpoints while the experiment logger already owns durable history.

| Memory | Throughput Follows the Data Axis | Date: 04 September 2026 | Time: 12:52 PM | Name: Ghost |

    1. Total distributed world size is not necessarily sample parallelism; tensor-parallel workers cooperate on the same samples, so only the Strategy's data-parallel dimension multiplies local batch cardinality.
    2. End-to-end training throughput must include dataloader wait and optimizer work but exclude validation pauses from the training clock.
    3. Accelerator timing requires synchronization immediately before reading the clock, or queued CUDA and MPS work makes elapsed time and rate measurements dishonest.
    4. Recursive inference may accept a batch size only when all reachable tensors agree. A domain-specific callable is the honest boundary for structures whose cardinality cannot be inferred generically.
    5. Accumulation windows emit on optimizer progress while counting every contributing microbatch, preserving both sample and batch rates.

| Memory | Device Statistics Belong to Accelerators | Date: 04 September 2026 | Time: 12:58 PM | Name: Ghost |

    1. A device-monitor callback should schedule and route observations, while the selected Accelerator owns which counters exist, how the device is addressed, and what each backend can report.
    2. CUDA allocator state, Apple MPS memory state, and CPU process or load state are different domains; forcing them through CUDA globals misreports the selected runtime.
    3. Filter validation must occur against the live backend's available keys so misspellings fail before silently producing an empty metric stream.
    4. Device statistics are per-rank observations; the current single experiment logger records global rank zero, while callback values remain on the selected device.

| Memory | Early Stopping Requires One Distributed Score | Date: 04 September 2026 | Time: 01:03 PM | Name: Ghost |

    1. Early stopping patience counts monitor checks, not epochs; validation frequency therefore changes the minimum training horizon without changing callback patience.
    2. Rank-zero-only metrics require a rank-zero broadcast before evaluation, otherwise some workers stop while others raise a missing-monitor error or continue into unmatched collectives.
    3. Target attainment, divergence, non-finite values, patience exhaustion, and improvement are distinct state transitions and should retain explicit reasons rather than one opaque boolean.
    4. Configuration identity and persisted best score, wait count, epoch, and reason must align so several stopping policies can coexist and resume without stealing one another's state.
    5. Missing monitors must fail consistently on every rank under strict mode or produce one rank-zero warning without mutating patience under non-strict mode.

| Memory | Timer Separates Budget and Presentation Clocks | Date: 04 September 2026 | Time: 04:50 PM | Name: Ghost |

    1. The resumable training budget is continuous wall time from train start through validation and callback work, while active epoch-training time pauses around validation so progress rates remain honest.
    2. Latest validation, test, and prediction durations are stage observations, not additional offsets in the training budget; duplicate dictionaries for cumulative and latest state create contradictory authorities.
    3. A distributed time limit must be decided by one rank and broadcast because independent wall clocks cross the boundary at different instants.
    4. Restored elapsed time must be checked before the next training batch, the reached flag must cross launcher state, and persisted duration plus interval must match the new Timer before continuation.
    5. Duration strings and mappings are boundary input and require strict field ranges, unit names, integer values, and a positive finite resolved interval owned by Timer rather than a static parser class.
