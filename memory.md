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

| Memory | TorchScript Export Boundaries | Date: 19 August 2026 | Time: 11:47 PM | Name: Ghost |

    1. TorchScript introspection evaluates Python properties even when they are later excluded, so detached Trainer access requires a narrowly scoped scripting state that is restored under every outcome.
    2. Export mode restoration must capture every named submodule rather than only the root training flag; restoring through `train(root_mode)` destroys deliberately mixed child modes.
    3. A dictionary registered as `example_input_array` represents named forward arguments, while a positional dictionary remains available by wrapping the mapping in a one-element tuple.
    4. Representative and check inputs must follow the same model-or-datamodule batch-transfer ownership used during execution so accelerator exports do not trace host tensors against device-resident parameters.
    5. Export serialization belongs to the shared filesystem boundary, while TorchScript compilation and tracing remain PyTorch-owned operations with their original keyword controls and diagnostics.

| Memory | ONNX Export Boundaries | Date: 19 August 2026 | Time: 11:55 PM | Name: Ghost |

    1. PyTorch 2.9 defaults ONNX export to the dynamo engine, which returns an `ONNXProgram`; the legacy TorchScript engine returns `None` and cannot export without a destination.
    2. ONNX positional samples are tuples, dictionaries represent named forward inputs, and a trailing dictionary within a positional tuple carries mixed keyword inputs under PyTorch's compatibility contract.
    3. Exporters can mutate model training flags, especially on legacy failures. Exact named-submodule capture and unconditional restoration are therefore required around conversion and serialization.
    4. Modern ONNX external-data output may create sibling artifacts even for a single requested path; remote persistence must transfer the complete generated sibling set rather than only the primary protobuf.
    5. A parseable ONNX file is insufficient evidence. Model checking, ONNX Runtime execution, numerical comparison, dynamic-shape execution, accelerator-origin export, and failure restoration establish usable capability.

| Memory | TensorRT Compilation Boundaries | Date: 20 August 2026 | Time: 12:06 AM | Name: Ghost |

    1. TensorRT compilation is a CUDA engine capability rather than a generic accelerator export path; MPS placement cannot establish engine compatibility and must fail before compilation when no CUDA target exists.
    2. Torch-TensorRT owns engine lowering and format serialization, while SyntheticMind owns optional-dependency diagnostics, input precedence, transfer hooks, exact model restoration, destination handling, and public method compatibility.
    3. Positional, named, and mixed inputs require distinct `inputs` and `kwarg_inputs` routing. TorchScript IR does not accept named sample inputs through this surface and uses `torch.jit.save` rather than the Dynamo persistence path.
    4. A source-grounded test double can prove wrapper control flow, argument forwarding, exception restoration, and artifact routing, but it cannot prove TensorRT engine construction, numerical correctness, performance, or reload on a CUDA runtime.
    5. Current upstream source and platform support must be separated from the pinned Lightning contract. The current Torch-TensorRT source supports CUDA platforms but not Darwin, while its PyTorch 2.9 tests exclude TorchScript IR because that path may crash.
    6. Optional accelerator capability remains incomplete until its actual hardware and vendor runtime execute successfully; the environmental boundary must remain visible in progress state rather than converted into a false completion claim.

| Memory | Scope Control and Guideline Enforcement | Date: 20 August 2026 | Time: 12:35 AM | Name: Ghost |

    1. A recursive objective does not authorize uninterrupted broad mutation. Every implementation phase still requires a small accepted boundary, direct review of the affected existing code, and an explicit reason for each new abstraction or public contract.
    2. PyTorch Lightning is capability and terminology evidence only. Its decomposition, compatibility imports, helper layout, and internal state boundaries must never enter SyntheticMind without independent justification under `code-guidelines.md`.
    3. Generic helper names and wide base-class expansion hide ownership rather than clarify it when their exact invariant and caller set have not been agreed first.
    4. Fast progress is measured by accepted, verified capability slices rather than elapsed autonomous activity or the volume of changed files.
    5. Interrupted work must be labeled unverified and preserved as a review snapshot, never promoted into the verified architecture state merely because it compiles or resembles the reference.
    6. Rahul-directed checkpoints are now mandatory before resuming the model-parallel diff or opening another capability phase.

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
