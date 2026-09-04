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

| Repository Policy | Private Development and Reference Boundaries | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. The GitHub repository is private, `origin` is canonical, `main` is the stable baseline, and active implementation occurs on `dev`.
    2. Every completed development phase must be committed locally, pushed to `origin/dev`, and verified to leave local `dev` and `origin/dev` at the same commit.
    3. Promotion to `main` requires an explicit stabilization decision; ordinary hardening commits do not enter `main` automatically.
    4. The complete `docs/` tree is ignored. The pinned PyTorch Lightning checkout under `docs/references/code/pytorch_lightning/pytorch-lightning` remains read-only comparative evidence.
    5. Additional source references may be cloned only into ignored `external/` when they answer a concrete unresolved engineering question.
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

| Architecture State | Verified Framework Runtime Baseline | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. ✅ Fit, validation, test, prediction, sanity, failure, and teardown lifecycles preserve standard hook order, module modes, stage state, causal exceptions, and run controls.
    2. ✅ CPU, CUDA, MPS, precision, automatic and manual optimization, scheduler cadence, distributed launchers, samplers, signals, and parent-state reconciliation form an executable training runtime.
    3. ✅ Model, DataModule, direct, multiple, nested, sized, unsized, stateful, and distributed data paths preserve ownership, identity, limits, ordering, and exact continuation.
    4. ✅ Metric production, graph state, reduction, TorchMetrics, callback views, progress views, logger output, scheduler monitoring, early stopping, and checkpoint ranking remain distinct and coherent.
    5. ✅ Model, data, checkpoint, and callback hooks; model-provided callbacks; logger access; rank-safe printing; nested gathering; and model checkpoint reconstruction preserve established extension semantics.
    6. ✅ Checkpoints, profiler actions, model summaries, storage, asynchronous I/O, barebones execution, and terminal presentation retain explicit owners and verified local or distributed paths.

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

| Architecture Decision | Stateful Strategy Hierarchy | Date: 04 September 2026 | Time: 09:00 AM | Name: Ghost |

    1. `Strategy` now owns universal model connection, wrapped-step dispatch, precision-mediated backward and optimizer stepping, single-process collective defaults, initialization context, and ordinary checkpoint collaboration without using an ABC.
    2. `ParallelStrategy` now owns the cluster environment, process-group lifecycle, rank topology, process launcher, distributed collectives, sampler coordinates, and distributed tail-gradient synchronization directly.
    3. `SingleDeviceStrategy` is a named execution policy with inherited single-device semantics; `DDPStrategy` contains only DDP wrapping, layer synchronization, and manual-optimization forward behavior; `ModelParallelStrategy` contains device-mesh, parallelization, precision restrictions, and distributed checkpoint selection.
    4. The former `distributed_runtime.py`, `forward_redirection.py`, and `model_parallel_topology.py` modules were removed because their behavior belongs to the Strategy concepts they served rather than to independent public building blocks.
    5. Direct source comparison found thirty-one method names repeated across all three former concrete strategies, nine identical method bodies across all three, and twenty-three identical bodies between DDP and model parallelism; the new hierarchy removes that duplication through true semantic inheritance.
    6. Single-device fitting, two-rank DDP manual optimization and wrapped forwards, multi-stage distributed dataloaders, spawn and fork launch, and full and sharded model-parallel exact resume pass after the hierarchy change.

| Architecture Decision | Accelerator, Precision, and Connector Ownership | Date: 04 September 2026 | Time: 09:07 AM | Name: Ghost |

    1. Runtime-checkable Accelerator, ClusterEnvironment, LayerSync, and checkpoint storage contracts retain current substitution boundaries with explicit implementations and genuine `@override` methods.
    2. Trainer accelerator composition belongs to one internal `AcceleratorConnector` covering backend and device parsing, nodes, Strategy, precision, cluster environment, checkpoint I/O, and layer synchronization.
    3. The former temporary topology, plugin-configuration, and resolver objects were removed; a stateful GPU parser owns count, identifier, uniqueness, and availability invariants.
    4. Accelerators own backend discovery, binding, and device-local questions; Strategies own execution semantics; the connector validates their compatibility and injects plugins in deterministic order.
    5. Precision construction covers CPU float16 fallback, model-parallel restrictions, recursive conversion, and the NVIDIA Transformer Engine 2.18 autocast and strict delayed-scaling boundary.
    6. Trainer remains the sole public composition surface. CPU, MPS, unavailable CUDA, precision contexts, layer synchronization, plugin conflicts, DDP launchers, model parallelism, and Transformer Engine adapter behavior pass.

| Architecture Decision | Callback Connector Ownership | Date: 04 September 2026 | Time: 09:13 AM | Name: Ghost |

    1. Callback configuration now belongs to one internal `CallbackConnector` covering Trainer-supplied callbacks, framework defaults, checkpoint ordering, timers, progress display, model summaries, and callback state identities.
    2. Model-provided callbacks are configured once per stage, replace compatible Trainer callbacks through the established inheritance rule, and remain ordered before checkpoint callbacks.
    3. The connector accepts a bound configuration operation rather than a concrete Trainer reference, preserving module hook context and profiler ownership without introducing a circular import or synthetic Trainer-host Protocol.
    4. Trainer continues to own callback hook execution and stage sequencing; the connector owns only callback composition and configuration state, so no competing lifecycle entrypoint was introduced.
    5. Invalid return values, non-callback sequence members, duplicate state keys, illegal configure-hook logging, and callback setup failures leave the configuration boundary reusable for a later stage.
    6. Fit, validation, test, prediction, checkpoint participation, repeated stages, serialization, DDP spawn, and DDP fork verification pass after the migration.

| Architecture Decision | Signal Connector Placement | Date: 04 September 2026 | Time: 09:15 AM | Name: Ghost |

    1. Process-signal registration, nested registration depth, launched-worker termination, original-handler chaining, serialization cleanup, and teardown now live in `trainer/connectors/signal_connector.py`.
    2. The connector depends only on the selected Strategy because worker termination is a launcher responsibility; the former full-Trainer dependency and conditional type import were removed.
    3. Signal handler state now uses a closed union covering Strategy, handlers, booleans, integers, and absence rather than a generic object dictionary.
    4. DDP spawn serialization and live SIGTERM propagation pass, the originating process exits through the expected interruption path, and no owned child remains attached after completion.

| Architecture Decision | Data Connector Ownership | Date: 04 September 2026 | Time: 09:26 AM | Name: Ghost |

    1. Dataloader source ownership, module or DataModule factory invocation, direct and nested loaders, combination modes, limits, reload cadence, overfit determinism, sampler injection, and prepared-loader state now belong to `trainer/connectors/data_connector.py`.
    2. Trainer retains stage sequencing, data preparation and setup hook order, while loops consume the connector-owned prepared loaders and batch totals rather than parallel Trainer fields.
    3. Distributed training uses padded `DistributedSampler` semantics where equal work is required, while prediction uses an unrepeated strided sampler so the union of rank partitions contains every sample exactly once without overlap.
    4. DataLoader subclass reconstruction preserves standard constructor values and current subclass-required attributes during sampler replacement, including stateful loader continuation and worker configuration.
    5. The connector's cohesive class exceeds the generic 500-line module budget under Rahul's explicit SyntheticMind concept-integrity authority. Splitting source resolution, combination, limits, sampler policy, and reconstruction would fracture one data-composition lifecycle and is therefore rejected.
    6. The connector contains no `Any`, generic `object`, module-level function, source comment, custom callable object, or unannotated assignment; Trainer source comments were removed from the rewritten module rather than retained as stale architecture.
    7. Direct, module-owned, and DataModule-owned execution, nested loaders, spawn and fork transfer, two-rank DDP, model parallelism, stateful checkpoint continuation, overfit batches, reload cadence, batch limits, persistent workers, and unrepeated prediction all pass.

| Architecture Decision | Logger Connector Ownership | Date: 04 September 2026 | Time: 09:33 AM | Name: Ghost |

    1. Logger selection, default run-version allocation, callback metrics, logger metrics, progress metrics, stage reset boundaries, scalar conversion, hyperparameter merging, and external logger emission now belong to the Trainer logger connector.
    2. Modules retain metric declaration and computation, loops retain step and epoch reduction cadence, and concrete loggers retain backend persistence. The connector owns routing among these concepts without adding a parallel logging API.
    3. The former `loggers/configuration.py` free-function module was removed because logger selection and destination identity are part of connector composition rather than a standalone utility surface.
    4. Callback, logged, and progress views remain distinct; sanity results do not leak into live views, public properties return defensive mappings, and launcher transfer preserves the selected worker's observable state.
    5. The connector contains no broad annotation, free function, source prose, custom callable object, or unannotated assignment and remains below the project class threshold without artificial fragments.
    6. Local metric views, CSV metrics and hyperparameters, distributed reductions, TorchMetrics, DDP stage transitions, spawn, fork, progress rendering, and launcher state transfer pass after the migration.

| Architecture Decision | Checkpoint Connector Ownership | Date: 04 September 2026 | Time: 09:51 AM | Name: Ghost |

    1. The Trainer checkpoint connector is now the sole full-trajectory coordinator for checkpoint path selection, canonical payload construction, Strategy state loading, scheduler and callback restoration, DataModule and precision state, random streams, loop and loader continuation, storage dispatch, and synchronization.
    2. The former behavior-rich `CheckpointState` namespace class was deleted. Its assembly, restoration, random-state, coordinate, DataModule, and loader-continuation behavior moved onto the live connector that owns checkpoint lifecycle state.
    3. Strategy-side `checkpoint_connector.py` was renamed to `checkpoint.py`, and its ordinary implementation is `_SingleFileCheckpoint`; Connector terminology now refers only to Trainer composition rather than Strategy-local state representation.
    4. SyntheticMind emits and accepts one private schema with `schema_version=1`, `state_dict`, plural optimizer and scheduler state, callbacks, component state, loop continuation, random state, and metadata. Historical duplicate singular, alias, and unversioned forms are rejected.
    5. `ModelCheckpoint` retains save cadence and retention policy, Strategy retains topology-specific state representation, and `CheckpointIO` retains storage transport. Neither callback nor Strategy independently coordinates the full training trajectory.
    6. The cohesive connector exceeds the generic source-size budget under the explicit concept-integrity rule; splitting path, payload, restoration, random streams, and continuation would recreate competing checkpoint owners.
    7. Best and last selection, user-managed paths, weights-only artifacts, strict schema rejection, asynchronous storage, callback and hyperparameter state, exact stateful-loader continuation, DDP spawn and fork, full and sharded model-parallel resume, and injected distributed publication failure pass.

| Architecture Decision | Optimization Concept Boundaries | Date: 04 September 2026 | Time: 10:09 AM | Name: Ghost |

    1. `core/optimizer.py` now owns the model-facing `configure_optimizers` vocabulary, strict scheduler configuration, managed optimizer access, no-op optimizer, and declaration normalization without defining optimizer or scheduler factory families.
    2. `loops/optimization/automatic.py` owns automatic loss extraction, accumulation normalization, closure execution, zeroing, and backward invocation; `manual.py` owns manual-step output normalization while the research Module controls backward and optimizer calls.
    3. TrainingEpochLoop retains batch iteration, accumulation boundaries, scheduler cadence, metric dispatch, and validation scheduling. Strategy retains topology-sensitive model and optimizer execution, and precision retains closure wrapping, numerical stepping, scaling, unscaling, hooks, and clipping.
    4. `SyntheticOptimizer` now uses explicitly connected step and toggle operations. Dynamic `__class__` mutation, generic attribute forwarding, concrete Trainer references, and four optimizer-only Trainer Protocols were removed.
    5. The previous optimizer and scheduler factory configurations and their free-function utility builders were dead parallel APIs that renamed PyTorch algorithms and were deleted.
    6. Automatic, multi-optimizer manual, managed manual DDP, LBFGS repeated closure, accumulation tail, clipping, mixed precision, step scheduling, plateau monitoring, invalid configuration, spawn, fork, full model-parallel resume, and sharded model-parallel resume pass.

| Architecture Decision | Canonical Trainer Package | Date: 04 September 2026 | Time: 10:16 AM | Name: Ghost |

    1. `trainer/trainer.py` is the sole implementation and canonical import path for `Trainer`; `trainer/__init__.py` exposes the same class for the concise public package import.
    2. `TrainerState` moved to `trainer/states.py`, and the former top-level `state/` package was removed because Trainer status and progress are Trainer lifecycle concepts rather than a cross-cutting state subsystem.
    3. Every production dependency and verification program now imports the canonical Trainer and Trainer state paths directly. No compatibility module, forwarding alias, or duplicate import surface remains under `core/` or `state/`.
    4. Existing conditional annotation blocks changed only their Trainer import target under the explicit project topology authority; no new `TYPE_CHECKING` block was introduced.
    5. Canonical imports, Module and DataModule back-references, callback composition, metric views, subprocess DDP, spawn, fork, stateful checkpoint continuation, and sharded model parallelism pass after relocation.

| Architecture Decision | Core Research and Hook Surface | Date: 04 September 2026 | Time: 11:04 AM | Name: Ghost |

    1. Rahul confirmed that established machine-learning concepts and their runtime semantics control SyntheticMind architecture before generic software decomposition rules. The three hook families therefore remain explicit framework contracts rather than being folded into one oversized model implementation.
    2. `ModelHooks`, `DataHooks`, and `CheckpointHooks` now preserve the recognized lifecycle vocabulary and no-op extension semantics used by the Trainer and loops. `Module` follows a cooperative initialization order equivalent to the proven Lightning class graph, while concrete overrides are marked only where a parent hook or PyTorch method is actually replaced.
    3. Metric logging admissibility is owned privately by `ModelHooks` because it classifies the hook lifecycle itself. The unused logging-configuration model and separate rule object were removed because they introduced data-model and class surfaces without an independent boundary or consumer.
    4. `DataModule` and `DatasetDataModule` retain preparation, setup, teardown, loader construction, batch transfer, checkpoint state, and direct-dataset construction as one data concept. Dataset and loader inputs now use recursive domain aliases instead of generic `object` annotations.
    5. Standalone model checkpoint reconstruction moved into `core/saving.py`, while the former checkpoint-loader module and TorchScript, ONNX, and TensorRT convenience modules were removed as deployment baggage outside the active training-harness objective. The canonical checkpoint schema and model hydration order remain executable.
    6. The core model, data, hook, saving, and hyperparameter modules contain no `Any`, generic `object`, or conditional type-only import. Checkpoint-key identifier names are lower snake case while their serialized string values remain unchanged.
    7. Every project-owned ALL_CAPS assignment and terminal trailing comma was removed without renaming external framework constants or environment-variable keys. This naming cleanup is subordinate to the preserved PyTorch and distributed runtime meanings.
    8. Package import, hook inheritance, direct and nested dataset construction, callback composition, metric views, TorchMetrics under spawned two-rank DDP, local checkpoint hydration, optimizer execution, nested distributed loaders, exact stateful-loader resume, and two-rank model parallelism pass after the reconstruction.

| Architecture Decision | Plugin and Launcher Topology | Date: 04 September 2026 | Time: 11:11 AM | Name: Ghost |

    1. Numerical precision is now located under `plugins/precision/` because full, half, mixed, and Transformer Engine policies are interchangeable Trainer plugins controlling numerical execution rather than a peer training lifecycle.
    2. Checkpoint transport moved from the vague `plugins/io.py` name to `plugins/checkpoint_io.py`. The current plugin family owns only checkpoint persistence, so a generic I/O package would imply unsupported transport concepts and reproduce upstream baggage.
    3. The process launcher moved from `strategies/launcher.py` into `strategies/launchers/launcher.py`, placing start-method selection, process launch delegation, state transfer, environment reconstruction, child observation, and termination under one distributed-launch concept.
    4. Public class identities and Trainer configuration meanings remain unchanged, but obsolete import paths were removed without compatibility façades. Precision now explicitly inherits the checkpoint-hook contract and marks its checkpoint hook implementations as overrides.
    5. Full, half, mixed, and Transformer Engine precision execution, asynchronous checkpoint I/O, local optimizer execution, spawned DDP runtime state, subprocess DDP forward and consensus behavior, and sharded two-rank model parallelism pass through the new canonical paths.

| Architecture Decision | Cross-Cutting Ownership Reconstruction | Date: 04 September 2026 | Time: 11:41 AM | Name: Ghost |

    1. Trainer owns run seeding, worker randomness, terminal rendering, deterministic algorithms, benchmark flags, and complete-run configuration; Loop owns evaluation gradient context.
    2. Core data transfer owns recursive batch placement, Module owns its one-consumer gather traversal, Strategy owns topology-sensitive initialization, and the logger connector owns metric accumulation.
    3. `TorchCheckpointIO` owns atomic local and remote serialization, model-parallel checkpointing owns distributed shards, and the explicit storage package owns protocol-aware paths shared by artifact producers.
    4. Framework exception and recursive domain type contracts live at the package root so semantic subsystems depend inward without a utility layer or circular ownership.
    5. Dead compilation, channels-last, context, torch-flag, checkpoint, tensor-traversal, and helper façades were deleted without compatibility aliases.
    6. Local, MPS, spawned DDP, nested collective, TorchMetrics, checkpoint, remote storage, asynchronous I/O, failure-atomic model-parallel, and model-hydration paths pass through the assigned owners.

| Architecture Decision | Training Data Composition Ownership | Date: 04 September 2026 | Time: 11:53 AM | Name: Ghost |

    1. `CombinedLoader` moved from the transitional utility package to the canonical `syntheticmind.combined_loader` module and is exported from the package root as a first-class training-data composition concept.
    2. Root placement preserves dependency direction because both model-facing `DatasetDataModule` construction and outward Trainer stage execution consume the concept; placing it beneath Trainer connectors would force core data contracts to import orchestration.
    3. The four private mode iterators remain colocated because minimum-size, maximum-size, cycling, and sequential traversal are real algorithmic variants constructed, reset, serialized, and consumed exclusively by `CombinedLoader`.
    4. Module-level functions, the mutable mode-dispatch dictionary, generic `object` annotations, and the legacy utility import surface were removed. Recursive batch, iterable, sampler, state, and pickle domains are now explicit, and byte batches are represented by the shared batch scalar contract.
    5. The cohesive 588-line module and 337-line primary class exceed generic structural limits under Rahul's explicit SyntheticMind concept-integrity authority. Splitting tree structure, limits, iteration, state continuation, and worker teardown would create implementation fragments and is therefore rejected.
    6. All four combination modes, finite and unsized limits, nested dataset construction, serialization, state capture and restoration, persistent workers, spawned DDP lifecycle execution, exact stateful resume, and direct two-rank prediction ordering pass through the canonical module.

| Architecture Decision | Strategy-Owned Distributed Reduction | Date: 04 September 2026 | Time: 12:04 PM | Name: Ghost |

    1. The selected Strategy now owns the standard tensor `reduce` operation alongside barrier, broadcast, boolean consensus, gather, ranks, and process-group lifetime; `ParallelStrategy` implements mean, sum, minimum, and maximum collectives with optional process-group selection.
    2. Trainer, loops, and callbacks now gate output and side effects through `trainer.is_global_zero`, preserving selected cluster-environment rank semantics even before a process group is initialized instead of consulting ambient global state.
    3. Logged metrics and metric accumulators receive the Strategy's bound reduction operation at their computation boundary. This supplies only the required capability and avoids a circular core-to-Strategy dependency or a synthetic host Protocol.
    4. The former `DistributedUtils` namespace and its seven duplicate free-function entrypoints were deleted without a replacement utility module. Process-group teardown remains exclusively owned by `ParallelStrategy`.
    5. Local reduction validation, two-rank sum and mean, standard distributed mean, sum, minimum and maximum metrics, TorchMetrics, tensor graph state, progress rendering, checkpoint selection, callback artifacts, spawned metric views, and sharded model parallelism pass through Strategy ownership.

| Architecture Decision | Hook Inspection and Module Mode Ownership | Date: 04 September 2026 | Time: 12:11 PM | Name: Ghost |

    1. Exact recursive train and evaluation mode snapshots now belong to `Module`, the object whose nested PyTorch state is captured and restored by Trainer stages, evaluation and prediction loops, and model-summary forwarding.
    2. Step-signature inspection now belongs to `Module` because it adapts the recognized validation, test, and prediction hook contracts. Trainer owns inherited-hook override discovery beside centralized hook dispatch and preserves wrapped, mocked, and partial callable semantics.
    3. Callback persistence admission uses direct class-contract comparison, while model parallelism compares the concrete `configure_model` implementation against `ModelHooks`; neither narrow case requires the complete generic inspection algorithm.
    4. The former `ModuleMode` data holder and three module-level helper functions were deleted with `utilities/model_helpers.py`. No new class, Protocol, module, public entrypoint, conditional import, or generic annotation replaced them.
    5. Inherited and overridden hooks, decorated and partial callables, DataModule transfer precedence, batch-index adaptation, exact mixed child modes on success and failure, model-summary forwarding, callback state, spawned DDP, and sharded model parallelism pass after consolidation.

| Architecture Decision | Model Summary Capability Ownership | Date: 04 September 2026 | Time: 12:23 PM | Name: Ghost |

    1. Model-summary lifecycle, inspection, and optional rendering now form the semantic `callbacks/model_summary/` package with separate callback, private report, and Rich adapter modules; the callback remains the canonical public lifecycle entrypoint.
    2. Layer shape collection uses a bound inspector method rather than a nested forward-hook function, and deterministic `finally` cleanup replaces destructor-dependent hook removal.
    3. Model size now sums each realized parameter's actual storage bytes instead of inferring storage from Trainer precision labels, preserving correct mixed-precision and custom-dtype accounting without a lookup table.
    4. Example-input forwarding transactionally restores every nested module mode plus Python, NumPy, torch CPU, CUDA, or MPS random state after success or failure.
    5. The former public utility functions, duplicate report name, generic annotations, module constants, local imports, and utility import paths were removed. The final tracked utility files and generated cache were deleted, leaving no `utilities/` package.
    6. Plain and Rich rendering, lazy parameters, hook cleanup, failure rollback, MPS transfer and random state, default Trainer integration, callback configuration, metric regression, and exactly-once spawned rank-zero presentation pass.

| Architecture Decision | Callback Baggage Removal | Date: 04 September 2026 | Time: 12:28 PM | Name: Ghost |

    1. `DataLeakageGuard` was deleted because its sample-identifier batch schema is project-specific, its evidence disappears on resume, and it cannot establish general dataset split integrity from a Trainer callback.
    2. `WatchdogCallback` was deleted because it could observe a long gap only after execution resumed and therefore did not detect, interrupt, or recover a live stall despite its name.
    3. The callback `RuntimeProfiler` and its accumulator were deleted because the canonical profiler subsystem already measures dataloader and step actions, while the callback duplicated logger emission, artifact persistence, device telemetry, and timing state through private Trainer coupling.
    4. The RuntimeProfiler-only dataloader timing dictionaries, accessors, and loop measurements were removed. Existing profiler action contexts remain the single timing authority.
    5. Callback package width fell from fourteen to eleven direct modules, below the hard topology limit. No additional folder was created because the remaining direct modules are independently selectable training concepts rather than fragments of one capability.
    6. SimpleProfiler action evidence, local stage execution, spawned DDP metrics and progress, and sharded model parallelism pass after deletion.

| Architecture Decision | Resumable EMA Weight Averaging | Date: 04 September 2026 | Time: 12:38 PM | Name: Ghost |

    1. The unvalidated `EMACallback` shadow-model copy was replaced by the standard `EMAWeightAveraging` concept with strict decay, update cadence, starting step, and buffer configuration.
    2. Averaging initializes after model configuration, placement, and checkpoint restoration, advances only when global optimizer-step progress changes, copies the first eligible state exactly, averages floating tensors, and copies non-floating buffers.
    3. Average tensors, processed step, and update count persist through the canonical callback state channel. Checkpoints created while average weights are active restore the live training tensors under the sole canonical `state_dict` key, including weights-only saves.
    4. Validation, test, and prediction swap average weights transactionally; exception and teardown paths restore live tensors, while successful training ends with the average state only after at least one real update.
    5. DTensor and sharded model state are rejected explicitly because correct distributed average persistence is not implemented; DDP and MPS are supported and verified.
    6. The cohesive 402-line module and 376-line callback exceed the generic class threshold under the concept-integrity authority; splitting one averaging algorithm into a one-consumer manager or store is rejected.
    7. CPU parameter and buffer behavior, accumulation cadence, invalid configuration, failure rollback, full and weights-only mid-swap checkpoints, exact interrupted resume, MPS placement, spawned DDP transfer, and model-parallel rejection pass.

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

| Architecture Decision | Resumable Wall-Clock Timer | Date: 04 September 2026 | Time: 01:10 PM | Name: Ghost |

    1. Timer now has one continuous resumable training budget clock, one active epoch-training clock paused during validation, and one latest-pass clock for fit, validation, test, and prediction.
    2. Step and epoch interruption cadence remains explicit, while validation completion checks a step-level or epoch-boundary limit so long validation cannot hide an exhausted budget.
    3. Global rank zero evaluates the wall-clock boundary and broadcasts the decision, avoiding drift-dependent all-rank consensus.
    4. Elapsed training time and the reached flag persist with strict finite state; a depleted checkpoint stops before another batch and spawned workers reconcile the outcome to their parent.
    5. Duration strings enforce clock-field ranges, mappings enforce supported non-negative integer units, and every duration resolves to a positive finite interval.
    6. Stage-clock isolation, validation pause, step stopping, depleted restore, invalid duration and state, progress rendering, and spawned DDP consensus pass.
    7. The cohesive 464-line module and 443-line class exceed generic structural limits under the concept-first authority; fake clock-store, duration-factory, or stop-manager classes are rejected.
