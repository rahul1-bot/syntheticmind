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

| Architecture State | Hardened Trainer Runtime Foundation | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. ✅ Standard fit, validation, test, and prediction lifecycles preserve hook order, nested execution context, model modes, stage state, failure causality, and transactional teardown.
    2. ✅ Epoch, optimizer-step, validation-cadence, batch-limit, sanity, wall-clock, fast-development, overfit, stop, reload, and unsized-loader controls use established public meanings.
    3. ✅ CPU, CUDA, MPS, device parsing, node topology, distributed sampler ownership, subprocess, spawn, fork, notebook-fork, signal coordination, and parent-state reconciliation form one execution layer.
    4. ✅ Precision, inference mode, gradient enablement, deterministic execution, anomaly detection, clipping, accumulation, automatic optimization, manual optimization, scheduler cadence, and optimizer-state placement are executable.
    5. ✅ Logger orchestration, checkpoint callbacks and paths, profiling, model summaries, barebones mode, plugin composition, artifact roots, remote filesystems, and asynchronous checkpoint I/O retain independent ownership.
    6. ✅ The terminal renderer uses minimal rank-zero output, honest stage timing, one active status line, bounded selected metrics, stable redirected text, and no dense telemetry dump.

| Architecture State | Hardened Data, Metrics, and Continuation | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. ✅ Model-owned, datamodule-owned, and direct Trainer dataloaders share explicit ownership without fabricated adapters or identity loss.
    2. ✅ Multiple and nested loaders use standard combination modes, loader indices, metric suffixes, batch limits, prediction ordering, transfer hooks, and distributed sharding.
    3. ✅ Stateful and stateless loader continuation preserves logical progress, cursor ownership, random streams, unsized prefetch boundaries, subclass reconstruction, and distributed parent reconciliation.
    4. ✅ `Module.log` and `log_dict` preserve standard parameter vocabulary, scalar validation, step and epoch routing, weighted reductions, distributed process groups, TorchMetrics ownership, and multiple-dataloader identity.
    5. ✅ Callback, logged, progress, logger, evaluation-result, scheduler, early-stopping, and checkpoint-monitor metric consumers retain distinct routing and tensor-versus-scalar boundaries.
    6. ✅ Tensor metric state preserves dtype, device, detachment, explicit graph propagation, safe persistence, and safe multiprocessing transfer.

| Architecture State | Hardened Model Extension Surface | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. ✅ `ModelHooks`, `DataHooks`, `CheckpointHooks`, and Callback method families preserve the pinned extension names, positional shapes, defaults, and dispatch points under SyntheticMind ownership.
    2. ✅ `configure_callbacks` composes model callbacks transactionally before setup and checkpoint restoration with inheritance-based replacement, stable ordering, and launcher-safe invocation.
    3. ✅ `Module.loggers` exposes ordered logger ownership, and `Module.print` retains Python print semantics with globally rank-safe, progress-compatible output.
    4. ✅ `Module.all_gather(data, group=None, sync_grads=False)` preserves nested structures, device conversion, world-size-one identity, subgroup rank order, and optional differentiable gathering.
    5. ✅ `Module.load_from_checkpoint` reconstructs subclasses from saved or external hyperparameters, applies explicit overrides, honors map location and strictness, runs configuration and load hooks before state application, and restores the mapped device.
    6. ✅ Live Trainer checkpoint restoration shares the model classmethod's hook order and strictness meanings while retaining complete optimizer, scheduler, callback, datamodule, precision, loop, loader, and random state.

| Architecture Decision | Checkpoint Construction Security and Compatibility | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. Saved constructor hyperparameters, historical SyntheticMind keys, YAML or CSV replacement files, and explicit keyword overrides follow deterministic precedence.
    2. Unknown saved values are filtered when a subclass has no arbitrary keyword parameter, while wrapper-style `hparams` constructors receive overrides within their wrapped mapping.
    3. Local, remote-filesystem, and seekable binary checkpoint sources are supported with PyTorch-compatible explicit map-location forms and automatic unavailable-accelerator fallback to CPU.
    4. `configure_model` executes before `on_load_checkpoint`, and the load hook executes before `load_state_dict`, allowing deferred layers and payload mutation to participate correctly.
    5. Explicit model `strict_loading` and call-level `strict` settings are consistent across class reconstruction and Trainer restoration; conflicts fail and relaxed key differences remain visible.
    6. Checkpoint-controlled external instantiators are rejected because CLI integration code does not own SyntheticMind model construction and arbitrary import paths are executable behavior.
    7. Three focused checkpoint services separate payload I/O, hyperparameter reconstruction, and model restoration without copied Lightning internals or new source comments.
    8. Manual, Trainer-produced, CPU, MPS, unavailable-device, subprocess, spawn, fork, stateful-continuation, and direct two-rank Gloo evidence establishes the current contract without static-analysis tools.

| Architecture Decision | Standard Model Runtime State | Date: 19 August 2026 | Time: 11:37 PM | Name: Ghost |

    1. `freeze()` and `unfreeze()` now return the same model instance after changing parameter gradient state and recursive module mode, preserving the standard fluent contract.
    2. `current_epoch`, `global_step`, `global_rank`, and `local_rank` now read the attached Trainer as the runtime authority and return zero while detached.
    3. Detached modules no longer inherit an unrelated externally initialized process group's ranks, while attached models continue to reflect the configured Strategy across every launcher.
    4. `on_gpu` now preserves the established CUDA-specific meaning; MPS and CPU placement correctly return false.
    5. Focused CPU, MPS, parameter-state, detached, attached, direct Gloo, subprocess, spawn, fork, compilation, signature, and diff verification passes without static-analysis tools.

| Architecture Decision | Standard TorchScript Export | Date: 19 August 2026 | Time: 11:47 PM | Name: Ghost |

    1. ✅ `Module.to_torchscript(file_path=None, method="script", example_inputs=None, **kwargs)` now preserves the standard inherited public name, parameter order, defaults, scripting and tracing meanings, and framework-neutral ownership.
    2. Script and trace execution use evaluation mode without permanently changing any root or nested module mode, including when compilation, tracing, input hooks, or validation fail.
    3. Trace inputs use explicit values before `example_input_array`, pass through the established batch-transfer hook chain, move to the model device, support positional tuples and documented keyword dictionaries, and forward trace options including prepared check inputs.
    4. Exported artifacts save through the existing local and remote filesystem boundary and reload through `torch.jit.load`; nested SyntheticMind modules exclude runtime-only Trainer, logger, metric, device, and hyperparameter properties from scripting.
    5. CPU, MPS, attached and detached models, nested scripting, positional and keyword tracing, hook precedence, local and remote reload, exception restoration, compilation, signatures, inheritance, source, and diff checks pass without a linter or static type checker.

| Architecture Decision | Standard ONNX Export | Date: 19 August 2026 | Time: 11:55 PM | Name: Ghost |

    1. ✅ `Module.to_onnx(file_path=None, input_sample=None, **kwargs)` now preserves the standard inherited public signature and returns the installed PyTorch exporter's `ONNXProgram` or `None` without vendor branding.
    2. The modern dynamo path follows the installed PyTorch default, supports in-memory programs and exporter verification, while the legacy path requires a destination and retains its `None` return contract.
    3. Explicit input samples precede `example_input_array`; positional tuples, named dictionaries, mixed positional and keyword calls, and explicit exporter keyword inputs pass through the established model-or-datamodule transfer ownership.
    4. Exact nested model modes restore after modern, legacy, hook, serialization, and forward failures; inference export defaults to evaluation while explicit PyTorch training-mode controls retain their meanings.
    5. Local paths, `Path`, `BytesIO`, remote filesystems, and external-data sidecars use one artifact boundary, with parent creation and remote sibling preservation handled independently of ONNX graph conversion.
    6. ONNX model checking and real ONNX Runtime inference pass for modern, legacy, dynamic-shape, dynamic-axis, positional, keyword, CPU, and MPS exports; compilation, signatures, inheritance, imports, source, diff, TorchScript, and checkpoint regressions also pass without static-analysis tools.

| Architecture Decision | Standard TensorRT Compilation Surface | Date: 20 August 2026 | Time: 12:06 AM | Name: Ghost |

    1. ✅ `Module.to_tensorrt(file_path=None, input_sample=None, ir="default", output_format="exported_program", retrace=False, default_device="cuda", **compile_kwargs)` now preserves the pinned public name, parameter order, defaults, accepted IR choices, persistence controls, and framework-neutral ownership.
    2. Explicit input samples precede `example_input_array`, are copied before transfer, follow model-or-datamodule transfer ownership, and retain positional, named, and mixed forward-call meanings supported by the selected Torch-TensorRT IR.
    3. Compilation requires the real `torch_tensorrt` package and a CUDA target; configuration, duplicate input ownership, unsupported output formats, missing samples, and unavailable CUDA fail before engine construction with explicit diagnostics.
    4. Exact nested model modes and original device placement restore after input preparation, compilation, and failure; TorchScript compilation also restores the recursive scripting guard.
    5. TorchScript artifacts use PyTorch serialization, while Dynamo and default IR artifacts use Torch-TensorRT serialization with input, output-format, and retrace state preserved across local, byte-buffer, and remote destinations.
    6. The local Apple-silicon host cannot execute a real TensorRT engine build because Torch-TensorRT supports CUDA platforms and the installed PyTorch runtime has no CUDA. Dependency, validation, transfer, restoration, dispatch, and persistence orchestration pass against a source-grounded test double, but CUDA engine compilation and reload remain explicitly unverified.
    7. The contract was compared with the pinned Lightning source and current Torch-TensorRT source revision `da868ec8fcc6ec56fbb63fd3dba944ca7f072304`; current upstream tests also exclude TorchScript IR on PyTorch 2.9 because that combination can crash.
    8. TensorRT, TorchScript, ONNX Runtime, runtime-state, checkpoint, compilation, signature, inheritance, source, and diff regressions pass without invoking any linter or static type checker.

| Active Hold | Model-Parallel Work and Recursive Execution | Date: 20 August 2026 | Time: 12:35 AM | Name: Ghost |

    1. Rahul stopped the recursive hardening run after the model-parallel phase expanded into broad cross-cutting changes without an accepted narrow design checkpoint.
    2. The current `dev` snapshot contains interrupted model-parallel work across Module, Trainer, Strategy, checkpoint state, DDP timeout handling, package exports, and three new strategy modules.
    3. This snapshot is unverified, unaccepted, and not a completed capability. Its commit preserves the exact review state only and must not be interpreted as architectural approval.
    4. The current additions, including the new future-annotations import, vague helper boundaries, and expanded generic Strategy surface, require line-by-line review against `code-guidelines.md` before any continuation.
    5. Future implementation must proceed through Rahul-aligned, narrowly bounded slices with the relevant existing files and runtime path reviewed before mutation.
    6. The recursive goal is stopped. No further code changes, verification expansion, or parity work may continue until Rahul explicitly selects the next bounded action.

| Architecture Decision | Model-Parallel Baseline Recovery | Date: 04 September 2026 | Time: 05:26 AM | Name: Ghost |

    1. Rahul explicitly resumed complete SyntheticMind hardening and established the current root workspace `code-guidelines.md` as the governing Python standard; the repository-local historical guideline copy is reference evidence only and has no current authority.
    2. The interrupted model-parallel checkpoint functions were consolidated into one private persistence owner that controls full and sharded state extraction, metadata loading, restoration, path recognition, and removal without adding a public API or Protocol.
    3. Distributed module materialization now belongs to the existing module-initialization capability, while parallel-size validation remains a private operation of `ModelParallelStrategy`; the one-function materialization module was removed.
    4. The default `Module.predict_step` now invokes the inherited PyTorch module call path, preserving registered forward hooks and compiled-call machinery instead of calling `forward` directly.
    5. Executable evidence covers two-rank CPU tensor-parallel training and resume for both sharded and full checkpoints, rejection of optimizer-incomplete resumable state before model mutation, meta-parameter materialization, ordinary checkpoint loading, callback composition, DDP multiple-loader execution, distributed logging, and local or MPS gathering.
    6. Sharded checkpoint replacement still removes the prior directory before the next distributed generation commits and therefore remains outside the accepted durability contract; atomic generation publication is the next bounded checkpoint task.

| Architecture Decision | Atomic Sharded Checkpoint Generations | Date: 04 September 2026 | Time: 05:44 AM | Name: Ghost |

    1. Sharded checkpoints now store immutable distributed generations beneath one checkpoint directory and publish visibility through a strict versioned `active_generation` record rather than deleting the readable checkpoint before writing its replacement.
    2. The pointer records the active and immediately previous published UUIDv7 generation identifiers; readers that resolved the former active generation remain valid across one publication, while failed and unpublished generations are removed after the next successful commit.
    3. Local pointer replacement uses a same-directory temporary file, file synchronization, atomic replacement, and directory synchronization; non-local fsspec publication runs within the filesystem transaction boundary.
    4. The storage owner rejects incomplete generations, invalid or future pointer schemas, optimizer-incomplete resume state, and destructive full-to-sharded or sharded-to-full path collisions before replacing the existing artifact.
    5. Model-parallel checkpoint internals now occupy the semantic `strategies/model_parallel_checkpointing/` subpackage, preserving the established `ModelParallelStrategy` import and class identity while reducing the top-level strategy package from the topology-review threshold to five direct modules.
    6. Full and sharded uninterrupted-versus-resumed runs produce bit-identical three-step model state, and distributed failure injection confirms that shard, metadata, and pointer failures preserve the previously published generation on both ranks.

| Architecture Decision | Structural Strategy and Launcher Contracts | Date: 04 September 2026 | Time: 07:05 AM | Name: Ghost |

    1. The complete current root workspace `code-guidelines.md` was reread before the strategy rewrite, and the abandoned ABC-based draft was rejected before commit because it combined contract, shared state, and default lifecycle behavior.
    2. `Strategy`, `ParallelStrategy`, `StrategyCheckpoint`, `CheckpointIO`, and the launcher-facing `LaunchStrategy` are structural Protocol contracts with current multiple implementations or consumers. Concrete project implementations inherit their contracts explicitly and mark implemented methods with `@override`.
    3. `ModelParallelStrategy` no longer inherits `DDPStrategy`. DDP owns replica wrapping, model parallelism owns device-mesh topology and distributed state, and both compose the process-group resource and wrapped-step redirection algorithm without a generic mutable strategy context.
    4. Checkpoint reading, state restoration, optimizer restoration, storage selection, and teardown now cross one strategy-owned checkpoint capability. Local storage uses the selected `CheckpointIO`; model-parallel storage owns full or sharded distributed state and atomic generation publication.
    5. Process groups are process-scoped resources and therefore survive individual fit, validation, test, and prediction teardown. Failure handling and process exit release the owned resource, preventing both sequential-stage rendezvous failure and invalidation of live DTensor meshes.
    6. The complete `strategies/` Python package now contains no `Any`, generic `object`, ABC, `abstractmethod`, module-level function, custom `__call__`, source comment, docstring, or terminal trailing comma. Existing `TYPE_CHECKING` blocks remain exactly preserved as required by the governing guideline.
    7. Two-rank automatic and manual optimization, all-rank and any-rank decisions, validation, test, prediction, forward-hook traversal, checkpoint selection, full and sharded exact resume, failure injection, subprocess, spawn, fork, signal propagation, asynchronous checkpoint snapshots, Python compilation, and direct diff review establish the accepted runtime path without a linter or static type checker.

| Architecture Decision | Accelerator and Execution Plugin Contracts | Date: 04 September 2026 | Time: 07:21 AM | Name: Ghost |

    1. `Accelerator`, `ClusterEnvironment`, and `LayerSync` are now contract-only runtime-checkable Protocols because each is a current Trainer plugin or substitution boundary. CPU, CUDA, MPS, local-cluster, and synchronized-batch-normalization implementations inherit their contracts explicitly and mark all implementations with `@override`.
    2. GPU device selection moved from an orphan module function into a stateful parser that owns accelerator identity, available-device count, count expansion, explicit identifiers, uniqueness, and availability invariants.
    3. `DeviceTopology` is now a strict immutable Pydantic resolved-configuration model with class-owned resolution. Its device identifiers and participating devices use tuples at the immutable boundary, while Trainer preserves the established public list representation.
    4. `PluginConfiguration` is now a strict immutable Pydantic resolved record rather than a dataclass, and plugin conflict resolution validates checkpoint, precision, environment, and layer-synchronization categories through their actual contracts.
    5. Recursive floating-point conversion moved onto the concrete precision owner, eliminating the stateless converter class while preserving nested tensor, mapping, tuple, named-tuple, and list behavior across half, double, and Transformer Engine modes.
    6. Transformer Engine integration now targets the NVIDIA 2.18 API through one optional adapter, uses `autocast(enabled=True, recipe=...)`, enforces current sixteen-by-sixteen Linear replacement constraints, and accepts a strict delayed-scaling configuration or live structural recipe instead of an unvalidated mapping.
    7. Accelerator, plugin, and precision Python modules now contain no `Any`, generic `object`, ABC, abstract method, module-level function, custom `__call__`, source comment, docstring, or terminal trailing comma. CPU, MPS, unavailable CUDA, direct and spawned DDP, local model parallelism, precision contexts, layer-sync round trips, plugin conflicts, strict recipe validation, and a source-shaped Transformer Engine adapter probe pass without static analysis.

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

| Architecture Decision | Accelerator Connector Ownership | Date: 04 September 2026 | Time: 09:07 AM | Name: Ghost |

    1. Trainer accelerator composition now belongs to one internal `AcceleratorConnector` covering backend selection, device parsing, node count, Strategy selection, precision, cluster environment, checkpoint I/O, and layer synchronization.
    2. The former `DeviceTopology`, `PluginConfiguration`, `PluginResolver`, and `PrecisionResolver` surfaces were removed because they fragmented one compatibility decision and used temporary runtime records instead of an owning lifecycle component.
    3. Accelerators retain only backend-specific discovery, parsing, and binding. Strategies retain execution semantics, while the connector validates their combined topology and injects the selected plugins in deterministic order.
    4. Precision construction remains dependent on accelerator and Strategy compatibility, including CPU float16 fallback, explicitly supplied precision plugins, and model-parallel precision restrictions.
    5. The connector is internal and creates no alternate stage entrypoint; Trainer remains the public orchestration surface and retains direct properties for the resolved runtime graph.
    6. Configuration matrices, CPU and MPS execution, unavailable CUDA rejection, checkpoint-plugin injection, DDP subprocess, spawn and fork launch, and model-parallel full and sharded execution pass after the migration.

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
