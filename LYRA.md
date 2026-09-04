| Strategic Note | SyntheticMind Development Intent | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. SyntheticMind is private infrastructure for Rahul's study, research, experimentation, and direct ownership of complete machine-learning workflows.
    2. The target is full user-facing core PyTorch Lightning capability parity without becoming a Lightning clone or importing its product identity.
    3. Fabric, public documentation, release machinery, promotional work, and upstream test-suite replication remain outside the active objective.
    4. The harness is incomplete. Capability claims require implemented runtime behavior and focused evidence rather than declared parameters or surface resemblance.
    5. New source comments and docstrings remain prohibited. Rewritten legacy modules are explicitly in scope for removal of stale or misleading prose, while unrelated surviving comments remain unchanged.

| Strategic Note | Public Contracts and Internal Ownership | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. Established machine-learning method names, variable names, parameter names, defaults, meanings, hooks, state, and inheritance behavior are mandatory compatibility boundaries.
    2. PyTorch Lightning provides mature reference evidence, while SyntheticMind owns its classes, architecture, implementation, errors, and non-vendor terminology.
    3. `SyntheticOptimizer` is the accepted managed adapter name; the algorithm remains `torch.optim.Optimizer`, and raw access remains explicit.
    4. Callback hooks use `module`, not Lightning-branded `pl_module`, while preserving standard hook names, positional order, defaults, and override behavior.
    5. Internal implementation follows `code-guidelines.md` through direct review and uses composition and focused ownership rather than copied framework internals.

| Strategic Note | Sequential Operations and Evidence | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. Work proceeds from foundational contracts upward after tracing each runtime path completely; random feature insertion into unmapped code is prohibited.
    2. Local source, pinned reference source, installed runtime behavior, Git history, official primary material, and ignored external checkouts form the evidence escalation order.
    3. Every accepted phase requires runtime success, failure probes, signature and inheritance inspection, compilation, diff review, ledger maintenance, commit, push, and synchronized branches.
    4. `ruff`, `mypy`, `pyright`, all other linters, and all static type checkers are prohibited; direct guideline review and executable verification control acceptance.
    5. Git retains chronology. Ledgers retain current strategy, durable knowledge, verified capability state, and live next actions only.

| Strategic Note | Verified Core Runtime Position | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. ✅ Trainer lifecycle, termination, validation cadence, limits, debugging controls, topology, launchers, signals, precision, optimization, scheduling, cleanup, and failure causality form a verified runtime foundation.
    2. ✅ Logger, checkpoint, profiler, model-summary, plugin, filesystem, asynchronous I/O, artifact-root, and barebones capabilities retain distinct ownership.
    3. ✅ Model, datamodule, direct, multiple, nested, stateful, stateless, sized, unsized, worker, and distributed data paths preserve identity and continuation.
    4. ✅ Metric logging, reduction, inspection, TorchMetrics, graph state, monitoring, progress, result, logger, and multiprocessing boundaries are coherent.
    5. ✅ Model callback configuration, logger access, rank-safe printing, nested gathering, and class checkpoint reconstruction now extend the standard model inheritance surface.

| Strategic Note | Console Experience Boundary | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. The terminal remains a minimal presentation subscriber rather than a telemetry warehouse or duplicate experiment logger.
    2. One active interactive line, one completed-epoch line, and one final stage line expose stage, progress, at most two selected metrics, and honest estimated time when available.
    3. Independent stage timing, sanity exclusion, training pauses around validation, stage-isolated metrics, and truthful unknown horizons prevent polished misinformation.
    4. Redirected and non-interactive execution remains stable plain text, and distributed rendering remains rank-zero-owned.
    5. Detailed device, throughput, dataloader, optimizer, profiler, and metric telemetry belongs in structured programmatic and logger outputs.

| Strategic Note | Checkpoint Reconstruction Contract | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. ✅ Model subclasses reconstruct through the standard `load_from_checkpoint` classmethod without requiring a Trainer or exposing a Lightning-owned identity.
    2. Saved, legacy, YAML, CSV, and explicit hyperparameters construct the architecture before state application under deterministic precedence and signature filtering.
    3. `configure_model`, `on_load_checkpoint`, strictness, mapped device placement, missing and unexpected keys, and payload mutation share one coherent model-loading contract.
    4. Trainer restoration now uses the same pre-state model and callback hook order and respects model-owned strictness while restoring complete training state.
    5. Checkpoint-directed external instantiators remain rejected because arbitrary import execution is not a core model-loading capability.
    6. CPU, MPS, unavailable-device, subprocess, spawn, fork, direct Gloo, stateful-continuation, malformed-input, compilation, and manual guideline evidence establish the phase.

| Strategic Note | Model Runtime State Contract | Date: 19 August 2026 | Time: 11:37 PM | Name: Ghost |

    1. ✅ Model freezing and unfreezing now preserve fluent inherited usage while applying standard gradient and recursive mode changes.
    2. Epoch, optimizer-step, and rank inspection now use attached Trainer ownership and neutral detached values rather than leaking ambient distributed process state.
    3. CUDA-specific `on_gpu` behavior is available without misclassifying MPS placement.
    4. Local, MPS, direct Gloo, subprocess, spawn, fork, compilation, and direct guideline evidence establish the corrected surface without static-analysis tools.

| Strategic Note | Model TorchScript Export Contract | Date: 19 August 2026 | Time: 11:47 PM | Name: Ghost |

    1. ✅ SyntheticMind now exposes the standard `to_torchscript` inheritance surface without Lightning branding or copied internal architecture.
    2. Scripting and tracing preserve exact nested model modes, remove runtime-only framework properties, route representative inputs through model or datamodule transfer ownership, and retain PyTorch keyword options.
    3. Positional tuples, documented keyword dictionaries, local paths, remote filesystems, nested modules, CPU, MPS, and failure restoration have executable evidence.
    4. Direct source and guideline review, Python compilation, signatures, inheritance, exception probes, artifact reload, and regressions establish the phase without static-analysis tools.

| Strategic Note | Model ONNX Export Contract | Date: 19 August 2026 | Time: 11:55 PM | Name: Ghost |

    1. ✅ SyntheticMind now exposes standard ONNX export through the installed modern dynamo engine and the explicit legacy engine without renaming either concept.
    2. Input ownership, keyword semantics, dynamic shapes, dynamic axes, evaluation behavior, exact nested mode restoration, return types, dependency diagnostics, and artifact destinations retain distinct executable meanings.
    3. `ONNXProgram`, local files, byte buffers, remote files, and external-data siblings persist through one SyntheticMind-owned boundary while PyTorch owns graph conversion.
    4. ONNX checking and ONNX Runtime inference on CPU and MPS-originated models establish usable artifacts rather than serialization-only claims.

| Strategic Note | Model TensorRT Compilation Contract | Date: 20 August 2026 | Time: 12:06 AM | Name: Ghost |

    1. ✅ SyntheticMind now exposes the pinned `to_tensorrt` inheritance surface without importing Lightning product identity or copying its internal architecture.
    2. Torch-TensorRT owns CUDA engine lowering; SyntheticMind owns dependency guards, CUDA target validation, input and transfer semantics, exact model restoration, IR dispatch, and destination handling.
    3. Local orchestration evidence covers dependency, validation, positional and named inputs, model and datamodule hooks, compile and persistence calls, local, buffer, and remote destinations, and restoration failures.
    4. Real TensorRT engine construction is not verified or claimed because the current Apple-silicon host and installed PyTorch runtime have no CUDA support.
    5. Direct source and guideline review, Python compilation, signatures, inheritance, exception probes, export regressions, and diff validation establish the locally reachable contract without linters or static type checkers.

| Strategic Hold | Controlled Restart Requirement | Date: 20 August 2026 | Time: 12:35 AM | Name: Ghost |

    1. Rahul stopped the recursive hardening run because the model-parallel implementation expanded too broadly and departed from the required `code-guidelines.md` discipline.
    2. The current code is preserved only as an interrupted review snapshot. It is not accepted model-parallel architecture and carries no completed-capability claim.
    3. The added future-annotations import, helper boundaries, generic Strategy methods, Trainer wiring, checkpoint decomposition, names, and inheritance effects must be challenged individually before reuse.
    4. A restart requires one Rahul-aligned slice at a time, full reading of the directly affected runtime path, minimal code movement, and focused executable verification before the next slice begins.
    5. Lightning remains a behavioral and terminology reference, while SyntheticMind architecture remains independently governed by `code-guidelines.md`.
    6. No implementation or parity work remains active until Rahul explicitly resumes it.

| Strategic Note | Controlled SyntheticMind Restart | Date: 04 September 2026 | Time: 05:26 AM | Name: Ghost |

    1. Rahul explicitly resumed the complete harness objective under the current root workspace `code-guidelines.md`; older repository-local guideline material remains historical and cannot override the current standard.
    2. The first accepted slice recovered model-parallel persistence ownership without reproducing the interrupted function collection or introducing a public abstraction family.
    3. Full and sharded two-rank training and resume now execute through `Module`, `Trainer`, `ModelParallelStrategy`, optimizer state, callback checkpointing, and Trainer progress restoration rather than through a standalone PyTorch primitive probe.
    4. The model call contract was corrected so default prediction passes through inherited `nn.Module.__call__` behavior and registered forward hooks.
    5. The next checkpoint slice must provide atomic sharded generation publication and failure recovery before model-parallel checkpoint durability can be marked complete.
    6. Subsequent work remains sequential: finish checkpoint durability, repair strategy ownership and dependency direction, then recompute missing core Trainer capability against pinned and current Lightning evidence.

| Strategic Note | Model-Parallel Checkpoint Durability | Date: 04 September 2026 | Time: 05:44 AM | Name: Ghost |

    1. Sharded checkpoint overwrite is now generation-based, so new shards and Trainer metadata remain invisible until one versioned pointer is atomically published.
    2. The active pointer retains the exact preceding published generation instead of guessing from directory modification times, preventing failed or orphaned writes from displacing the recovery generation.
    3. Legacy flat sharded directories remain readable and migrate on their next successful save without deleting the legacy generation before publication.
    4. Exact model-state comparison establishes trajectory equivalence for full and sharded resume, while injected distributed shard, metadata, and publication failures establish preservation of the active artifact.
    5. Checkpoint format collisions now fail visibly rather than converting a file into a directory or a directory into a file through destructive replacement.
    6. The next architectural target is the Strategy hierarchy and its dependency direction, especially the incorrect inheritance of model parallelism from DDP and the duplicated distributed-step contracts.

| Strategic Note | Strategy Boundary Reconstruction | Date: 04 September 2026 | Time: 07:05 AM | Name: Ghost |

    1. The strategy hierarchy now uses earned structural contracts instead of an ABC carrying hundreds of lines of shared mutable implementation.
    2. DDP and model parallelism are independent parallel strategies. Their common process-group resource, launcher capability, and forward-redirection algorithm are composed explicitly rather than represented through false DDP inheritance.
    3. Local and model-parallel checkpoints implement one restoration and persistence contract, while interchangeable storage plugins implement a separate `CheckpointIO` contract. The boundary prevents duplicated strategy aliases and retains distinct distributed-state semantics.
    4. Broad `object` annotations were removed from the complete strategy and launcher package and replaced with named batch, optimizer, backward, model-state, optimizer-state, stage-argument, and checkpoint-storage contracts.
    5. Process-group lifetime now follows the participating process rather than an individual Trainer stage, which preserves sequential distributed stages and live DTensor access while retaining exception and exit cleanup.
    6. The next dependency-ordered phase must audit accelerators, precision plugins, cluster environments, and layer synchronization because these collaborators still feed the reconstructed strategy boundary and retain historical abstraction debt.

| Strategic Note | Execution Collaborator Reconstruction | Date: 04 September 2026 | Time: 07:21 AM | Name: Ghost |

    1. Device, cluster-environment, and layer-synchronization extension points now use structural contracts without inherited placeholder behavior.
    2. Device selection has one invariant owner, and resolved topology is an immutable validated boundary rather than a mutable dataclass assembled by a free function.
    3. Precision retains one concrete shared lifecycle while recursive conversion now belongs to that lifecycle directly, avoiding a stateless converter object and duplicate conversion paths.
    4. Transformer Engine is isolated as an optional external adapter and follows the current NVIDIA `autocast` and delayed-scaling contract instead of the historical `fp8_autocast` call shape.
    5. Runtime evidence covers the local Apple MPS device, CPU true and mixed precision, distributed CPU execution, cluster launch reconstruction, reversible synchronized batch normalization, and the unavailable optional dependency boundary.
    6. The next phase is optimizer and training-closure ownership, where historical module functions, callable objects, opaque closure returns, and dynamically mutated optimizer classes still violate the governing architecture.

| Strategic Note | Shared Contract Direction | Date: 04 September 2026 | Time: 11:41 AM | Name: Ghost |

    1. Framework-wide type and exception contracts now live at the SyntheticMind package root rather than inside the transitional utility package.
    2. This placement lets every training subsystem name shared data and failure domains without introducing a dependency on an implementation graveyard or another lifecycle owner.
    3. The remaining utility modules must be reassigned by machine-learning concept: nested loader composition to the training-data lifecycle, model inspection to callback ownership, and distributed operations to Strategy or their actual consumers.
    4. Each reassignment remains bounded by executable process, checkpoint, and lifecycle evidence; moving filenames without correcting ownership is not progress.

| Strategic Note | Training Data Composition Boundary | Date: 04 September 2026 | Time: 11:53 AM | Name: Ghost |

    1. `CombinedLoader` is now an explicit SyntheticMind data-composition API rather than a utility implementation detail.
    2. Its package-root placement keeps model-facing DataModule construction inward from Trainer orchestration while allowing every stage loop to consume one canonical loader tree.
    3. Distinct combination algorithms remain private collaborators of the public owner, preserving recognized machine-learning semantics without exporting a class family.
    4. The next ownership target is distributed behavior still concealed in utilities; Strategy must absorb execution semantics while consumers retain only their real presentation or metric responsibilities.

| Strategic Note | Distributed Execution Authority | Date: 04 September 2026 | Time: 12:04 PM | Name: Ghost |

    1. Strategy is now the sole authority for rank topology and tensor collectives across local, DDP, and model-parallel execution.
    2. Metric owners declare the requested reduction but receive the selected Strategy's narrow reduction operation at computation time, preserving inward dependency direction.
    3. Callback and persistence gates use Trainer rank state instead of ambient process globals, while collective participation remains symmetric before rank-specific output decisions.
    4. The distributed utility namespace and aliases were removed; the remaining utility evacuation concerns hook inspection, module-mode restoration, and model-summary ownership.

| Strategic Note | Hook and Module State Ownership | Date: 04 September 2026 | Time: 12:11 PM | Name: Ghost |

    1. Exact nested training modes now belong to Module and remain restorable both by normal loop completion and by Trainer failure cleanup.
    2. Hook signature knowledge belongs to Module, while general override discovery belongs beside Trainer hook dispatch; narrow callback and model-parallel cases use direct contract comparison.
    3. The generic model-helper module, its state holder, and its free functions were deleted without exporting a substitute abstraction family.
    4. Model-summary computation and rendering are the final behaviors remaining under utilities and form the next concept-first ownership slice.

| Strategic Note | Model Inspection and Utility Elimination | Date: 04 September 2026 | Time: 12:23 PM | Name: Ghost |

    1. Model summary is now a callback capability family whose lifecycle, private inspection report, and optional Rich renderer have explicit ownership.
    2. Inspection measures actual parameter storage and executes example-input forwarding as a reversible observation transaction across model modes, hooks, and random generators.
    3. The generic utility package has been eliminated completely; every surviving behavior now belongs to an ML lifecycle, execution policy, core contract, callback, or infrastructure boundary.
    4. Callback topology is the next audit because its remaining flat package width crosses the generic review boundary and must be judged by selectable training concepts rather than file count alone.

| Strategic Note | Callback Capability Subtraction | Date: 04 September 2026 | Time: 12:28 PM | Name: Ghost |

    1. Three callback surfaces were removed because their names overstated their executable semantics or duplicated the canonical profiler subsystem.
    2. Deleting the runtime callback also removed its private timing state from Trainer and loops, leaving profiler action contexts as the sole timing authority.
    3. The remaining eleven direct callback modules are independently selectable training concepts and do not require cosmetic subpackages merely to reduce visual width.
    4. Callback work now shifts from topology to correctness, with weight averaging, learning-rate monitoring, numerical checks, and persistence lifecycle defects taking priority.

| Strategic Note | EMA Weight Averaging Contract | Date: 04 September 2026 | Time: 12:38 PM | Name: Ghost |

    1. EMA now follows optimizer-step progress, supports accumulation, validates its configuration and tensor topology, and persists through the standard callback state channel.
    2. Evaluation uses reversible average-weight swaps, failures restore live tensors, and final training state becomes averaged only after a real update.
    3. Checkpoints created during an average swap preserve live training state under the canonical schema for both full and weights-only artifacts.
    4. DDP and MPS paths are verified; sharded and DTensor paths are rejected until their average-state persistence can be represented correctly.

| Strategic Note | Preventive Numerical Failure Boundary | Date: 04 September 2026 | Time: 12:41 PM | Name: Ghost |

    1. Finite-value enforcement moved ahead of optimizer mutation: loss is checked before backward and gradients immediately after backward.
    2. Automatic, manual, accumulated, precision-mediated, and distributed paths share the same callback boundary instead of inspecting batch output after stepping.
    3. The renamed callback exposes only the real optional cost, gradient scanning, and reports positioned failures before state corruption.

| Strategic Note | Learning Rate Telemetry Contract | Date: 04 September 2026 | Time: 12:47 PM | Name: Ghost |

    1. Learning-rate telemetry now follows optimizer and scheduler lifecycle rather than batch or decorative configuration state.
    2. Multiple optimizers, repeated types, parameter groups, explicit names, momentum, weight decay, and dynamic fine-tuning groups retain distinct metric identities.
    3. Logger emission remains rank-zero-owned while callback metric views and resume coordinates remain coherent across workers.

| Strategic Note | Data-Axis Throughput Semantics | Date: 04 September 2026 | Time: 12:52 PM | Name: Ghost |

    1. Strategy now distinguishes data-parallel sample replication from total distributed world size, preventing tensor-parallel ranks from inflating throughput.
    2. Training rates include data wait and compute, exclude validation pauses, and synchronize accelerator work before measuring elapsed time.
    3. Accumulation windows count every microbatch while emitting on optimizer progress, with explicit custom cardinality for nonstandard batch structures.

| Strategic Note | Accelerator Statistics Contract | Date: 04 September 2026 | Time: 12:58 PM | Name: Ghost |

    1. Device-stat meaning now belongs to the selected Accelerator, while the callback owns only lifecycle cadence, filtering, metric routing, and rank-zero persistence.
    2. CPU, CUDA, and MPS expose distinct native counter families without pretending that CUDA globals represent every accelerator.
    3. Live MPS, source-shaped CUDA, CPU, filter, stage, and spawned DDP evidence establish the current monitoring boundary.

| Strategic Note | Early Stopping Consensus | Date: 04 September 2026 | Time: 01:03 PM | Name: Ghost |

    1. Early stopping now evaluates one broadcast scalar and one synchronized decision across every worker.
    2. Patience, target, divergence, and non-finite transitions retain explicit reasons and strict resumable state.
    3. Rank-zero-only monitor values, missing metrics, validation frequency, and checkpoint continuation no longer permit worker disagreement.

| Strategic Note | Wall-Clock Budget Semantics | Date: 04 September 2026 | Time: 01:10 PM | Name: Ghost |

    1. Timer separates continuous resumable budget time from active training progress time and latest stage durations.
    2. Rank-zero broadcast controls distributed stopping, and depleted restored state stops before new optimizer work.
    3. Validation pauses progress timing without hiding its wall-clock cost from the run budget.
