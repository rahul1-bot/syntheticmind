| Strategic Note | SyntheticMind Development Intent | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. SyntheticMind is private infrastructure for Rahul's study, research, experimentation, and direct ownership of complete machine-learning workflows.
    2. The target is full user-facing core PyTorch Lightning capability parity without becoming a Lightning clone or importing its product identity.
    3. Fabric, public documentation, release machinery, promotional work, and upstream test-suite replication remain outside the active objective.
    4. The harness is incomplete. Capability claims require implemented runtime behavior and focused evidence rather than declared parameters or surface resemblance.
    5. Existing comments remain untouched, new source comments remain prohibited, and code remains the sole active deliverable.

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
