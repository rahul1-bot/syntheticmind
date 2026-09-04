| Strategic Direction | SyntheticMind Harness Hardening | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. SyntheticMind is Rahul's private machine-learning harness for FAU study, independent research, experimentation, and direct end-to-end control of machine-learning workflows.
    2. The harness is intentionally incomplete and must not be represented as production-ready, release-ready, or capability-complete.
    3. The active objective is full user-facing capability parity with core PyTorch Lightning while retaining an independent SyntheticMind architecture.
    4. Fabric, public documentation, release machinery, promotional work, and replication of the upstream PyTorch Lightning test suite are excluded.
    5. Code behavior, inherited extension compatibility, failure handling, and proportionate executable verification control the work queue.
    6. Existing source comments remain untouched, and new source-code comments remain prohibited until Rahul explicitly changes that constraint.

| Architecture Decision | Public Terminology and Framework Ownership | Date: 19 August 2026 | Time: 11:29 PM | Name: Ghost |

    1. Established machine-learning and PyTorch terminology is a public compatibility boundary. Standard concepts retain their recognized names, parameter names, defaults, meanings, and lifecycle effects.
    2. Public base classes are inheritance contracts. Method-resolution order, cooperative initialization, overridable names, signatures, invocation frequency, return meanings, and state transitions require executable preservation.
    3. PyTorch Lightning is the pinned capability and behavior reference, not a namespace, product identity, implementation template, or source of vendor branding for SyntheticMind.
    4. New public terms are prohibited when they merely rename an established concept. Project-specific terminology requires genuinely different semantics enforced by implementation and verification.
    5. `LightningOptimizer` is vendor-owned wrapper terminology rather than an optimizer algorithm. SyntheticMind exposes `SyntheticOptimizer` for its managed adapter while the underlying algorithm remains `torch.optim.Optimizer`.
    6. Callback parameters use the SyntheticMind-owned name `module`; Lightning-branded `pl_module`, `use_pl_optimizer`, and equivalent copied identifiers remain prohibited.
    7. Internal architecture follows `/Users/rahulsawhney/LocalCode/AI/syntheticmind/docs/references/guidelines/code-guidelines.md` without blindly reproducing Lightning internals.

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
