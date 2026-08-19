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

| Active Next Step | Standard TorchScript Export | Date: 19 August 2026 | Time: 11:37 PM | Name: Ghost |

    1. Audit the pinned `to_torchscript` signature, method selection, example-input precedence, mode preservation, file output, and inherited override behavior.
    2. Keep ONNX and TensorRT exports independent because they have separate dependency, serialization, and accelerator contracts.
    3. Keep `device_mesh` deferred until a model-parallel Strategy owns and initializes that state.
    4. Preserve all scope, terminology, guideline, evidence, and branch-synchronization policies.
