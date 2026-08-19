| Strategic Note | SyntheticMind Development Intent | Date: 19 August 2026 | Time: 04:31 AM | Name: Ghost |

    1. SyntheticMind is maintained as Rahul's private machine-learning control layer for study, research, experimentation, and creative enjoyment.
    2. The architecture is valuable because it exposes the complete workflow rather than hiding consequential behavior behind a framework boundary.
    3. That control is useful only when the behavior is correct. The current implementation is incomplete and must be hardened sequentially while the missing capability surface is implemented and verified.
    4. Review decisions must favor deterministic execution, explicit state transitions, recoverable checkpoints, correct optimization accounting, and truthful interfaces while progressing toward full user-facing PyTorch Lightning capability parity.
    5. The public API must speak established machine-learning language. Internally invented aliases for standard concepts are prohibited unless a real semantic distinction requires them.

| Strategic Note | Working Branch Discipline | Date: 19 August 2026 | Time: 04:31 AM | Name: Ghost |

    1. The stable baseline and all root ledgers are retained on `main` and `origin/main`.
    2. Active implementation work occurs on `dev` and is published to `origin/dev` after every completed development commit.
    3. A commit is not operationally complete while local `dev` and `origin/dev` differ.
    4. Promotion from `dev` to `main` requires an explicit stabilization decision supported by relevant verification.
    5. The repository remains private because it is personal research and study infrastructure, not a public framework release.

| Strategic Note | Capability Parity and Ordered Hardening | Date: 19 August 2026 | Time: 05:19 AM | Name: Ghost |

    1. SyntheticMind's end state is full user-facing capability parity with core PyTorch Lightning, with Fabric, public documentation, release work, and replication of the upstream PyTorch Lightning test suite excluded.
    2. Established machine-learning and PyTorch variable names, method names, parameter contracts, state meanings, and inherited extension hooks are compatibility requirements rather than cosmetic preferences. Lightning-specific branding is reference ownership, not an industry convention.
    3. SyntheticMind will preserve its own internal architecture and apply the generic Python guidelines while matching the external semantics that machine-learning practitioners already understand.
    4. Exceptions require genuinely different semantics or algorithms and an explicit rationale; private terminology must never disguise an established concept.
    5. Hardening proceeds sequentially from source-backed compatibility contracts through foundational runtime layers. No isolated feature will be inserted into the middle of an unmapped execution path.
    6. Existing code comments remain in place, and new source-code comments are deferred during the current phase.
    7. Each capability requires focused local verification even though upstream test-suite replication is excluded; an unexercised interface is not accepted as implemented.

| Strategic Note | Research Persistence and Operational Memory | Date: 19 August 2026 | Time: 08:21 AM | Name: Ghost |

    1. Unknown implementation details must be resolved through source inspection, executable probes, installed-environment checks, current primary sources, and repository history before accepting a compromise.
    2. Carefully selected external repositories may be cloned into the ignored `external/` tree when they answer a concrete engineering question that the pinned PyTorch Lightning reference does not resolve.
    3. The ledgers preserve current direction and reusable knowledge only. Git history preserves chronological implementation detail, and duplication between those systems is prohibited.
    4. Every completed phase must leave local `dev` and `origin/dev` synchronized after verification and review.

| Strategic Note | Extension Contracts and Verification Boundary | Date: 19 August 2026 | Time: 09:03 AM | Name: Ghost |

    1. SyntheticMind's base classes must preserve standard machine-learning terminology, inheritance chains, method-resolution behavior, override names, signatures, parameter meanings, defaults, and lifecycle effects because downstream models and integrations extend these contracts directly.
    2. `code-guidelines.md` governs every implementation decision, while PyTorch and PyTorch Lightning conventions govern public concepts that users already understand.
    3. Ruff and other linters are excluded. Verification proceeds through focused executable behavior, signature and inheritance inspection, source comparison, failure-path probes, and compilation.

| Strategic Note | Optimization Foundation | Date: 19 August 2026 | Time: 11:37 AM | Name: Ghost |

    1. The optimization boundary must preserve meaningful optimizer access, scheduler access, manual backward, optimizer stepping, zeroing, clipping, and scheduler stepping contracts without importing Lightning's product identity.
    2. The accepted `configure_optimizers` structures are normalized behind a private parser, allowing SyntheticMind to retain a smaller internal architecture without changing the public machine-learning contract.
    3. Closure execution, manual multi-optimizer behavior, parameter toggling, scheduler cadence, mixed-precision ordering, checkpoint restoration, and optimizer-state device placement are executable capabilities rather than declared placeholders.
    4. `LightningOptimizer` is a Lightning-owned managed wrapper rather than an industry optimizer term. SyntheticMind's equivalent public wrapper is `SyntheticOptimizer`, with the actual algorithm remaining a `torch.optim.Optimizer`.
    5. ✅ The ownership correction removed `LightningOptimizer`, `use_pl_optimizer`, `_lightning_optimizers`, `LRSchedulerPLType`, and `pl_module`. The resulting `SyntheticOptimizer`, explicit raw-access flag, generic scheduler union, and `module` callback parameter were reverified across extension and execution paths.
    6. Existing comments remain untouched, new code comments remain prohibited, and no linter may substitute for direct guideline review and behavioral evidence.

| Strategic Note | Console Experience Boundary | Date: 19 August 2026 | Time: 11:28 AM | Name: Ghost |

    1. SyntheticMind's terminal experience should operate as a coherent live run console rather than independent progress bars interrupted by raw logger sentences.
    2. Visual polish is subordinate to measurement truth. Independent stage timers, sanity exclusion, stage-isolated progress metrics, boundary resets, and prediction batch totals now provide the verified substrate required by the renderer.
    3. Existing runtime telemetry should feed the console through structured events so terminal rendering, CSV and TensorBoard persistence, callbacks, and distributed execution remain separate concerns.
    4. The approved renderer uses one active TTY line, one completed-epoch line, and one final stage line. It shows only stage, progress, at most two selected metrics, and honest ETA by default; all detailed telemetry stays in structured experiment outputs.
    5. Non-interactive output remains stable plain text, distributed rendering remains rank-zero only, and unknown dataloader lengths produce counts rather than false percentages or ETA values.
    6. Callback monitoring metrics, experiment logging, profiling telemetry, and terminal-selected metrics remain separate channels. The renderer consumes runtime truth but does not become another authority for it.
    7. ✅ One renderer now covers fit, validation, testing, and prediction; routine framework information no longer competes with the active line, and completed epochs are deferred across failure-prone hook boundaries before becoming permanent output.
    8. ✅ Trainer cleanup is now one exception-safe transaction. Strategy, datamodule, ordinary callbacks, module, and logger complete their cleanup attempts before the progress callback is permitted to declare a successful stage outcome.
    9. Runtime, setup, interruption, exit, exception-hook, and teardown failures preserve the first causal exception while retaining secondary failures as notes; module setup and teardown are now dispatched for every public stage and all stage markers clear after failure.

| Strategic Note | Standard Training Limits | Date: 19 August 2026 | Time: 12:41 PM | Name: Ghost |

    1. ✅ SyntheticMind now distinguishes finite epoch budgets, finite optimizer-step budgets, and explicit unlimited horizons using the standard `max_epochs` and `max_steps` meanings.
    2. Minimum epoch and step requirements constrain only cooperative stopping. EarlyStopping may request termination immediately, but the fit continues until both minima are satisfied unless a hard maximum is reached first.
    3. Automatic optimization, accumulation, manual managed optimizers, LBFGS closures, schedulers, and checkpoint continuation share one authoritative optimizer-step counter and termination boundary.
    4. `Trainer.should_stop` is the public stop contract; loop state, EarlyStopping, and downstream callbacks no longer require a project-specific state access path.
    5. Unlimited execution is represented honestly in step estimates and terminal epoch labels, while checkpoint operations remain quiet and atomic-save interruption leaves no temporary artifact.
    6. The next parameter correction is validation scheduling: replace project-specific scope and end-validation controls with the established `val_check_interval` and `check_val_every_n_epoch` contract before expanding further.

| Strategic Note | Standard Validation Cadence | Date: 19 August 2026 | Time: 12:55 PM | Name: Ghost |

    1. ✅ Validation cadence is now expressed exclusively through the standard `val_check_interval` and `check_val_every_n_epoch` contract rather than SyntheticMind-specific scope and epoch-end switches.
    2. Batch cadence follows raw training progress, float cadence follows the effective epoch horizon, and duration cadence follows monotonic elapsed time. Optimizer accumulation cannot distort any of these meanings.
    3. Epoch eligibility, sanity checks, cooperative stopping, unsized dataloaders, dataloader reloads, and disabled validation share one normalized scheduling authority.
    4. Cumulative raw-batch checkpoint state preserves cross-epoch integer cadence during exact mid-epoch restoration, while final-batch checkpoints remain true epoch boundaries.
    5. The next sequential boundary is the standard `limit_train_batches`, `limit_val_batches`, `limit_test_batches`, `limit_predict_batches`, and sanity-step semantics on which later Trainer capabilities depend.

| Strategic Note | Standard Data-Pass Limits | Date: 19 August 2026 | Time: 01:02 PM | Name: Ghost |

    1. ✅ Every stage now resolves its batch budget through one standard contract owned by the Trainer rather than a separate loop-local interpretation.
    2. Integer counts, fractional counts, full-loader values, zero disabling, empty loaders, unsized loaders, and impossible sub-batch fractions now retain distinct and verified meanings.
    3. Sanity validation defaults to two batches, clips to the real validation budget, supports all-batch execution through `-1`, and never replays during checkpoint continuation.
    4. Effective training, validation, sanity, testing, and prediction batch counts are available through the established public Trainer inspection properties.
    5. The next foundational termination capability is the standard `max_time` contract, which must stop safely at batch boundaries without confusing hard elapsed-time termination with cooperative stopping.
