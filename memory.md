| Memory | SyntheticMind Mission and Boundaries | Date: 19 August 2026 | Time: 04:31 AM | Name: Ghost |

    1. SyntheticMind is personal infrastructure rather than a public framework product. Its value is direct control over machine-learning execution for FAU work, research, experiments, and creative enjoyment.
    2. The codebase was built from the ground up so training behavior can be inspected, changed, and reasoned about end to end.
    3. The present objective is reliability, not breadth for its own sake. A smaller harness with correct lifecycle and optimization semantics is superior to a large interface containing ceremonial or misleading parameters.
    4. The harness remains incomplete, and future assessments must distinguish implemented behavior from intended behavior or README claims.
    5. Documentation and release work remain subordinate to executable correctness, focused tests, and capabilities required by actual workloads.

| Memory | Terminology Compatibility Rule | Date: 19 August 2026 | Time: 04:31 AM | Name: Ghost |

    1. Machine-learning users carry established expectations for terms such as `model`, `max_steps`, `val_check_interval`, `ckpt_path`, `devices`, `precision`, and `global_step`.
    2. Reusing a standard term with different semantics is as dangerous as inventing a replacement term because both create silent operational confusion.
    3. Trainer comparisons must therefore inspect parameter names, accepted values, defaults, validation, lifecycle effects, and checkpoint behavior rather than comparing signatures mechanically.
    4. SyntheticMind should adopt only the Lightning concepts that strengthen its private mission. Framework-specific compatibility baggage and public-product surface area must not be copied without a concrete workload.
    5. The generic Python guidelines at `/Users/rahulsawhney/LocalCode/AI/syntheticmind/docs/references/guidelines/code-guidelines.md` remain the implementation baseline for typing, boundaries, validation, composition, and maintainability.

| Memory | Trainer Audit Evidence | Date: 19 August 2026 | Time: 04:48 AM | Name: Ghost |

    1. The controlling Lightning comparison revision is `fcef404516264798b6c582ef556eaa17e5c29bfb`; comparison claims must not be attributed to another checkout or release.
    2. SyntheticMind currently defines `global_step` as an optimizer-step count in state documentation but increments it once per manual training batch, irrespective of how many optimizer steps actually execute.
    3. Mid-fit validation changes the Trainer running stage to validation and restores only the module stage, causing later training callbacks to observe stale state.
    4. Global-step validation cadence can trigger repeatedly on accumulation microbatches because the validation predicate does not prove that a new optimizer step occurred.
    5. Unsized iterable datasets can finish inside a distributed `no_sync()` accumulation window; the subsequent tail optimizer step uses rank-local gradients and produces divergent model parameters.
    6. The distributed dataloader reconstruction path does not preserve custom batch samplers, loader subclasses, or every current PyTorch DataLoader constructor field.
    7. Several parameter names collide with established Lightning terminology while implementing different values or edge-case semantics. Hardening must correct existing semantics before adding more parameters.
    8. The preferred boundary is a small Trainer facade backed by immutable validated configuration and focused collaborators. Lightning is a semantic reference, not an architectural template to copy wholesale.
