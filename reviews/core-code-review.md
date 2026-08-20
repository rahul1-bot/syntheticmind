# SyntheticMind `core/` Code Review

## Review baseline

- SyntheticMind: `dev@a377421801ee65c5c5bd710255bfdf399cbd6422`
- Comparative PyTorch Lightning reference: `fcef404516264798b6c582ef556eaa17e5c29bfb`
- Scope: Python code under `core/` only
- This review does not modify Python source.

## Review precedence

The target is user-capability parity with core PyTorch Lightning, not a line-for-line clone.

1. Standard ML/PyTorch/Lightning public contracts take precedence over generic application-style rules when the two conflict. Established method names, parameter names, defaults, hook names, return meanings, and inheritance behavior must remain recognizable and compatible.
2. SyntheticMind owns its internal implementation. Private services, state containers, validation, orchestration, and decomposition should follow `code-guidelines.md` rather than copy Lightning internals.
3. Public framework inheritance is intentional. `Module`, `DataModule`, hook families, and other extension surfaces are not candidates for Protocol replacement merely because composition is preferred internally.
4. Existing source comments are outside this review's mutation scope. Future implementation should leave them untouched and add no new source comments.
5. Fabric, public documentation, release machinery, and upstream test-suite replication are excluded.

## Priority findings

### P0 — `DataModule` does not preserve the standard data-hook inheritance contract

**File:** `core/datamodule.py`

Current declaration:

```text
DataModule(HyperparametersMixin)
```

The pinned Lightning contract is structurally:

```text
LightningDataModule(DataHooks, HyperparametersMixin)
```

SyntheticMind already has `DataHooks`, but `DataModule` re-declares `prepare_data`, `setup`, `teardown`, all four dataloader methods, and the batch-transfer hook trio instead of inheriting the existing contract.

This is not only duplication. It changes the user-facing inheritance contract:

- `isinstance(datamodule, DataHooks)` is false.
- `prepare_data_per_node` is not inherited.
- `allow_zero_length_dataloader_with_multiple_devices` is not inherited.
- cooperative `super()` behavior differs from the standard extension surface.
- the batch-transfer signatures are independently maintained and can drift from `DataHooks`.
- `DataModule` and `Module` no longer share one authoritative data-hook definition.

**Required direction:** make `DataModule` inherit `DataHooks` and `HyperparametersMixin` in the standard relative order. Remove duplicated hook implementations unless `DataModule` genuinely needs different behavior. Any retained concrete override must use `@override`.

This should be fixed before more datamodule or distributed-data work.

---

### P0 — `DataModule.load_from_checkpoint` capability is missing

**Files:** `core/datamodule.py`, `core/model_checkpoint_loader.py`, `core/hyperparameters.py`

The pinned `LightningDataModule` exposes class-level checkpoint reconstruction with the standard `checkpoint_path`, `map_location`, `hparams_file`, `weights_only`, and constructor-override semantics. SyntheticMind implements a mature `Module.load_from_checkpoint` path but exposes no corresponding `DataModule.load_from_checkpoint`.

That leaves an asymmetry in a core user capability: model subclasses can reconstruct themselves from harness checkpoints while datamodule subclasses cannot.

**Required direction:** extract/generalize the already implemented checkpoint payload and hyperparameter reconstruction machinery so both model and datamodule reconstruction use one owned service. Preserve the standard `DataModule.load_from_checkpoint` public name, parameter meanings, defaults, and subclass-construction behavior. Do not create a second unrelated checkpoint loader.

---

### P1 — `Trainer` exposes stale/custom constructor parameters outside the pinned public contract

**File:** `core/trainer.py`

The constructor adds:

- `resume_from_checkpoint`
- `seed`

Neither exists in the pinned core `Trainer` constructor.

`resume_from_checkpoint` is particularly harmful because `Trainer.fit(..., ckpt_path=...)` already exposes the modern stage-level resume contract. SyntheticMind currently gives the same concept two public control paths and internally falls back from `ckpt_path` to `resume_from_checkpoint`.

`seed` also moves run-global reproducibility behavior into a constructor surface where the pinned Trainer does not own it.

**Required direction:**

- make `ckpt_path` the authoritative training-resume control at the standard stage entry points;
- move seeding behind the standard explicit seeding surface rather than a SyntheticMind-only Trainer constructor argument;
- remove compatibility aliases once callers have migrated instead of carrying parallel semantics indefinitely.

Missing integration/product parameters such as model-registry or integration-suggestion controls are not findings here because they fall outside the stated private-harness scope.

---

### P1 — `Module` MRO differs from the public inheritance contract and needs an explicit parity decision

**File:** `core/module.py`

SyntheticMind currently declares:

```text
Module(ModelHooks, DataHooks, CheckpointHooks, HyperparametersMixin, nn.Module)
```

The pinned Lightning relative order places hyperparameter behavior before the model/data/checkpoint hook families and then the PyTorch module base.

SyntheticMind does not need to copy Lightning's private device/dtype mixin. It does, however, need to treat the relative order of public extension mixins as API because cooperative `super()`, subclass mixins, and method-resolution behavior are user-observable.

**Required direction:** audit and align the public relative MRO while keeping SyntheticMind-owned device/dtype implementation internal. Add no private compatibility base merely to resemble Lightning. The result should preserve cooperative initialization and inherited hook behavior under subclass extension.

---

### P1 — `Trainer` has become a monolithic runtime owner that conflicts with sequential hardening

**File:** `core/trainer.py`

`Trainer` currently owns, in one class:

- public configuration validation and normalization;
- accelerator/strategy/plugin resolution;
- callback/logger/default-component construction;
- loop construction;
- fit/validate/test/predict orchestration;
- model/datamodule attachment;
- dataloader ownership and distributed sampler replacement;
- checkpoint path selection, preload, restore, and save coordination;
- metric routing state;
- profiling and console coordination;
- exception handling and teardown;
- launcher reconciliation and stage-local reset state.

The class is far beyond the guideline's single-responsibility and method-size boundaries. More importantly, this is now a correctness problem: a change to one capability regularly touches unrelated lifecycle state, which is exactly what made the interrupted model-parallel phase expand across the runtime.

**Required direction:** keep `Trainer` as the stable public facade and exact standard entry-point surface, but move private ownership into small composed services. Decompose by invariants, not by copying Lightning's connector directory layout. Good boundaries include configuration normalization, data attachment/loading, checkpoint coordination, hook/callback dispatch, and stage execution/teardown only where each boundary has a clear caller set and state owner.

Do this incrementally from foundational contracts upward.

---

### P1 — `Module` is also carrying too many independent runtime responsibilities

**File:** `core/module.py`

`Module` currently combines:

- the public model/hook inheritance surface;
- metric logging validation and buffering;
- TorchMetrics type discovery;
- manual-optimization access;
- optimizer/scheduler accessors;
- gradient clipping and optimizer toggling;
- device and dtype mirror management;
- trainer/runtime-state access;
- distributed gathering;
- checkpoint reconstruction entry;
- TorchScript, ONNX, and TensorRT entry points;
- logged-metric runtime record definitions.

The public methods must remain on `Module` because users subclass it. Their implementation does not need to live in one class.

**Required direction:** preserve the public facade and method signatures while delegating private mechanisms to composed helpers. Export/checkpoint work has already started moving in that direction; logging/runtime-state/optimization plumbing should follow. The decomposition must reduce circular dependencies rather than introduce more helper layers with vague ownership.

---

### P1 — `optimizer.py` contains two competing optimization models and one should be authoritative

**File:** `core/optimizer.py`

The active `Module.configure_optimizers()` and `Trainer` path follows the established Lightning/PyTorch user contract: users return optimizers, optimizer sequences, optimizer/scheduler pairs, or mapping-based scheduler configuration.

The same file still carries the older SyntheticMind factory layer:

- `_OptimizerFactoryConfig`
- `_SchedulerFactoryConfig`
- `_OptimizationConfiguration`
- `_OptimizerDefinition`
- `_SchedulerDefinition`

These are private, are not part of the current exported `configure_optimizers()` contract, and are not used by the inspected `Module`/`Trainer` execution path. Keeping both models leaves two vocabularies for the same concept and makes future maintenance ambiguous.

**Required direction:** confirm there are no remaining private call sites and remove the obsolete factory configuration layer. One optimization contract should exist.

Separately, `LRSchedulerConfig` is currently a mutable `@dataclass`. It is an owned internal value object with invariants that are validated later by helper functions. Under the project guidelines this should be a frozen validated model so invalid scheduler configuration cannot exist between construction and later validation. Preserve the established field names and defaults; change the internal representation, not the user-facing scheduler dictionary vocabulary.

`OptimizerConfig`, `OptimizerLRSchedulerConfig`, and related `TypedDict` shapes are not violations: they model externally supplied dictionary contracts from `configure_optimizers()` and are appropriate interop types.

---

### P2 — Python 3.14 rules are violated by `Module` type-loading and mutable module state

**File:** `core/module.py`

`Module` contains:

- `from __future__ import annotations`;
- `TYPE_CHECKING`-guarded type imports;
- module-level mutable caches `_metric_type` and `_metric_collection_type`;
- a helper that mutates those globals with `global`.

This directly conflicts with the Python 3.14 and no-global-mutable-state rules.

The global TorchMetrics cache is also architectural coupling: import state becomes hidden process-wide state owned by no object.

**Required direction:** remove the future-annotations compatibility import, remove mutable module caches, and resolve type/import cycles through cleaner dependency boundaries. If lazy optional import remains necessary, let a class-owned collaborator or a direct local import own it without mutable module state.

`core/datamodule.py` also imports `TYPE_CHECKING`; correct the same Python-3.14 pattern when the inheritance fix is made.

---

### P2 — override markers are missing on concrete datamodule overrides

**File:** `core/datamodule.py`

`DatasetDataModule` overrides `train_dataloader`, `val_dataloader`, `test_dataloader`, `predict_dataloader`, and `_configured_loader_stages` without `@override`.

This matters more in a framework than in ordinary application code because these names are extension contracts. A base-class rename or signature change should fail loudly.

**Required direction:** add `@override` to genuine overrides. After `DataModule` inherits `DataHooks`, either remove redundant base hook redefinitions or mark any intentionally retained redefinitions explicitly.

---

### P2 — assignment-site typing is inconsistent in mutable runtime state

**Files:** `core/module.py`, `core/trainer.py`

The guidelines require the complete attribute type to be restated at every mutation site. The code follows this in many places but not consistently.

Representative examples include unannotated mutations of:

- `Module._dtype` in `double()` / `bfloat16()`;
- `Module._trainer` in the trainer setter;
- `Module._current_dataloader_idx` during logging reset;
- multiple `Trainer` run-reset fields during stage entry and teardown.

Because these classes mutate substantial state across long lifecycle methods, inconsistent mutation typing defeats the exact area where the project wants mutation contracts to be most visible.

**Required direction:** fix this after the large classes are decomposed. Performing thousands of annotation-only edits inside the current monolith first would create review noise without reducing architectural risk.

---

### P2 — internal helpers return anonymous multi-value tuples where named validated results are clearer

**Files:** `core/trainer.py`, `core/optimizer.py`, `core/model_tensorrt.py`

Examples include:

- `_resolve_training_limits()`;
- `_init_optimizers_and_lr_schedulers()`;
- `_configure_optimizers()`;
- optimizer/scheduler normalization hand-offs;
- `TensorRTInputPreparation.prepare()` returning positional and keyword input sets as a pair.

These are conceptual result records, not ordinary tensor shapes or positional model inputs. Anonymous tuple position is easy to swap and spreads index/unpack knowledge across callers.

**Required direction:** use small frozen named result models for true multi-field service results. Do not mechanically replace tuples whose domain meaning is genuinely positional, such as positional arguments passed to a model.

---

## Intentional exceptions and non-findings

These are deliberately **not** findings:

- Standard public names such as `training_step`, `validation_step`, `configure_optimizers`, `optimizer_step`, `manual_backward`, `ckpt_path`, `prog_bar`, `sync_dist`, `dataloader_idx`, and the Trainer parameter vocabulary must not be renamed to satisfy generic naming preferences.
- Public hook/base-class inheritance must remain inheritance. Protocols are useful for private collaborator contracts but must not be inserted into public MRO solely to satisfy a generic composition rule.
- `SyntheticOptimizer` intentionally mirrors the managed optimizer-wrapper capability while avoiding Lightning-owned branding. Its dynamic optimizer-class compatibility behavior matches the reference concept and is not an invented-name defect.
- `Module.print()` is an established rank-aware framework capability; the generic production `print()` rule does not justify removing or renaming that public method.
- User-supplied `configure_optimizers()` mapping shapes are correctly represented with `TypedDict`-style interop contracts rather than forced into a SyntheticMind-only Pydantic API.
- `str | Path`-style path boundaries are retained where standard checkpoint/export behavior includes remote filesystem URIs; the generic local-`Path` rule must not destroy remote-path capability.
- Existing source comments are not reviewed as cleanup work and must remain untouched.
- No finding requests replication of Lightning's upstream test suite. Focused executable probes and assertions remain sufficient for this private research harness under the current project policy.

## File-by-file coverage

| File | Review result |
|---|---|
| `core/__init__.py` | No blocking code finding. |
| `core/datamodule.py` | P0 inheritance contract; P0 missing checkpoint reconstruction; P2 override markers; Python-3.14 type-loading cleanup. |
| `core/hooks.py` | Public hook vocabulary is structurally appropriate; inheritance is intentional. No blocking finding independent of `DataModule` integration. |
| `core/hyperparameters.py` | Core semantics broadly match the standard mutable `hparams` / copied `hparams_initial` behavior. No critical independent finding. |
| `core/model_checkpoint_loader.py` | Good separation of payload I/O, constructor reconstruction, and state application. Reuse/generalization target for `DataModule.load_from_checkpoint`. |
| `core/model_export.py` | Responsibilities are already comparatively well composed. No critical finding. |
| `core/model_tensorrt.py` | Main structural finding is the anonymous multi-value preparation result; otherwise responsibilities are separated cleanly. |
| `core/module.py` | P1 MRO contract audit; P1 responsibility overload; P2 Python-3.14/global-state violation; P2 mutation typing. |
| `core/optimizer.py` | P1 dual optimization models; scheduler config value-object issue; P2 anonymous multi-return helpers. |
| `core/protocols.py` | Internal structural protocols are useful. Do not force public classes to inherit them when doing so would alter public MRO. |
| `core/saving.py` | Hook names and role are appropriate. No blocking independent finding. |
| `core/trainer.py` | P1 public constructor drift; P1 responsibility overload; P2 mutation typing and anonymous multi-result helpers. |

`core/README.md` is not included because this review is code-only and public-documentation work is explicitly excluded.

## Recommended hardening order

1. **Repair `DataModule` public inheritance and reconstruction contracts.**
2. **Resolve `Module` public MRO and Python-3.14/global-state issues without adding new features.**
3. **Collapse `optimizer.py` to one authoritative standard optimization contract.**
4. **Extract cohesive private services from `Module` while preserving every public method/hook signature.**
5. **Extract cohesive private services from `Trainer`, then remove stale constructor aliases only with migration coverage.**
6. **Apply the systematic assignment-site and named-result cleanup inside the smaller components.**
7. **Only then resume or re-audit the interrupted model-parallel work.**

The core objective is not fewer lines for its own sake. It is to make every standard ML contract obvious at the public boundary and every internal invariant owned by one small component, so future capability parity can be added sequentially without cross-cutting surgery.
