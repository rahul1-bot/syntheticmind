# syntheticmind.core: Model-Facing Contracts

`core/` defines the contracts client models inherit and the internal value owners that make those contracts executable. It does not own Trainer orchestration, data-source lifecycle, distributed process policy, or durable checkpoint transport.

| Module | Contents |
|---|---|
| `module.py` | `Module`, the model contract and client runtime surface. |
| `hooks.py` | `ModelHooks`, `DataHooks`, and `CheckpointHooks`, including logging admissibility by lifecycle position. |
| `optimizer.py` | Native optimizer and scheduler declaration forms, `LRSchedulerConfig`, and managed `SyntheticOptimizer`. |
| `metric.py` | `LoggedMetric`, the routed scalar or TorchMetrics observation produced by `Module.log`. |
| `hyperparameters.py` | `HyperparametersMixin`, constructor capture for reconstruction and experiment metadata. |
| `saving.py` | Model and DataModule checkpoint reconstruction from schema-v2 artifacts. |
| `activation_checkpointing.py` | Client-selected non-reentrant and selective activation recomputation. |
| `context_parallel_attention.py` | Local-query attention, differentiable packed global K/V, and rank-offset causal masking. |

## Module integration

Client models subclass `syntheticmind.Module`, which already inherits `torch.nn.Module`. A trainable model implements `forward`, `training_step`, and `configure_optimizers`. Validation, test, and prediction steps are added for the stages the project uses.

```python
from typing import override

import torch
from torch import nn

from syntheticmind import Module
from syntheticmind.primitives.types import Batch, TrainingStepOutput


class Regressor(Module):
    def __init__(self, input_width: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.network: nn.Linear = nn.Linear(input_width, 1)

    @override
    def forward(self, inputs: Batch) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("Regressor expects tensor inputs.")
        return self.network(inputs)

    @override
    def training_step(
        self,
        batch: Batch,
        batch_idx: int
    ) -> TrainingStepOutput:
        del batch_idx
        if not isinstance(batch, list | tuple) or len(batch) != 2:
            raise TypeError("Regressor expects an input-target pair.")
        inputs, targets = batch
        loss: torch.Tensor = torch.nn.functional.mse_loss(
            self(inputs),
            targets
        )
        self.log("train_loss", loss, on_epoch=True)
        return {"loss": loss}

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
```

Automatic optimization owns zeroing, forward, loss extraction, backward, accumulation, gradient synchronization, clipping, optimizer stepping, scheduler cadence, and progress advancement. `global_step` counts completed managed optimizer steps, not batches.

## Metric production

`Module.log` accepts a Python numeric scalar, a single-element tensor, or a registered `torchmetrics.Metric`. It records an observation; loops own when observations are computed and accumulated, the logger connector owns consumer views and rank policy, and loggers own persistence.

Training-step hooks default to step publication. Validation and test step hooks default to epoch publication. Epoch hooks permit epoch publication only. Prediction, transfer, configuration, checkpoint, and teardown hooks reject logging because they have no coherent metric cadence.

Scalar epoch means synchronize numerator and denominator across data-parallel ranks. Stateful TorchMetrics objects retain their own compute and synchronization semantics. `enable_graph=True` preserves graph state deliberately but cannot cross a resumable epoch checkpoint. A custom process group is likewise rejected at a partial-epoch persistence boundary.

`log_dict` applies one option set to every mapping or `MetricCollection` entry. Multiple dataloaders append `/dataloader_idx_N` when `add_dataloader_idx=True`, preventing stage-local observations from overwriting one another.

## Manual optimization

Set `automatic_optimization=False` when the model must choose optimizer order, objectives, or update frequency. `Module.optimizers()` returns managed `SyntheticOptimizer` instances by default.

```python
self.automatic_optimization = False
generator, discriminator = self.optimizers()

with discriminator.toggle_model():
    discriminator.zero_grad()
    discriminator_loss: torch.Tensor = self.discriminator_loss(batch)
    self.manual_backward(discriminator_loss)
    discriminator.step()

with generator.toggle_model():
    generator.zero_grad()
    generator_loss: torch.Tensor = self.generator_loss(batch)
    self.manual_backward(generator_loss)
    generator.step()
```

Managed optimizers retain native parameter groups and state dictionaries while routing steps through precision, Strategy, hooks, and progress ownership. Pass `use_synthetic_optimizer=False` only when the client intentionally needs the raw `torch.optim.Optimizer` and accepts responsibility for bypassed managed behavior.

Multiple optimizers are prohibited under automatic optimization. Scheduler mappings normalize to immutable `LRSchedulerConfig` values. Step schedulers follow real optimizer progress; epoch schedulers advance only after a true loader epoch completes; plateau schedulers run only after their monitor exists.

## Hooks and model construction

Hook defaults are no-ops or identity operations. A client overrides only the transitions it needs and marks genuine replacements with `@override`. `configure_model` is the construction hook for layers that depend on Strategy initialization context, a device mesh, precision-directed dtype, empty initialization, tensor parallelism, or composable FSDP2. Ordinary models may construct all layers in `__init__`.

`DataHooks` are available on Module and DataModule. When both override a transfer operation, DataModule takes precedence. Checkpoint hooks run around payload construction and restoration; client state must be namespaced and may not delete framework trajectory fields.

## Hyperparameters and reconstruction

Call `save_hyperparameters` from the constructor to capture all arguments, selected names, or an explicit mapping. Initial values are deep-copied for logging and reconstruction. Explicit `load_from_checkpoint` keyword arguments override external CSV or YAML values, which override values stored in the checkpoint.

`Module.load_from_checkpoint` constructs the subclass, invokes `configure_model`, calls `on_load_checkpoint`, applies state under resolved strictness, reports non-strict key differences, and places the result on the mapped device. It does not construct a Trainer. Artifact-controlled instantiators are rejected.

## Runtime access

An attached Module exposes current epoch, global step, global and local rank, device mesh, loggers, optimizers, schedulers, and rank-safe printing. Nested SyntheticMind modules receive the same Trainer recursively. Pickle state removes the live Trainer reference so spawned workers reconstruct their own runtime graph.

`all_gather` accepts tensors, NumPy arrays, numeric scalars, and nested collections. The selected Strategy owns communication. `sync_grads=True` uses differentiable gathering; ordinary gathering does not retain a graph.

## Activation recomputation

`ActivationCheckpointing` is imported from
`syntheticmind.core.activation_checkpointing`. Wrap only a client-selected region
when constructing the model. Non-reentrant recomputation preserves the inherited
PyTorch call/hook path, and optional `saved_operations` selects operator overloads
to retain. No-grad calls bypass recomputation. RNG preservation is the default;
explicit eager RNG advancement is rejected under compilation.

The wrapper becomes part of the architecture and changes child state-dict paths;
reconstruct the same graph for loading. It does not pick layers, resize batches,
or retry OOMs. The [control guide](../docs/guides/runtime_control.md) records the
memory contract and CPU gradient verification.

## Context-parallel attention

Import `ContextParallelAttention` from its semantic module and construct it with
the context mesh in `configure_model`. It consumes equal contiguous local
`[batch, heads, sequence, features]` Q/K/V shards, differentiates packed K/V
communication, and applies global causal or bidirectional attention. It is not a
streaming ring algorithm and materializes global K/V. Replicated parameter
gradient reduction and weighted loss normalization belong to
`ContextParallelStrategy`, not this layer. See
[parallel algorithms](../docs/guides/parallel_algorithms.md).

## Related components

`../trainer/` orchestrates these contracts. `../data/` owns DataModule and loader composition. `../loops/` owns iterative execution. `../strategies/` owns placement and collectives. `../trainer/connectors/logger_connector/` consumes `LoggedMetric` observations.
