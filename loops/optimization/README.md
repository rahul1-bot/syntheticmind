# syntheticmind.loops.optimization: Step Ownership

`optimization/` contains the two internal training-step execution contracts selected by `TrainingEpochLoop`. Clients do not import these classes; they choose behavior through `Module.automatic_optimization`.

| Module | Contents |
|---|---|
| `automatic.py` | Automatic loss extraction, normalization, zeroing, accumulation scaling, and backward dispatch. |
| `manual.py` | Manual training-step execution and reporting-output normalization. |

## Automatic output contract

`training_step` may return:

1. A scalar tensor used as the loss.
2. A mapping containing a scalar tensor under `"loss"`; additional values reach batch-end hooks.
3. `None` to skip one local batch outside distributed execution.

Any other return type fails. A mapping without a tensor loss fails. Distributed execution rejects `None` because one rank skipping backward would violate collective participation.

The loss divides by `accumulate_grad_batches` before backward. The detached normalized loss enters reporting output, while the original scaled graph drives differentiation. Zeroing occurs only at the beginning of an accumulation window.

## Manual output contract

Manual optimization accepts the same tensor, mapping, or `None` shapes for reporting. Tensor losses detach before batch-end hooks. No backward, zeroing, clipping, optimizer mutation, or scheduler step is inferred from the return value.

The client performs those operations through Module and `SyntheticOptimizer`:

```python
self.automatic_optimization = False
optimizer = self.optimizers()
optimizer.zero_grad()
loss: torch.Tensor = self.compute_loss(batch)
self.manual_backward(loss)
optimizer.step()
return {"loss": loss.detach()}
```

## Related components

`../training_epoch_loop.py` composes these operations with data, hooks, metrics, validation, schedulers, checkpoint durability, and progress. `../../core/optimizer.py` defines managed optimizer behavior. `../../plugins/precision/` owns backward and optimizer numerical policy.
