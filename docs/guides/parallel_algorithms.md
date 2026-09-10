# Context and pipeline execution

These execution policies add algorithms beyond the generic ModelParallelStrategy
mesh axes. They use the existing Trainer and manual optimization loop; they do
not create an alternative training entrypoint. CPU runtime verification is in
[the executable probe](parallel_algorithms_verification.md).

## Context parallelism

Construct `ContextParallelConfiguration(context_parallel_size=2)` and pass it to
`ContextParallelStrategy(configuration)`. In `Module.configure_model`, construct
`ContextParallelAttention(self.device_mesh["context_parallel"])` from
`syntheticmind.core.context_parallel_attention`.

Each rank receives the same complete sample and slices its sequence into equal,
contiguous shards. Attention accepts local `[batch, heads, sequence, features]`
query, key and value tensors. It differentiably all-gathers K/V, computes only
local queries, and applies the global rank-offset causal mask. Bidirectional
attention is also selectable. This is an exact all-gather implementation, **not**
a streaming ring implementation: it materializes global K/V on each rank. Unequal
shards, padding masks and grouped-query attention are not supported here.

Use manual optimization: zero the managed optimizer, compute the local loss,
call `manual_backward(loss)`, then `optimizer.step()`. For unequal valid-token or
element weights, call `strategy.normalize_loss(local_loss_sum, local_weight)`
first: pass a local summed objective and its count of contributing units. Global
weight must be positive; a rank may contribute zero. The method accounts for
mean versus sum parameter-gradient reduction without detaching the loss. It does
not change attention masking or infer a dataset denominator. The Strategy reduces
replicated parameter gradients inside the optimizer closure before pre-step
hooks. Mean reduction matches equal-size local mean losses; sum reduction is an
explicit alternate policy. Do not separately reduce the same gradients.

The policy is CP-only with replicated ordinary tensors, one managed optimizer,
and float32 or true BF16. It excludes LBFGS and DTensor composition. Initial
parameters/buffers are broadcast from rank zero; subsequent buffer semantics are
client-owned. Mutable buffers and rank RNG are checkpointed separately. This does
not implement synchronized batch normalization or promise compatibility with
arbitrary mutable client state.

## Pipeline parallelism

`PipelineParallelConfiguration` contains a tuple of `PipelineStageShape` values,
`n_microbatches`, `schedule="gpipe"` or `"1f1b"`, a declared `loss_reduction`, and a
client-owned `loss_id`. Pass it and a scalar `loss_fn` to
`PipelineParallelStrategy(configuration, loss_fn)`.

One ordinary Module is one stage on one rank. Its `forward` maps a single tensor
to a single tensor. Stage shapes describe a microbatch and must connect exactly.
The default float32 or selected true BF16 determines communication dtype. There
is no tracing or dummy forward during shape setup. Intermediate input and all output
metadata are differentiable so native schedules allocate gradient communication.
First-stage batches must match the configured floating dtype and must not require
input gradients. Model construction remains
inside the client's `configure_model` contract.

Each training_step zeros its managed optimizer and calls `strategy.pipeline_step`
on every rank. The first rank passes the full input batch, the final rank passes
the full target, and other arguments are None. Native PyTorch GPipe or 1F1B owns
microbatch splitting, forward/backward scheduling, and activation communication.
The method returns the same detached reporting loss on every rank. Then call
`optimizer.step()` once. **Do not call manual_backward on the returned loss:**
the native pipeline schedule has already differentiated it.

The loss declaration must match the function: mean divides accumulated gradients
by microbatch count, while sum does not. The current policy supports fit without
validation, static equal-size tensor microbatches, one managed optimizer per
stage, and no DP/TP/CP composition or LBFGS. Native microbatch backward does not
invoke SyntheticMind's ordinary manual_backward hook family. Optimizer hooks and
managed global-step accounting remain active. GPU execution requires separate
hardware verification; CPU/Gloo evidence is not a throughput claim.

## Partition-aware checkpoints

Both policies use sharded checkpoints and optionally the existing asynchronous
generation store. Tensor keys are prefixed with the rank, preventing collisions
between stages with identically named layers. Every rank contributes its own
ordinary schema-2 Trainer metadata, including RNG, callbacks, data and loop
continuation. A version-1 rank-local envelope stores the exact parallel policy.

Loading rejects unknown envelope fields/versions, a changed parallel policy,
world size or optimizer count before applying live weights. A pipeline schedule
in progress, unapplied pipeline gradients, unapplied context backward or failed
execution is not a valid checkpoint boundary. Finish the managed optimizer step
before saving. Reconstruction is through Trainer.fit, not single-file
Module.load_from_checkpoint.

Rank-local state costs more storage than deduplicated replicated checkpoints.
That is deliberate: a rank-zero-only metadata shortcut cannot preserve each
stage's stochastic trajectory. This implementation does not support topology
changes during exact resume.

## Verification

The [10 September CPU run](https://github.com/rahul1-bot/syntheticmind/actions/runs/34428718274)
passes output/gradient comparisons, context parameter updates, unequal objective
weights, native GPipe and 1F1B, and exact per-rank stochastic continuation. A
separate actual worker-exit fixture restarts through TorchElastic and compares
model, optimizer, step and torch RNG against an uninterrupted baseline. The
[probe sources](parallel_algorithms_verification.md) explain aligned sample order
and the numerical distinction between cross-topology equivalence and exact
same-topology continuation. These results do not certify GPU or hybrid operation.
