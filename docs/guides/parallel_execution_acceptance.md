# Parallel execution acceptance

This document records outstanding implementation requirements. It does not
advertise completed context-parallel attention or pipeline scheduling.
The existing `context_parallel_size` and `pipeline_parallel_size` options in
ModelParallelStrategy supply topology only and require manual optimization.

## Context parallelism

A complete path must execute attention over a sequence partitioned across ranks,
not merely name a mesh axis. The implementation must define sequence placement,
attention-mask and causal-position semantics, gradient communication, loss
normalization, and interactions with the data- and tensor-parallel axes.

Acceptance requires a deterministic tiny model comparing the distributed output,
loss, parameter gradients, and optimizer result to an unpartitioned reference.
Include noncausal and causal attention, invalid shapes, multiple optimizer steps,
accumulation where supported, and interrupted/resumed execution. Unequal token
counts or padding must be correctly weighted or explicitly rejected. Do not
multiply sample accounting by the number of context collaborators.

An all-gather reference algorithm and a ring-attention implementation have
different memory and communication costs. Report which algorithm executes rather
than claiming the performance characteristics of another implementation.

## Pipeline parallelism

A complete path must execute stage-local forwards and backwards under an actual
microbatch schedule. Native GPipe or 1F1B integration must define stage boundaries,
input/target routing, output ownership, loss scaling, and optimizer cadence.
Do not let the normal automatic loop backward execute again after a native
pipeline schedule already performed backward.

Acceptance requires at least a real two-stage distributed run comparing its
loss, gradients, and optimizer result with the same nonpartitioned model. Test
several microbatches, stage-local parameter ownership, empty/malformed inputs,
forward-only evaluation, final-stage outputs, and rank-consistent global-step
accounting. State exactly which schedule and microbatch shapes are supported.

## Checkpoint representation

Pipeline stages may have distinct parameters and optimizer state. Identically
named local fields cannot be treated as replicated state and silently overwrite
one another. Stage identities, parameter names, optimizer parameter groups, and
scheduler state must have an unambiguous distributed representation.

A resumable checkpoint may be published only at a coherent optimizer boundary,
with no incomplete microbatch work or unrepresented gradients. Restore must
reconstruct the same stage/mesh plan or explicitly establish a supported
migration. Equal world size alone does not prove equivalent DP/TP/CP/PP topology.

Compare uninterrupted and resumed runs, including optimizer and scheduler state,
not only final weights. Inject shard and publication failures and verify that
the previous generation remains active. All-rank work must not be introduced
inside a rank-zero-only inspection or retention path.

## Lifecycle and failure

The selected Strategy owns process groups and execution placement. Existing
Module, Precision, loop, and checkpoint owners retain their established
responsibilities. Any new runtime owner must own an actual algorithm, resource,
state machine, or failure boundary, not merely forward one call to PyTorch.

Setup, client hooks, optimizer operations, sequential stage reuse, exception
propagation, and teardown must be tested together. Report unsupported precision,
compilation, and sharding combinations before mutation. CPU reference execution
does not establish CUDA/NCCL, multi-node, or large-model performance.

## Submission gate

Keep the PR in draft until the implemented scope and executable evidence agree.
Documentation must separate completed functionality, verified environments,
explicitly rejected combinations, and remaining work. Source comments and
package guides must use the actual public imports and signatures. No change to
`dev` or `main`, merge, or force-push is authorized by this implementation task.
