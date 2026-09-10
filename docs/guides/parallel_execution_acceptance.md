# Parallel execution acceptance record

## Verified CPU implementation

Implementation commit `b4037f433e3fdd3275800aa43cc55218237014fd` is verified by
[workflow run 34428718274](https://github.com/rahul1-bot/syntheticmind/actions/runs/34428718274)
on 10 September 2026. It contains executable algorithms, not merely named mesh
axes. The exact source/runtime/dependency snapshot is retained as a CI artifact.

| Boundary | Observed evidence |
|---|---|
| Context computation | Local outputs and Q/K/V gradients match full-sequence causal attention. |
| Context optimization | Replicated parameter updates match a sequential reference; unequal objective weights match under mean and sum gradient reductions. |
| Context continuation | Interrupted/resumed weights and rank-local torch RNG match exactly. |
| Pipeline schedules | Native GPipe and 1F1B match sequential stage parameters under AdamW. |
| Pipeline continuation | Dropout training resumes with bit-exact local weights and next torch RNG. |
| Worker recovery | An actual rank-1 exit at step 2 triggers worker-group restart, automatic durable step-1 restore, and exact final model/optimizer/RNG/step equality. |
| Sharded persistence | Snapshot isolation, concurrent training collectives, retention and failed publication preserve the previous active generation. |
| Compiler controls | CPU Inductor and identity/resume tests pass; bounded native IDs, graph-break reasons, report serialization and backend-failure cleanup pass. |

The [algorithm guide](parallel_algorithms.md), [algorithm probes](parallel_algorithms_verification.md),
and [failure probes](runtime_failure_verification.md) specify the tested paths.
Correctness and ownership are separate obligations: the existing Trainer/manual
loop still owns lifecycle and managed steps, Strategies own communication and
schedules, the attention Module owns computation, and the rank-local checkpoint
representation owns partition names and per-rank trajectory records.

## Boundaries that remain outside this evidence

CP uses equal contiguous shards and global K/V all-gather, not a streaming ring,
padding-aware attention, GQA, or hybrid parallel composition. PP uses one static
single-tensor stage per rank and fit without validation; native microbatch
backward is not a claim to execute the ordinary manual_backward hook family.
Both require fixed topology and supported ordinary floating precision.

CUDA/NCCL, FP8/FSDP2 composition, MPS, remote asynchronous storage, physical-node
loss, network partitions, AOT/cache export and automatic activation-memory budgets
require separate work or hardware evidence. The current tests do not emulate
those paths and label them successful.
