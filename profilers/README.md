# syntheticmind.profilers: Runtime Measurement

Profilers observe named Trainer, loop, hook, data, optimization, and persistence actions without owning those operations. `Profiler` owns shared stage setup, scoped action lifetime, report output, and deterministic teardown. Each concrete profiler owns one measurement technique.

| Module | Contents |
|---|---|
| `profiler.py` | Shared profiler lifecycle and the disabled pass-through implementation. |
| `simple.py` | Low-overhead wall-clock duration aggregation by action. |
| `advanced.py` | Python and C call attribution through `cProfile`. |
| `pytorch/` | Operator, device, module, scheduled-step, Chrome trace, and flame-graph profiling through `torch.profiler`. |

XLA profiling is intentionally absent until SyntheticMind owns an executable XLA accelerator and strategy path. A backend-specific profiler without a backend lifecycle is decorative compatibility baggage, not a usable training capability.
