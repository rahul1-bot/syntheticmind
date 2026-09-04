# syntheticmind.precision: Numerical Execution

`precision/` owns parameter conversion, input conversion, forward contexts, backward preparation, gradient scaling, optimizer stepping, clipping, and precision checkpoint state.

| Module | Ownership |
|---|---|
| `precision.py` | The concrete full-precision policy and shared precision lifecycle. |
| `half.py` | True float16 or bfloat16 parameters and inputs. |
| `double.py` | True float64 parameters and inputs. |
| `mixed.py` | Autocast plus optional gradient scaling and unscaling. |
| `conversion.py` | Default-dtype context ownership. |
| `transformer_engine_contracts.py` | Strict delayed-scaling configuration and the live recipe contract. |
| `transformer_engine_adapter.py` | The optional NVIDIA package boundary. |
| `transformer_engine.py` | Transformer Engine module replacement and FP8 execution policy. |

Transformer Engine support targets the current `transformer_engine>=2.18.0` API and calls `autocast(enabled=True, recipe=...)`. Recipe dictionaries are not accepted as unvalidated bags; use `DelayedScalingConfiguration` or a live object satisfying `DelayedScalingRecipe`. See the [NVIDIA Transformer Engine common API](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html) and [FP8 usage guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html).

Precision vocabulary and compatibility are resolved by the Trainer accelerator connector because the valid policy depends on the selected accelerator, Strategy, and explicitly supplied precision plugin.
