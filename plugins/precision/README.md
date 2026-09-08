# syntheticmind.plugins.precision: Numerical Execution

`plugins/precision/` owns parameter conversion, input conversion, forward contexts, backward preparation, gradient scaling, optimizer stepping, clipping, and precision checkpoint state.

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
| `torchao.py` | Explicit optional TorchAO float8 conversion and recipe-state compatibility. |

Transformer Engine support targets the current `transformer_engine>=2.18.0` API and calls `autocast(enabled=True, recipe=...)`. Recipe dictionaries are not accepted as unvalidated bags; use `DelayedScalingConfiguration` or a live object satisfying `DelayedScalingRecipe`. See the [NVIDIA Transformer Engine common API](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html) and [FP8 usage guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html).

Precision vocabulary and compatibility are resolved by the Trainer accelerator connector because the valid policy depends on the selected accelerator, Strategy, and explicitly supplied precision plugin.

## Policy matrix

| Trainer value | Parameter storage | Forward compute | Gradient scaling |
|---|---|---|---|
| `32-true` | Unchanged, normally float32 | Float32 | None |
| `64-true` | Float64 | Float64 | None |
| `16-true` | Float16 | Float16 | None |
| `bf16-true` | Bfloat16 | Bfloat16 | None |
| `16-mixed` | Float32 | Float16 autocast | Yes |
| `bf16-mixed` | Float32 | Bfloat16 autocast | No |
| `transformer-engine` | Bfloat16 | Transformer Engine FP8 with bfloat16 fallback | No GradScaler |
| `transformer-engine-float16` | Float16 | Transformer Engine FP8 with float16 fallback | No GradScaler |

True precision changes model storage and floating batch inputs. Mixed precision keeps float32 master weights and selects autocast compute. Integer, boolean, string, byte, and unsupported custom leaves remain unchanged.

## Lifecycle order

Trainer and Strategy invoke precision through one ordered lifecycle:

1. Tensor and module construction contexts select initialization dtype.
2. `convert_module` applies storage conversion or layer replacement.
3. `convert_input` prepares floating batch leaves.
4. The stage step runs inside its precision context.
5. `pre_backward`, Module backward, and `post_backward` dispatch hooks around differentiation.
6. Optimizer closure completion exposes unscaled gradients to pre-step hooks and clipping.
7. Optimizer mutation and scaler update occur.
8. Precision state enters checkpoints and teardown clears transient state.

Manual optimization uses the same backward and managed step paths. Automatic clipping is suppressed in manual mode so client code chooses exactly when it runs.

## Custom precision

A custom policy subclasses `Precision` and overrides only the lifecycle transitions whose numerical semantics differ. Supply the instance through `Trainer(plugins=[precision_plugin])`. `connect` may replace model, optimizers, or schedulers, but returned identities must remain coherent with Strategy and checkpoint ownership.

## Mixed precision constraints

Float16 mixed precision rejects LBFGS and persists GradScaler state for exact continuation. Gradients unscale once per optimizer before inspection or clipping. Optimizers that advertise internal AMP scaling cannot also use framework clipping because duplicate unscale is invalid.

## Transformer Engine constraints

Transformer Engine requires CUDA, `transformer_engine>=2.18.0`, float16 or bfloat16 weights, and eligible layer dimensions. Native Linear replacement requires input and output widths divisible by 16. LayerNorm replacement requires one normalized dimension. Unsupported layers remain native with a warning.

`replace_layers=True` redirects construction during `configure_model`. `replace_layers=None` recursively converts an already constructed model only when no Transformer Engine layer exists. `replace_layers=False` preserves client layer types.

## TorchAO float8 constraints

Import `TorchAOFloat8Precision` explicitly from
`syntheticmind.plugins.precision.torchao` and supply the instance through
`Trainer(plugins=[...])`. The resolved `torchao-float8` identity is not a Trainer
string shortcut. Ordinary precision imports do not load the optional package.

Recipes are `tensorwise`, `rowwise`, or `rowwise_with_gw_hp`. Parameter storage and
floating inputs are BF16; eligible Linear computation uses FP8. NVIDIA CUDA,
compute capability at least 8.9, and native Linear dimensions divisible by 16 are
required. Custom Linear subclasses are not replaced. Convert native regions
before applying a TP/FSDP2 plan; already-sharded native weights are rejected.
Version-1 precision state preserves the recipe and rejects changed continuation.

The TorchAO 0.18 CPU probe verifies import, recipe state, and rejection before
mutation. It does not validate CUDA numerics, compiled float8, or FSDP2 hardware
behavior. See [runtime control](../../docs/guides/runtime_control.md).

## Related components

`../../trainer/connectors/accelerator_connector.py` selects and validates the policy. `../../strategies/strategy.py` routes backward and optimizer stepping. `../../core/module.py` provides customization hooks. `../../loops/optimization/` owns automatic and manual closure execution.
