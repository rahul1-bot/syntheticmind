# syntheticmind.accelerators: Device Ownership

`accelerators/` owns device availability, selection, binding, and the resolved device topology consumed by strategies.

| Module | Ownership |
|---|---|
| `accelerator.py` | The structural `Accelerator` contract consumed by Trainer and strategies. |
| `device_parser.py` | Stateful GPU-selection parsing with count, identifier, uniqueness, and availability invariants. |
| `topology.py` | The strict immutable `DeviceTopology` resolved from accelerator, device, and node configuration. |
| `cpu.py` | CPU selection and binding. |
| `cuda.py` | CUDA discovery, identifier resolution, and process-current-device binding. |
| `mps.py` | Apple MPS discovery, identifier resolution, and binding. |

Strategies own process placement and model wrapping after the accelerator resolves the participating devices.
