# syntheticmind.accelerators: Device Ownership

`accelerators/` owns backend-specific device discovery, parsing, and binding. Cross-component topology selection belongs to the Trainer accelerator connector because accelerator, devices, nodes, Strategy, precision, and execution plugins form one compatibility decision.

| Module | Ownership |
|---|---|
| `accelerator.py` | The structural `Accelerator` contract consumed by Trainer and strategies. |
| `device_parser.py` | Stateful GPU-selection parsing with count, identifier, uniqueness, and availability invariants. |
| `cpu.py` | CPU selection and binding. |
| `cuda.py` | CUDA discovery, identifier resolution, and process-current-device binding. |
| `mps.py` | Apple MPS discovery, identifier resolution, and binding. |

The accelerator implementations answer device-local questions only. `trainer/connectors/accelerator_connector.py` resolves the participating devices and connects them to the selected Strategy.
