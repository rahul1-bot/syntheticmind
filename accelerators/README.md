# syntheticmind.accelerators: Device Ownership

`accelerators/` owns backend-specific device discovery, parsing, binding, and device-stat collection. Cross-component topology selection belongs to the Trainer accelerator connector because accelerator, devices, nodes, Strategy, precision, and execution plugins form one compatibility decision.

| Module | Ownership |
|---|---|
| `accelerator.py` | The structural `Accelerator` contract consumed by Trainer and strategies. |
| `device_parser.py` | Stateful GPU-selection parsing with count, identifier, uniqueness, and availability invariants. |
| `cpu.py` | CPU selection, binding, process-time, core-count, and load-average statistics. |
| `cuda.py` | CUDA discovery, identifier resolution, process-current-device binding, allocator counters, and device-memory statistics. |
| `mps.py` | Apple MPS discovery, identifier resolution, binding, and allocator-memory statistics. |

The accelerator implementations answer device-local questions only. `trainer/connectors/accelerator_connector.py` resolves the participating devices and connects them to the selected Strategy.

## Client selection

Clients normally select a backend and devices through Trainer:

```python
from syntheticmind import Trainer

trainer: Trainer = Trainer(
    accelerator="cuda",
    devices=[0, 2],
    precision="bf16-mixed"
)
```

`accelerator="auto"` prefers CUDA, then MPS, then CPU according to live availability. A positive GPU integer is a device count beginning at zero. A numeric or comma-separated string identifies explicit devices. `-1` selects every visible GPU. CPU integers represent worker-process count because CPU has no indexed accelerator slots.

Device parsing occurs before Strategy construction. Empty selections, duplicates, negative identifiers, and unavailable identifiers fail before a process group or device context is created.

## Custom accelerators

A custom implementation explicitly satisfies `Accelerator` and may be passed as `Trainer(accelerator=instance)`. It owns backend-local discovery, parsing, device construction, process binding, and statistics. It must not launch processes, select distributed algorithms, resolve precision, or perform model placement; those responsibilities belong to Strategy and the Trainer connector.

Class-level methods expose discovery and device topology before an instance is bound. `setup_device` performs process-local backend state after rank selection. `get_device_stats` returns scalar counters with stable backend-qualified names and native units.

## Statistics

CPU reports process time, logical CPU count, and available load averages. CUDA reports allocator current and peak bytes, retry and out-of-memory counts, and free and total device bytes. MPS reports current, driver, and recommended maximum memory bytes. `DeviceStatsMonitor` owns observation cadence and logger routing; it never switches directly on backend globals.

## Related components

`../trainer/connectors/accelerator_connector.py` resolves compatibility. `../strategies/` owns model and process execution. `../plugins/precision/` owns numerical policy. `../callbacks/device_stats_monitor.py` consumes statistics.
