# This module:
# 1. Resolves and connects Accelerator, devices, nodes, Strategy, precision,
#    ClusterEnvironment, CheckpointIO, and LayerSync from one Trainer
#    configuration boundary
# 2. Validates duplicate plugin families, conflicting ownership, unsupported
#    topology, unavailable backends, and numerical compatibility before runtime
#
# Design decisions:
# - These collaborators form one compatibility matrix and therefore resolve in a
#   strict order: plugins, accelerator, devices, Strategy, topology, checkpoint,
#   environment, layer sync, precision, then precision attachment
# - A configured Strategy may own accelerator, devices, precision, checkpoint,
#   environment, or layer policy; specifying the same concern through Trainer is
#   rejected instead of silently choosing precedence
# - Automatic accelerator selection prefers CUDA, then MPS, then CPU; ``"gpu"``
#   preserves its established MPS-before-CUDA compatibility behavior
# - Distributed topology requires a distributed Strategy and MPS remains
#   single-device because no verified SyntheticMind distributed MPS path exists
# - Precision aliases normalize once, discouraged shorthand warns, float16 CPU
#   mixed precision becomes bfloat16, and explicit precision plugins cannot be
#   combined with a separate precision value
# - No temporary topology or plugin data model duplicates live collaborator state
#
# Author: Rahul Sawhney

import warnings
from typing import cast

import torch

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.accelerators.cpu import CPUAccelerator
from syntheticmind.accelerators.cuda import CUDAAccelerator
from syntheticmind.accelerators.device_parser import DeviceInput
from syntheticmind.accelerators.mps import MPSAccelerator
from syntheticmind.plugins import Plugin
from syntheticmind.plugins.environments import ClusterEnvironment
from syntheticmind.plugins.checkpoint_io import CheckpointIO
from syntheticmind.plugins.layer_sync import LayerSync, TorchSyncBatchNorm
from syntheticmind.plugins.precision.double import DoublePrecision
from syntheticmind.plugins.precision.half import HalfPrecision
from syntheticmind.plugins.precision.mixed import MixedPrecision
from syntheticmind.plugins.precision.precision import Precision
from syntheticmind.plugins.precision.transformer_engine import TransformerEnginePrecision
from syntheticmind.strategies.ddp import DDPStrategy
from syntheticmind.strategies.parallel import ParallelStrategy
from syntheticmind.strategies.single_device import SingleDeviceStrategy
from syntheticmind.strategies.strategy import Strategy
from syntheticmind.primitives.exceptions import MisconfigurationError
from syntheticmind.primitives.types import PrecisionInput, PrecisionType

__all__: list[str] = []

type _ResolvedPlugins = tuple[
    Precision | None,
    CheckpointIO | None,
    ClusterEnvironment | None,
    LayerSync | None
]


class _AcceleratorConnector:
    # Private Trainer composition owner for device and execution collaborators.
    def __init__(
        self,
        accelerator: str | Accelerator,
        strategy: str | Strategy,
        devices: DeviceInput,
        num_nodes: int,
        precision: PrecisionInput | None,
        plugins: Plugin | list[Plugin] | None,
        sync_batchnorm: bool
    ) -> None:
        # Resolves the complete compatibility matrix and leaves Strategy fully
        # configured with topology, plugins, and numerical policy.
        resolved_plugins: _ResolvedPlugins = self._resolve_plugins(
            plugins,
            sync_batchnorm
        )
        configured_precision: Precision | None = resolved_plugins[0]
        configured_checkpoint_io: CheckpointIO | None = resolved_plugins[1]
        configured_cluster_environment: ClusterEnvironment | None = resolved_plugins[2]
        configured_layer_sync: LayerSync | None = resolved_plugins[3]
        configured_parallel_devices: list[torch.device] | None = None
        resolved_accelerator_input: str | Accelerator = accelerator
        accelerator_is_auto: bool = (
            isinstance(accelerator, str) and accelerator.lower() == "auto"
        )
        if isinstance(strategy, Strategy):
            if strategy.accelerator is not None:
                if not accelerator_is_auto:
                    raise MisconfigurationError(
                        "Set the accelerator through either Trainer or the strategy instance, not both."
                    )
                resolved_accelerator_input: str | Accelerator = strategy.accelerator
            if strategy.parallel_devices:
                if devices != "auto":
                    raise MisconfigurationError(
                        "Set devices through either Trainer or the strategy instance, not both."
                    )
                configured_parallel_devices: list[torch.device] | None = list(
                    strategy.parallel_devices
                )
                if accelerator_is_auto and strategy.accelerator is None:
                    resolved_accelerator_input: str | Accelerator = (
                        configured_parallel_devices[0].type
                    )
        self.accelerator: Accelerator = self._resolve_accelerator(
            resolved_accelerator_input
        )
        device_selection: tuple[int | list[int], list[torch.device]] = (
            self._resolve_devices(
                self.accelerator,
                devices,
                num_nodes,
                configured_parallel_devices
            )
        )
        self.devices: int | list[int] = device_selection[0]
        self.parallel_devices: list[torch.device] = device_selection[1]
        self.num_nodes: int = num_nodes
        self.strategy: Strategy = self._resolve_strategy(
            strategy,
            self.accelerator,
            self.parallel_devices,
            self.num_nodes
        )
        self.strategy.configure_topology(
            self.accelerator,
            self.parallel_devices,
            self.num_nodes
        )
        self._attach_checkpoint_io(configured_checkpoint_io)
        self._attach_cluster_environment(configured_cluster_environment)
        self._attach_layer_sync(configured_layer_sync)
        self.precision_plugin: Precision = self._resolve_precision(
            precision,
            self.accelerator.name(),
            configured_precision,
            self.strategy.configured_precision_plugin
        )
        self.strategy.attach_precision_plugin(self.precision_plugin)

    def _resolve_plugins(
        self,
        plugins: Plugin | list[Plugin] | None,
        sync_batchnorm: bool
    ) -> _ResolvedPlugins:
        # Classifies supplied structural plugin instances, rejects duplicate
        # families, and materializes sync_batchnorm as TorchSyncBatchNorm.
        if type(sync_batchnorm) is not bool:
            raise TypeError("sync_batchnorm must be a bool.")
        if plugins is None:
            plugin_list: list[Plugin] = []
        elif isinstance(
            plugins,
            (Precision, CheckpointIO, ClusterEnvironment, LayerSync)
        ):
            plugin_list: list[Plugin] = [plugins]
        elif isinstance(plugins, list):
            plugin_list: list[Plugin] = list(plugins)
        else:
            raise TypeError("plugins must be a plugin instance, a plugin list, or None.")
        precision: Precision | None = None
        checkpoint_io: CheckpointIO | None = None
        cluster_environment: ClusterEnvironment | None = None
        layer_sync: LayerSync | None = None
        duplicate_categories: list[str] = []
        plugin: Plugin
        for plugin in plugin_list:
            if isinstance(plugin, Precision):
                if precision is not None:
                    duplicate_categories.append("Precision")
                precision: Precision | None = plugin
            elif isinstance(plugin, CheckpointIO):
                if checkpoint_io is not None:
                    duplicate_categories.append("CheckpointIO")
                checkpoint_io: CheckpointIO | None = plugin
            elif isinstance(plugin, ClusterEnvironment):
                if cluster_environment is not None:
                    duplicate_categories.append("ClusterEnvironment")
                cluster_environment: ClusterEnvironment | None = plugin
            elif isinstance(plugin, LayerSync):
                if layer_sync is not None:
                    duplicate_categories.append("LayerSync")
                layer_sync: LayerSync | None = plugin
            else:
                raise MisconfigurationError(
                    f"Unsupported plugin {plugin!r}. Expected Precision, CheckpointIO, "
                    "ClusterEnvironment, or LayerSync."
                )
        if duplicate_categories:
            categories: str = ", ".join(dict.fromkeys(duplicate_categories))
            raise MisconfigurationError(
                f"Received multiple values for {categories} in plugins."
            )
        if sync_batchnorm:
            if layer_sync is not None and not isinstance(layer_sync, TorchSyncBatchNorm):
                raise MisconfigurationError(
                    "sync_batchnorm conflicts with the configured LayerSync plugin."
                )
            layer_sync: LayerSync | None = layer_sync or TorchSyncBatchNorm()
        return precision, checkpoint_io, cluster_environment, layer_sync

    def _resolve_accelerator(
        self,
        accelerator: str | Accelerator
    ) -> Accelerator:
        # Resolves built-in names or a custom structural implementation and
        # verifies live availability before device parsing.
        if isinstance(accelerator, Accelerator):
            resolved_accelerator: Accelerator = accelerator
        elif not isinstance(accelerator, str):
            raise TypeError(
                f"`accelerator` must be a string or Accelerator instance, got {accelerator!r}."
            )
        else:
            normalized_accelerator: str = accelerator.lower()
            match normalized_accelerator:
                case "cuda":
                    resolved_accelerator: Accelerator = CUDAAccelerator()
                case "mps":
                    resolved_accelerator: Accelerator = MPSAccelerator()
                case "cpu":
                    resolved_accelerator: Accelerator = CPUAccelerator()
                case "gpu":
                    if MPSAccelerator.is_available():
                        resolved_accelerator: Accelerator = MPSAccelerator()
                    elif CUDAAccelerator.is_available():
                        resolved_accelerator: Accelerator = CUDAAccelerator()
                    else:
                        raise MisconfigurationError(
                            "No supported GPU accelerator is available."
                        )
                case "auto":
                    if CUDAAccelerator.is_available():
                        resolved_accelerator: Accelerator = CUDAAccelerator()
                    elif MPSAccelerator.is_available():
                        resolved_accelerator: Accelerator = MPSAccelerator()
                    else:
                        resolved_accelerator: Accelerator = CPUAccelerator()
                case _:
                    raise ValueError(
                        f"Unknown accelerator {accelerator!r}. Available choices are "
                        "'auto', 'gpu', 'cpu', 'cuda', and 'mps'."
                    )
        if not type(resolved_accelerator).is_available():
            available_accelerators: list[str] = ["cpu"]
            if CUDAAccelerator.is_available():
                available_accelerators.append("cuda")
            if MPSAccelerator.is_available():
                available_accelerators.append("mps")
            raise MisconfigurationError(
                f"{type(resolved_accelerator).__name__} is not available on this system. "
                f"Available accelerators: {available_accelerators}."
            )
        return resolved_accelerator

    def _resolve_devices(
        self,
        accelerator: Accelerator,
        devices: DeviceInput,
        num_nodes: int,
        configured_parallel_devices: list[torch.device] | None
    ) -> tuple[int | list[int], list[torch.device]]:
        # Validates node count and resolves Trainer input or Strategy-configured
        # devices through the selected Accelerator.
        if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes < 1:
            raise ValueError(f"`num_nodes` must be positive, got {num_nodes!r}.")
        if configured_parallel_devices:
            parsed_devices: int | list[int] = self._parse_configured_devices(
                accelerator,
                configured_parallel_devices
            )
        else:
            resolved_devices: DeviceInput = (
                accelerator.auto_device_count()
                if devices == "auto"
                else devices
            )
            parsed_devices: int | list[int] = accelerator.parse_devices(
                resolved_devices
            )
        parallel_devices: list[torch.device] = accelerator.get_parallel_devices(
            parsed_devices
        )
        return parsed_devices, parallel_devices

    def _parse_configured_devices(
        self,
        accelerator: Accelerator,
        configured_parallel_devices: list[torch.device]
    ) -> int | list[int]:
        # Verifies preconfigured device types and converts concrete torch.device
        # objects back through Accelerator parsing for one canonical validation.
        expected_device_type: str = accelerator.name()
        actual_device_types: set[str] = set()
        device: torch.device
        for device in configured_parallel_devices:
            actual_device_types.add(device.type)
        if actual_device_types != {expected_device_type}:
            raise ValueError(
                f"Configured devices {configured_parallel_devices} do not match "
                f"accelerator={expected_device_type!r}."
            )
        if expected_device_type == "cpu":
            return accelerator.parse_devices(len(configured_parallel_devices))
        device_ids: list[int] = []
        device_index: int
        for device_index, device in enumerate(configured_parallel_devices):
            resolved_index: int = (
                device.index if device.index is not None else device_index
            )
            device_ids.append(resolved_index)
        return accelerator.parse_devices(device_ids)

    def _resolve_strategy(
        self,
        strategy: str | Strategy,
        accelerator: Accelerator,
        parallel_devices: list[torch.device],
        num_nodes: int
    ) -> Strategy:
        # Resolves standard names or a concrete Strategy and validates distributed
        # requirements plus the unsupported MPS parallel boundary.
        if isinstance(strategy, Strategy):
            resolved_strategy: Strategy = strategy
        elif not isinstance(strategy, str):
            raise TypeError(
                f"`strategy` must be a string or Strategy instance, got {strategy!r}."
            )
        else:
            normalized_strategy: str = strategy.lower()
            match normalized_strategy:
                case "ddp":
                    resolved_strategy: Strategy = DDPStrategy()
                case "ddp_spawn":
                    resolved_strategy: Strategy = DDPStrategy(start_method="spawn")
                case "ddp_fork" | "ddp_notebook":
                    resolved_strategy: Strategy = DDPStrategy(start_method="fork")
                case "single_device":
                    resolved_strategy: Strategy = SingleDeviceStrategy()
                case "auto":
                    if len(parallel_devices) > 1 or num_nodes > 1:
                        resolved_strategy: Strategy = DDPStrategy()
                    else:
                        resolved_strategy: Strategy = SingleDeviceStrategy()
                case _:
                    raise ValueError(
                        f"Unknown strategy {strategy!r}. Available choices are "
                        "'auto', 'single_device', 'ddp', 'ddp_spawn', "
                        "'ddp_fork', and 'ddp_notebook'."
                    )
        requires_distributed_strategy: bool = (
            len(parallel_devices) > 1 or num_nodes > 1
        )
        if isinstance(accelerator, MPSAccelerator) and resolved_strategy.is_distributed:
            raise MisconfigurationError(
                "Distributed strategies are not supported with the MPS accelerator. "
                "Use accelerator='cpu' or a single-device strategy."
            )
        if requires_distributed_strategy and not resolved_strategy.is_distributed:
            raise MisconfigurationError(
                f"{type(resolved_strategy).__name__} cannot manage num_devices="
                f"{len(parallel_devices)} and num_nodes={num_nodes}. Use strategy='ddp'."
            )
        return resolved_strategy

    def _attach_checkpoint_io(self, checkpoint_io: CheckpointIO | None) -> None:
        # Attaches transport only when Strategy did not already configure it.
        if checkpoint_io is None:
            return
        if self.strategy.checkpoint.configured_checkpoint_io is not None:
            raise MisconfigurationError(
                "CheckpointIO was configured through both Trainer plugins and the strategy. "
                "Configure it in exactly one place."
            )
        self.strategy.checkpoint.attach_checkpoint_io(checkpoint_io)

    def _attach_cluster_environment(
        self,
        cluster_environment: ClusterEnvironment | None
    ) -> None:
        # Restricts cluster topology plugins to ParallelStrategy and single-source
        # ownership.
        if cluster_environment is None:
            return
        if not isinstance(self.strategy, ParallelStrategy):
            raise MisconfigurationError("ClusterEnvironment requires a parallel strategy.")
        if self.strategy.configured_cluster_environment is not None:
            raise MisconfigurationError(
                "ClusterEnvironment was configured through both Trainer plugins and the strategy. "
                "Configure it in exactly one place."
            )
        self.strategy.attach_cluster_environment(cluster_environment)

    def _attach_layer_sync(self, layer_sync: LayerSync | None) -> None:
        # Restricts reversible layer synchronization to DDP and single-source
        # ownership.
        if layer_sync is None:
            return
        if not isinstance(self.strategy, DDPStrategy):
            raise MisconfigurationError("LayerSync requires DDPStrategy.")
        if self.strategy.configured_layer_sync is not None:
            raise MisconfigurationError(
                "LayerSync was configured through both Trainer plugins and the strategy. "
                "Configure it in exactly one place."
            )
        self.strategy.attach_layer_sync(layer_sync)

    def _resolve_precision(
        self,
        precision: PrecisionInput | None,
        device: str,
        configured_plugin: Precision | None,
        strategy_precision_plugin: Precision | None
    ) -> Precision:
        # Applies plugin precedence rules, normalizes public precision, performs
        # CPU fallback, and constructs the corresponding numerical policy.
        if configured_plugin is not None and strategy_precision_plugin is not None:
            raise MisconfigurationError(
                "A precision plugin was configured through both Trainer plugins and the strategy. "
                "Configure it in exactly one place."
            )
        selected_plugin: Precision | None = (
            configured_plugin or strategy_precision_plugin
        )
        if selected_plugin is not None:
            if precision is not None:
                raise MisconfigurationError(
                    f"Received precision={precision!r} together with a precision plugin. "
                    "Configure either precision or the plugin, not both."
                )
            return selected_plugin
        normalized_precision: PrecisionType = self._normalize_precision(precision)
        if normalized_precision == "16-mixed" and device == "cpu":
            warnings.warn(
                "Trainer(accelerator='cpu', precision='16-mixed') is unsupported because CPU AMP "
                "does not support float16. Using precision='bf16-mixed' instead.",
                UserWarning,
                stacklevel=3
            )
            normalized_precision: PrecisionType = "bf16-mixed"
        match normalized_precision:
            case "16-true" | "bf16-true":
                return HalfPrecision(normalized_precision)
            case "32-true":
                return Precision()
            case "64-true":
                return DoublePrecision()
            case "16-mixed" | "bf16-mixed":
                return MixedPrecision(normalized_precision, device)
            case "transformer-engine":
                return TransformerEnginePrecision(weights_dtype=torch.bfloat16)
            case "transformer-engine-float16":
                return TransformerEnginePrecision(weights_dtype=torch.float16)

    def _normalize_precision(
        self,
        precision: PrecisionInput | None
    ) -> PrecisionType:
        # Converts numeric and compatibility aliases to canonical explicit names
        # and rejects bools or unsupported modes.
        if precision is None:
            return "32-true"
        if isinstance(precision, bool):
            raise ValueError(
                f"Precision {precision!r} is invalid. Use 64, 32, 16, 'bf16', "
                "or an explicit precision mode."
            )
        if not isinstance(precision, (int, str)):
            raise TypeError(
                f"precision must be an int, a string, or None, got {type(precision).__name__}."
            )
        aliases: dict[int | str, PrecisionType] = {
            64: "64-true",
            32: "32-true",
            16: "16-mixed",
            "64": "64-true",
            "32": "32-true",
            "16": "16-mixed",
            "bf16": "bf16-mixed"
        }
        if precision in aliases:
            if precision in {16, "16", "bf16"}:
                warnings.warn(
                    f"precision={precision!r} is supported for compatibility but discouraged. "
                    f"Use precision={aliases[precision]!r} instead.",
                    UserWarning,
                    stacklevel=3
                )
            return aliases[precision]
        supported: tuple[str, ...] = (
            "16-true",
            "16-mixed",
            "bf16-true",
            "bf16-mixed",
            "32-true",
            "64-true",
            "transformer-engine",
            "transformer-engine-float16"
        )
        if precision not in supported:
            raise ValueError(
                f"Precision {precision!r} is invalid. Allowed precision values: "
                f"{supported + (64, 32, 16, '64', '32', '16', 'bf16')}."
            )
        return cast(PrecisionType, precision)
