# This module:
# 1. Exposes the checkpoint storage, cluster environment, layer synchronization,
#    and precision plugin families accepted by Trainer construction
# 2. Defines Plugin as the closed union of those extension boundaries
#
# Design decisions:
# - Plugin selection and compatibility remain with the Trainer accelerator
#   connector because validity depends on accelerator, Strategy, devices, and
#   other supplied plugins
# - Package exports preserve concrete ownership and do not introduce a registry,
#   resolver, or generic plugin base class
#
# Author: Rahul Sawhney

from syntheticmind.plugins.environments import ClusterEnvironment, LocalEnvironment
from syntheticmind.plugins.checkpoint_io import (
    AsyncCheckpointIO,
    CheckpointIO,
    CheckpointPath,
    CheckpointStorageOptions,
    TorchCheckpointIO
)
from syntheticmind.plugins.layer_sync import LayerSync, TorchSyncBatchNorm
from syntheticmind.plugins.precision.precision import Precision

__all__: list[str] = [
    "AsyncCheckpointIO",
    "CheckpointIO",
    "CheckpointPath",
    "CheckpointStorageOptions",
    "ClusterEnvironment",
    "LayerSync",
    "LocalEnvironment",
    "Plugin",
    "Precision",
    "TorchCheckpointIO",
    "TorchSyncBatchNorm"
]

type Plugin = Precision | CheckpointIO | ClusterEnvironment | LayerSync
