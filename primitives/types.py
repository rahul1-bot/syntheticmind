# This module:
# 1. Defines the recursive value vocabulary shared across model, data, Trainer,
#    optimization, checkpoint, launcher, logger, and strategy boundaries
# 2. Defines DeviceTransferable, the structural contract through which custom
#    batch containers participate in device placement
#
# Design decisions:
# - Recursive aliases name every accepted leaf and container explicitly instead
#   of escaping through Any or object, so a boundary states what can actually
#   cross it and nested checkpoint payloads remain reviewable
# - Types live here only when at least two semantic subsystems consume them;
#   implementation-local aliases stay beside their owner
# - StateDict describes lightweight component state, CheckpointDict describes
#   the complete serialized value domain, and OptimizerState preserves integer
#   parameter identifiers; these domains are intentionally not interchangeable
# - Batch and ModelOutput are distinct because batches may contain custom
#   device-transferable objects and bytes, while model outputs must remain
#   values the loops can route, gather, or return to clients
# - The module contains vocabulary only: no construction, validation, I/O, or
#   runtime policy is permitted here
#
# Author: Rahul Sawhney

from argparse import Namespace
from collections.abc import Callable, Iterable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import torch
from torch.distributed.checkpoint.state_dict import ValueType

__all__: list[str] = [
    "TrainerStage",
    "RunningStage",
    "PrecisionInput",
    "PrecisionType",
    "ClipAlgorithm",
    "SchedulerInterval",
    "Reduction",
    "TensorReducer",
    "BatchScalar",
    "DeviceTransferable",
    "Batch",
    "DatasetSource",
    "DataLoaderSource",
    "DataLoaderState",
    "StepArgument",
    "StepOutput",
    "TrainingStepOutput",
    "ModelScalar",
    "ModelOutput",
    "GatherData",
    "GatherResult",
    "HyperparameterScalar",
    "HyperparameterValue",
    "HyperparameterDict",
    "StateScalar",
    "StateValue",
    "StateDict",
    "TrainerStateDict",
    "CheckpointScalar",
    "CheckpointValue",
    "CheckpointDict",
    "ModelState",
    "OptimizerState",
    "OptimizerStateKey",
    "OptimizerStateValue",
    "OptimizerParameterGroup",
    "DistributedCheckpointState",
    "OptimizerClosure",
    "OptimizerClosureResult",
    "OptimizerStepOutput",
    "OptimizerStepValue",
    "OptimizerStepOptions",
    "BackwardInputs",
    "MapLocation"
]

# Public lifecycle names accepted by Trainer entry points, callbacks, profilers,
# and process launchers.
type TrainerStage = Literal["fit", "validate", "test", "predict"]

# Internal execution phases distinguish training and sanity validation from the
# public entry point that owns them.
type RunningStage = Literal["training", "validating", "testing", "predicting", "sanity_checking"]

# User-facing precision input accepts the established numeric and string forms;
# PrecisionType is the canonical resolved policy stored after configuration.
type PrecisionInput = (
    Literal[64, 32, 16]
    | Literal[
        "64",
        "32",
        "16",
        "bf16",
        "16-true",
        "16-mixed",
        "bf16-true",
        "bf16-mixed",
        "32-true",
        "64-true",
        "transformer-engine",
        "transformer-engine-float16"
    ]
)

type PrecisionType = Literal[
    "16-true",
    "16-mixed",
    "bf16-true",
    "bf16-mixed",
    "32-true",
    "64-true",
    "transformer-engine",
    "transformer-engine-float16"
]

type ClipAlgorithm = Literal["norm", "value"]

# Scheduler and metric reduction vocabularies cross core, loop, callback, and
# strategy boundaries and therefore remain framework primitives.
type SchedulerInterval = Literal["step", "epoch"]

type Reduction = Literal["mean", "sum", "min", "max"]

type TensorReducer = Callable[
    [torch.Tensor, torch.distributed.ProcessGroup | None, Reduction],
    torch.Tensor
]

type BatchScalar = bool | int | float | str | bytes | None


@runtime_checkable
class DeviceTransferable(Protocol):
    # Structural device-transfer contract for client batch containers. A type
    # participates without inheriting from SyntheticMind when it exposes a
    # Tensor-compatible to(device, non_blocking=...) operation.
    def to(self, device: torch.device, non_blocking: bool = False) -> DeviceTransferable:
        # Returns the same logical value placed on the requested device. CUDA
        # batches may receive non_blocking=True; other backends receive False.
        ...


# Recursive batch and dataloader domains preserve nested dict, list, and tuple
# topology from client data factories through CombinedLoader and every loop.
type Batch = (
    BatchScalar
    | torch.Tensor
    | DeviceTransferable
    | dict[str, Batch]
    | list[Batch]
    | tuple[Batch, ...]
)

type DatasetSource = (
    torch.utils.data.Dataset[Batch]
    | Iterable[torch.utils.data.Dataset[Batch]]
    | dict[str, DatasetSource]
    | list[DatasetSource]
    | tuple[DatasetSource, ...]
)

type DataLoaderSource = (
    Iterable[Batch]
    | dict[str, DataLoaderSource]
    | list[DataLoaderSource]
    | tuple[DataLoaderSource, ...]
)

type DataLoaderState = dict[str, CheckpointValue]

# StepArgument accounts for the optional dataloader index appended by sequential
# multi-loader execution, while step results remain scalar metric mappings.
type StepArgument = Batch | int

type StepOutput = dict[str, torch.Tensor | float]

type TrainingStepOutput = torch.Tensor | StepOutput | None

# ModelOutput is the recursive value domain returned from prediction and hook
# dispatch; collective gathering has a narrower numeric domain and a transformed
# result whose leaves are always tensors.
type ModelScalar = bool | int | float | str | None

type ModelOutput = (
    ModelScalar
    | torch.Tensor
    | list[ModelOutput]
    | tuple[ModelOutput, ...]
    | dict[str, ModelOutput]
)

type GatherData = (
    bool
    | int
    | float
    | np.ndarray
    | torch.Tensor
    | dict[str | int, GatherData]
    | list[GatherData]
    | tuple[GatherData, ...]
)

type GatherResult = (
    torch.Tensor
    | dict[str | int, GatherResult]
    | list[GatherResult]
    | tuple[GatherResult, ...]
)

# Hyperparameters include portable constructor values plus the richer objects
# that capture may encounter before logger normalization or checkpoint writing.
type HyperparameterScalar = bool | int | float | str | None

type HyperparameterValue = (
    HyperparameterScalar
    | bytes
    | Path
    | date
    | datetime
    | timedelta
    | torch.dtype
    | torch.device
    | torch.Tensor
    | torch.nn.Module
    | np.ndarray
    | np.generic
    | Namespace
    | dict[str, HyperparameterValue]
    | list[HyperparameterValue]
    | tuple[HyperparameterValue, ...]
    | set[HyperparameterValue]
    | frozenset[HyperparameterValue]
)

type HyperparameterDict = dict[str, HyperparameterValue]

# Component state is string-keyed and intentionally narrower than a complete
# checkpoint payload. TrainerStateDict restricts live progress to scalar values.
type StateScalar = bool | int | float | str | None

type StateValue = StateScalar | torch.Tensor | list[StateValue] | tuple[StateValue, ...] | dict[str, StateValue]

type StateDict = dict[str, StateValue]

type TrainerStateDict = dict[str, StateScalar]

# Checkpoint values admit tensors, NumPy arrays, and recursively nested
# containers because a full trajectory aggregates state from every participant.
type CheckpointScalar = StateScalar

type CheckpointValue = (
    CheckpointScalar
    | torch.Tensor
    | np.ndarray
    | list[CheckpointValue]
    | tuple[CheckpointValue, ...]
    | dict[str, CheckpointValue]
)

type CheckpointDict = dict[str, CheckpointValue]

# Distributed checkpoint state preserves PyTorch's model ValueType and the
# integer parameter identifiers used by optimizer state dictionaries.
type ModelState = dict[str, ValueType]

type OptimizerStateKey = str | int

type OptimizerStateValue = (
    bool
    | int
    | float
    | str
    | None
    | torch.Tensor
    | list[OptimizerStateValue]
    | tuple[OptimizerStateValue, ...]
    | dict[OptimizerStateKey, OptimizerStateValue]
)

type OptimizerState = dict[str, OptimizerStateValue]

type OptimizerParameterGroup = dict[
    str,
    OptimizerStateValue | torch.nn.Parameter | list[torch.nn.Parameter]
]

type DistributedCheckpointState = dict[str, ModelState | OptimizerState]

# Optimizer closures and step keyword values form the execution seam shared by
# Module, SyntheticOptimizer, precision plugins, strategies, and loops.
type OptimizerClosureResult = float | torch.Tensor | None

type OptimizerClosure = Callable[[], OptimizerClosureResult]

type OptimizerStepOutput = OptimizerClosureResult

type OptimizerStepValue = bool | int | float | str | torch.Tensor | None

type OptimizerStepOptions = dict[str, OptimizerStepValue]

type BackwardInputs = torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor] | None

# MapLocation mirrors torch.load while keeping the accepted callable, device,
# string, mapping, and default forms explicit at checkpoint APIs.
type MapLocation = (
    Callable[[torch.types.Storage, str], torch.types.Storage | None]
    | torch.device
    | str
    | dict[str, str]
    | None
)
