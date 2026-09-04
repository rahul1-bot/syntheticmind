# This module:
# 1. Declares the closed Literal vocabularies the harness accepts for trainer stages,
#    running stages, precision policies, gradient-clip algorithms, scheduler stepping
#    intervals, and metric reductions
# 2. Declares the recursive structural aliases for batches, step outputs, model outputs,
#    hyperparameters, mutable component state, and checkpoint payloads
# 3. Defines the DeviceTransferable protocol used to duck-type device movement for
#    tensor-like containers that are not raw torch.Tensor objects
#
# Design decisions:
# - PEP 695 `type` statements are used for every alias because they evaluate lazily,
#   which lets Batch, ModelOutput, and the other recursive unions reference themselves
#   without string forward references
# - The stage and policy vocabularies live here, in one importable location, so invalid
#   values fail at configuration boundaries instead of deep inside a running loop
# - DeviceTransferable is runtime_checkable so batch transfer can test hasattr-style
#   conformance at runtime without importing any concrete container class
# - Each domain (batch, model, hyperparameter, state, checkpoint) keeps its own scalar
#   and value family even where the unions currently coincide, so one contract can
#   tighten later without silently rewriting the others
# - numpy arrays are admitted only in checkpoint payloads, where serialized RNG state
#   legitimately contains them; runtime batch and state contracts stay tensor-only
#
# Author: Rahul Sawhney

from collections.abc import Callable, Iterable
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
    "BatchScalar",
    "DeviceTransferable",
    "Batch",
    "DataLoaderSource",
    "StepArgument",
    "StepOutput",
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
    "DistributedCheckpointState",
    "OptimizerClosure",
    "OptimizerClosureResult",
    "OptimizerStepOutput",
    "OptimizerStepValue",
    "BackwardInputs",
    "MapLocation"
]

type TrainerStage = Literal["fit", "validate", "test", "predict"]

type RunningStage = Literal["training", "validating", "testing", "predicting", "sanity_checking"]

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

type SchedulerInterval = Literal["step", "epoch"]

type Reduction = Literal["mean", "sum", "min", "max"]

type BatchScalar = bool | int | float | str | None


@runtime_checkable
class DeviceTransferable(Protocol):
    # Structural protocol for objects that can move themselves between devices.
    # Batch transfer checks against this protocol as a fallback after the concrete
    # tensor, dict, list, and tuple branches, which lets custom tensor-like containers
    # participate in device placement without the harness importing their classes.
    def to(self, device: torch.device, non_blocking: bool = False) -> DeviceTransferable:
        # Returns this object placed on the requested device. non_blocking mirrors the
        # torch.Tensor.to keyword so asynchronous host-to-device copies stay possible.
        ...


type Batch = (
    BatchScalar
    | torch.Tensor
    | DeviceTransferable
    | dict[str, Batch]
    | list[Batch]
    | tuple[Batch, ...]
)

type DataLoaderSource = (
    Iterable[Batch]
    | dict[str, DataLoaderSource]
    | list[DataLoaderSource]
    | tuple[DataLoaderSource, ...]
)

type StepArgument = Batch | int

type StepOutput = dict[str, torch.Tensor | float]

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
    | dict[object, GatherData]
    | list[GatherData]
    | tuple[GatherData, ...]
)

type GatherResult = (
    torch.Tensor
    | dict[object, GatherResult]
    | list[GatherResult]
    | tuple[GatherResult, ...]
)

type HyperparameterScalar = bool | int | float | str | None

type HyperparameterValue = object

type HyperparameterDict = dict[object, HyperparameterValue]

type StateScalar = bool | int | float | str | None

type StateValue = StateScalar | torch.Tensor | list[StateValue] | tuple[StateValue, ...] | dict[str, StateValue]

type StateDict = dict[str, StateValue]

type TrainerStateDict = dict[str, StateScalar]

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

type DistributedCheckpointState = dict[str, ModelState | OptimizerState]

type OptimizerClosureResult = float | torch.Tensor | None

type OptimizerClosure = Callable[[], OptimizerClosureResult]

type OptimizerStepOutput = OptimizerClosureResult

type OptimizerStepValue = bool | int | float | str | torch.Tensor | None

type BackwardInputs = torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor] | None

type MapLocation = (
    Callable[[torch.types.Storage, str], torch.types.Storage | None]
    | torch.device
    | str
    | dict[str, str]
    | None
)
