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

type BatchScalar = bool | int | float | str | bytes | None


@runtime_checkable
class DeviceTransferable(Protocol):
    def to(self, device: torch.device, non_blocking: bool = False) -> DeviceTransferable:
        ...


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

type StepArgument = Batch | int

type StepOutput = dict[str, torch.Tensor | float]

type TrainingStepOutput = torch.Tensor | StepOutput | None

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

type OptimizerParameterGroup = dict[
    str,
    OptimizerStateValue | torch.nn.Parameter | list[torch.nn.Parameter]
]

type DistributedCheckpointState = dict[str, ModelState | OptimizerState]

type OptimizerClosureResult = float | torch.Tensor | None

type OptimizerClosure = Callable[[], OptimizerClosureResult]

type OptimizerStepOutput = OptimizerClosureResult

type OptimizerStepValue = bool | int | float | str | torch.Tensor | None

type OptimizerStepOptions = dict[str, OptimizerStepValue]

type BackwardInputs = torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor] | None

type MapLocation = (
    Callable[[torch.types.Storage, str], torch.types.Storage | None]
    | torch.device
    | str
    | dict[str, str]
    | None
)
