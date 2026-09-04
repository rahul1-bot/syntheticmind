from pathlib import Path
from typing import ClassVar, Literal

import torch
from pydantic import BaseModel, ConfigDict

from syntheticmind.randomness import _RandomStateSnapshot
from syntheticmind.types import (
    CheckpointDict,
    DataLoaderState,
    DataLoaderSource,
    ModelOutput,
    RunningStage,
    StateDict,
    TrainerStateDict
)

__all__: list[str] = ["ProcessStartMethod"]

type ProcessStartMethod = Literal["popen", "spawn", "fork", "forkserver"]
type StageResult = (
    None
    | list[dict[str, float]]
    | list[ModelOutput]
    | list[list[ModelOutput]]
)
type StageArgument = DataLoaderSource | str | Path | bool | None
type StageArguments = dict[str, StageArgument]


class _GlobalState(BaseModel):
    schema_version: Literal[1] = 1
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        strict=True,
        validate_default=True
    )

    random_state: _RandomStateSnapshot
    deterministic_algorithms: bool
    deterministic_warn_only: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool

class _WorkerOutput(BaseModel):
    schema_version: Literal[1] = 1
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        strict=True,
        validate_default=True
    )

    result: StageResult
    module_state_dict: StateDict
    trainer_state_dict: TrainerStateDict
    should_stop: bool
    callback_metrics: dict[str, torch.Tensor]
    logged_metrics: dict[str, torch.Tensor]
    progress_bar_metrics_by_stage: dict[RunningStage, dict[str, float]]
    last_progress_bar_stage: RunningStage | None
    hyperparameters_logged: bool
    callback_states: list[StateDict]
    datamodule_state: StateDict | None
    precision_plugin_state: StateDict
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    total_train_batches: int | None
    total_val_batches: list[int | None] | None
    total_test_batches: list[int | None] | None
    total_predict_batches: list[int | None] | None
    train_dataloader_states: list[DataLoaderState]
    train_dataloader_state_active: bool
    train_dataloader_batches_completed: int
    train_total_batch_idx: int
    train_epoch_rng_state: CheckpointDict | None
    received_sigterm: bool
    is_last_batch: bool
    global_state: _GlobalState
