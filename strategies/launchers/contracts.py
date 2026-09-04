# This module:
# 1. Defines process start methods and the typed stage argument and result domains
#    transported through distributed launchers
# 2. Defines strict immutable snapshots for process-global execution state and
#    the complete rank-zero worker outcome reconciled into a dormant parent
#
# Design decisions:
# - Spawn and fork workers own live execution; the parent receives an explicit
#   snapshot rather than pretending its pre-launch Trainer ran the stage
# - WorkerOutput separates model, Trainer, callback, DataModule, precision,
#   metric, loader, optimizer, scheduler, signal, and global process state so
#   every continuation-critical participant crosses process ownership
# - Tensor values are permitted explicitly by strict Pydantic arbitrary-type
#   handling, while unknown fields and coercion remain forbidden
# - Random streams and deterministic backend settings travel together only as
#   process-global launch state; their runtime owners remain separate
#
# Author: Rahul Sawhney

from pathlib import Path
from typing import ClassVar, Literal

import torch
from pydantic import BaseModel, ConfigDict

from syntheticmind.rng.streams import _RandomStateSnapshot
from syntheticmind.primitives.types import (
    CheckpointDict,
    CheckpointValue,
    DataLoaderState,
    DataLoaderSource,
    ModelOutput,
    OptimizerState,
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
    # Process-global stochastic and deterministic state restored before a worker
    # enters Trainer execution and reconciled back into the parent afterward.
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
    # Complete rank-zero result of one multiprocessing-owned stage. Nonzero ranks
    # participate in collectives but do not publish a competing parent snapshot.
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
    train_epoch_metric_states_by_rank: list[CheckpointDict]
    train_optimizer_states: list[OptimizerState]
    train_scheduler_states: list[CheckpointValue]
    received_sigterm: bool
    is_last_batch: bool
    global_state: _GlobalState
