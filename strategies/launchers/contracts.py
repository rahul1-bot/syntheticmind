import random
from typing import ClassVar, Literal, cast

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from syntheticmind.utilities.types import (
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
type NumPyRandomState = tuple[str, np.ndarray, int, int, float]


class _GlobalState(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True
    )

    python_random_state: tuple[object, ...]
    numpy_random_state: NumPyRandomState
    torch_random_state: torch.Tensor
    torch_cuda_random_state: list[torch.Tensor]
    deterministic_algorithms: bool
    deterministic_warn_only: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool

    @classmethod
    def capture(cls) -> _GlobalState:
        return cls(
            python_random_state=cast(tuple[object, ...], random.getstate()),
            numpy_random_state=cast(NumPyRandomState, np.random.get_state()),
            torch_random_state=torch.random.get_rng_state(),
            torch_cuda_random_state=(
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic
        )

    def restore(self) -> None:
        random.setstate(self.python_random_state)
        np.random.set_state(self.numpy_random_state)
        torch.random.set_rng_state(self.torch_random_state)
        torch.cuda.set_rng_state_all(self.torch_cuda_random_state)
        torch.use_deterministic_algorithms(
            self.deterministic_algorithms,
            warn_only=self.deterministic_warn_only
        )
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        torch.backends.cudnn.deterministic = self.cudnn_deterministic


class _WorkerOutput(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True
    )

    result: StageResult
    module_state_dict: StateDict
    trainer_state_dict: TrainerStateDict
    should_stop: bool
    callback_metrics: dict[str, torch.Tensor]
    logged_metrics: dict[str, torch.Tensor]
    progress_bar_metrics_by_stage: dict[RunningStage, dict[str, float]]
    last_progress_bar_stage: RunningStage | None
    callback_states: list[StateDict]
    datamodule_state: StateDict | None
    precision_plugin_state: StateDict
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    total_train_batches: int | None
    total_val_batches: list[int | None] | None
    total_test_batches: list[int | None] | None
    total_predict_batches: list[int | None] | None
    train_dataloader_states: list[dict[str, object]]
    train_dataloader_state_active: bool
    train_dataloader_batches_completed: int
    train_total_batch_idx: int
    received_sigterm: bool
    is_last_batch: bool
    global_state: _GlobalState
