from typing import Literal, override

from syntheticmind.primitives.types import RunningStage, TrainerStage, TrainerStateDict

__all__: list[str] = ["TrainerState"]

type TrainerStatus = Literal["initializing", "running", "finished", "interrupted"]


class TrainerState:
    def __init__(self) -> None:
        self._status: TrainerStatus = "initializing"
        self._stage: TrainerStage | None = None
        self._running_stage: RunningStage | None = None
        self._current_epoch: int = 0
        self._global_step: int = 0
        self._should_stop: bool = False

    @property
    def status(self) -> TrainerStatus:
        return self._status

    @property
    def stage(self) -> TrainerStage | None:
        return self._stage

    @property
    def running_stage(self) -> RunningStage | None:
        return self._running_stage

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    @should_stop.setter
    def should_stop(self, value: bool) -> None:
        self._should_stop: bool = value

    def set_stage(self, stage: TrainerStage) -> None:
        self._stage: TrainerStage | None = stage
        self._status: TrainerStatus = "running"

    def set_running_stage(self, running_stage: RunningStage | None) -> None:
        self._running_stage: RunningStage | None = running_stage

    def increment_epoch(self) -> None:
        self._current_epoch: int = self._current_epoch + 1

    def increment_global_step(self) -> None:
        self._global_step: int = self._global_step + 1

    def finish(self) -> None:
        if self._status != "interrupted":
            self._status: TrainerStatus = "finished"
        self._stage: TrainerStage | None = None
        self._running_stage: RunningStage | None = None

    def interrupt(self) -> None:
        self._status: TrainerStatus = "interrupted"

    def reset_on_run(self) -> None:
        self._should_stop: bool = False
        self._current_epoch: int = 0
        self._global_step: int = 0

    def state_dict(self) -> TrainerStateDict:
        state_dict: TrainerStateDict = {
            "status": self._status,
            "stage": self._stage,
            "current_epoch": self._current_epoch,
            "global_step": self._global_step
        }
        return state_dict

    def load_state_dict(self, state_dict: TrainerStateDict) -> None:
        current_epoch: int | str | float | bool | None = state_dict.get("current_epoch", 0)
        global_step: int | str | float | bool | None = state_dict.get("global_step", 0)
        self._current_epoch: int = int(current_epoch if current_epoch is not None else 0)
        self._global_step: int = int(global_step if global_step is not None else 0)

    @override
    def __repr__(self) -> str:
        return (
            f"TrainerState("
            f"status={self._status!r}, "
            f"stage={self._stage!r}, "
            f"current_epoch={self._current_epoch}, "
            f"global_step={self._global_step})"
        )
