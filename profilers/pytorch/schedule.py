from collections.abc import Callable
from typing import ClassVar, Literal, cast

from torch.profiler import ProfilerAction

from syntheticmind.primitives.exceptions import MisconfigurationError
from syntheticmind.primitives.types import TrainerStage

__all__: list[str] = []

type _ProfilerStepName = Literal[
    "training_step",
    "validation_step",
    "test_step",
    "predict_step"
]
type _ProfilerScheduleFunction = Callable[[int], ProfilerAction]


class _ProfilerSchedule:
    _allowed_actions: ClassVar[frozenset[ProfilerAction]] = frozenset(
        {
            ProfilerAction.NONE,
            ProfilerAction.WARMUP,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE
        }
    )

    def __init__(self, schedule: _ProfilerScheduleFunction) -> None:
        self._schedule: _ProfilerScheduleFunction = schedule
        self._trainer_stage: TrainerStage | None = None
        self._current_step_name: _ProfilerStepName | None = None
        self._step_counts: dict[_ProfilerStepName, int] = self._initial_counts()
        self._finished_steps: set[_ProfilerStepName] = set()
        self._previous_action: ProfilerAction | None = None

    def setup(self, trainer_stage: TrainerStage) -> None:
        self._trainer_stage: TrainerStage | None = trainer_stage
        self._current_step_name: _ProfilerStepName | None = None
        self._step_counts: dict[_ProfilerStepName, int] = self._initial_counts()
        self._finished_steps: set[_ProfilerStepName] = set()
        self._previous_action: ProfilerAction | None = None

    def prepare_step(self, action_name: str) -> None:
        step_name: _ProfilerStepName = self._resolve_step_name(action_name)
        if (
            self._trainer_stage == "fit"
            and step_name == "validation_step"
            and self._step_counts["training_step"] == 0
        ):
            self._current_step_name: _ProfilerStepName | None = None
            return
        self._current_step_name: _ProfilerStepName | None = step_name

    def next_action(self, profiler_step_number: int) -> ProfilerAction:
        del profiler_step_number
        step_name: _ProfilerStepName | None = self._current_step_name
        if step_name is None or step_name in self._finished_steps:
            return ProfilerAction.NONE
        step_number: int = self._step_counts[step_name] + 1
        self._step_counts[step_name] = step_number
        action: ProfilerAction = self._schedule(step_number)
        if action not in self._allowed_actions:
            raise MisconfigurationError(
                "A PyTorch profiler schedule must return NONE, WARMUP, RECORD, "
                f"or RECORD_AND_SAVE, got {action!r}."
            )
        if (
            self._previous_action == ProfilerAction.RECORD
            and action == ProfilerAction.WARMUP
        ):
            action: ProfilerAction = ProfilerAction.RECORD
        self._previous_action: ProfilerAction | None = action
        if action == ProfilerAction.RECORD_AND_SAVE:
            self._finished_steps.add(step_name)
        return action

    @property
    def step_number(self) -> int:
        if self._current_step_name is None:
            return 0
        return self._step_counts[self._current_step_name]

    def _resolve_step_name(self, action_name: str) -> _ProfilerStepName:
        action_leaf: str = action_name.rsplit(".", 1)[-1]
        if action_leaf not in self._step_counts:
            raise ValueError(f"Action {action_name!r} is not a profiled stage step.")
        return cast(_ProfilerStepName, action_leaf)

    def _initial_counts(self) -> dict[_ProfilerStepName, int]:
        return {
            "training_step": 0,
            "validation_step": 0,
            "test_step": 0,
            "predict_step": 0
        }
