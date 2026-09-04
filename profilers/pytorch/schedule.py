# This module:
# 1. Defines the stage-aware adapter between SyntheticMind step actions and a
#    client torch.profiler schedule function
# 2. Tracks independent training, validation, test, and prediction schedule
#    progress within one Trainer stage
#
# Design decisions:
# - The adapter ignores torch.profiler's shared step number and owns one counter
#   per ML step concept, preventing validation batches from consuming a training
#   schedule and vice versa
# - Fit-time validation before any training step is identified as sanity
#   validation and excluded from schedule progress
# - A step family becomes finished after RECORD_AND_SAVE, so later batches do not
#   begin an accidental second trace cycle in the same Trainer stage
# - A RECORD-to-WARMUP regression is held at RECORD because returning to warmup
#   while an active trace is being accumulated loses the intended capture
# - Schedule outputs are restricted to the exact ProfilerAction vocabulary and
#   fail at the adapter boundary when a client callable returns foreign state
#
# Author: Rahul Sawhney

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
    # Private ML-stage schedule state composed by PyTorchProfiler when a client
    # supplies torch.profiler's schedule callable.
    _allowed_actions: ClassVar[frozenset[ProfilerAction]] = frozenset(
        {
            ProfilerAction.NONE,
            ProfilerAction.WARMUP,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE
        }
    )

    def __init__(self, schedule: _ProfilerScheduleFunction) -> None:
        # Stores the client schedule and initializes detached stage counters.
        self._schedule: _ProfilerScheduleFunction = schedule
        self._trainer_stage: TrainerStage | None = None
        self._current_step_name: _ProfilerStepName | None = None
        self._step_counts: dict[_ProfilerStepName, int] = self._initial_counts()
        self._finished_steps: set[_ProfilerStepName] = set()
        self._previous_action: ProfilerAction | None = None

    def setup(self, trainer_stage: TrainerStage) -> None:
        # Resets every step family for a new Trainer stage.
        self._trainer_stage: TrainerStage | None = trainer_stage
        self._current_step_name: _ProfilerStepName | None = None
        self._step_counts: dict[_ProfilerStepName, int] = self._initial_counts()
        self._finished_steps: set[_ProfilerStepName] = set()
        self._previous_action: ProfilerAction | None = None

    def prepare_step(self, action_name: str) -> None:
        # Selects the ML step family about to advance, excluding fit sanity
        # validation before the first training step.
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
        # Advances the selected family, validates the client decision, and marks
        # completed capture cycles without using torch's shared counter.
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
        # Returns the selected family's current count for torch profiler state.
        if self._current_step_name is None:
            return 0
        return self._step_counts[self._current_step_name]

    def _resolve_step_name(self, action_name: str) -> _ProfilerStepName:
        # Maps HookDispatcher's qualified action label onto the closed step
        # vocabulary accepted by the schedule.
        action_leaf: str = action_name.rsplit(".", 1)[-1]
        if action_leaf not in self._step_counts:
            raise ValueError(f"Action {action_name!r} is not a profiled stage step.")
        return cast(_ProfilerStepName, action_leaf)

    def _initial_counts(self) -> dict[_ProfilerStepName, int]:
        # Creates an independent mutable counter mapping for one stage.
        return {
            "training_step": 0,
            "validation_step": 0,
            "test_step": 0,
            "predict_step": 0
        }
