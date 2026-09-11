# This module:
# 1. Defines TrainerState, the live stage, running phase, epoch, optimizer-step,
#    cooperative-stop, completion, and interruption state of one Trainer
#
# Design decisions:
# - Public Trainer stage and internal running stage are separate because fit owns
#   both training and validation phases without changing its outer entry point
# - global_step counts completed managed optimizer mutations rather than batches
# - finish preserves interrupted status while clearing active stage identities,
#   keeping post-failure inspection truthful
# - reset_on_run clears trajectory counters only for a genuinely new run;
#   checkpoint restoration applies saved epoch and step afterward
# - state_dict includes status and stage for worker reconciliation, while
#   load_state_dict applies only durable trajectory counters and lets the owning
#   stage transaction restore live status explicitly
#
# Author: Rahul Sawhney

from typing import Literal, override

from syntheticmind.primitives.types import RunningStage, TrainerStage, TrainerStateDict

__all__: list[str] = ["TrainerState"]

type TrainerStatus = Literal["initializing", "running", "finished", "interrupted"]


class TrainerState:
    # Mutable runtime state shared by Trainer, loops, callbacks, checkpointing,
    # and multiprocessing reconciliation.
    def __init__(self) -> None:
        # Initializes a detached Trainer before any public stage claims runtime
        # identity or restored trajectory coordinates.
        self._status: TrainerStatus = "initializing"
        self._stage: TrainerStage | None = None
        self._running_stage: RunningStage | None = None
        self._current_epoch: int = 0
        self._global_step: int = 0
        self._should_stop: bool = False

    @property
    def status(self) -> TrainerStatus:
        # Returns initialization, execution, completion, or interruption state.
        return self._status

    @property
    def stage(self) -> TrainerStage | None:
        # Returns the active public fit, validate, test, or predict transaction.
        return self._stage

    @property
    def running_stage(self) -> RunningStage | None:
        # Returns the active inner training, evaluation, prediction, or sanity
        # phase within the public transaction.
        return self._running_stage

    @property
    def current_epoch(self) -> int:
        # Returns the count of completed training-loader epochs or the restored
        # partial epoch coordinate.
        return self._current_epoch

    @property
    def global_step(self) -> int:
        # Returns the count of completed managed optimizer mutations.
        return self._global_step

    @property
    def should_stop(self) -> bool:
        # Returns the cooperative stop request shared by callbacks and loops.
        return self._should_stop

    @should_stop.setter
    def should_stop(self, value: bool) -> None:
        # Applies already validated Trainer-level cooperative control state.
        self._should_stop: bool = value

    def set_stage(self, stage: TrainerStage) -> None:
        # Enters one public Trainer stage and marks execution running.
        self._stage: TrainerStage | None = stage
        self._status: TrainerStatus = "running"

    def set_running_stage(self, running_stage: RunningStage | None) -> None:
        # Selects the active inner phase such as training or sanity validation.
        self._running_stage: RunningStage | None = running_stage

    def increment_epoch(self) -> None:
        # Advances only after a true training-loader epoch completes.
        self._current_epoch: int = self._current_epoch + 1

    def increment_global_step(self) -> None:
        # Advances only after one managed optimizer mutation completes.
        self._global_step: int = self._global_step + 1

    def finish(self) -> None:
        # Marks successful completion unless interruption already occurred and
        # clears public and inner stage identity.
        if self._status != "interrupted":
            self._status: TrainerStatus = "finished"
        self._stage: TrainerStage | None = None
        self._running_stage: RunningStage | None = None

    def interrupt(self) -> None:
        # Records process, signal, or execution interruption without clearing
        # inspectable stage context immediately.
        self._status: TrainerStatus = "interrupted"

    def reset_on_run(self) -> None:
        # Starts a new trajectory at epoch and optimizer step zero with no stop
        # request; resume applies checkpoint counters afterward.
        self._should_stop: bool = False
        self._current_epoch: int = 0
        self._global_step: int = 0

    def state_dict(self) -> TrainerStateDict:
        # Serializes worker and checkpoint-visible state into the scalar domain.
        state_dict: TrainerStateDict = {
            "status": self._status,
            "stage": self._stage,
            "current_epoch": self._current_epoch,
            "global_step": self._global_step
        }
        return state_dict

    def load_state_dict(self, state_dict: TrainerStateDict) -> None:
        # Restores durable epoch and optimizer-step coordinates; the caller owns
        # live status and stage transition.
        current_epoch: int | str | float | bool | None = state_dict.get("current_epoch", 0)
        global_step: int | str | float | bool | None = state_dict.get("global_step", 0)
        self._current_epoch: int = int(current_epoch if current_epoch is not None else 0)
        self._global_step: int = int(global_step if global_step is not None else 0)

    @override
    def __repr__(self) -> str:
        # Reports observable lifecycle and trajectory coordinates without live
        # component references.
        return (
            f"TrainerState("
            f"status={self._status!r}, "
            f"stage={self._stage!r}, "
            f"current_epoch={self._current_epoch}, "
            f"global_step={self._global_step})"
        )
