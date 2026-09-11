# This module:
# 1. Defines the scalar checkpoint ranking state used for monitored top-k and
#    unmonitored latest-artifact retention
# 2. Validates, snapshots, restores, serializes, and resumes ranking metadata
#    independently of storage side effects
#
# Design decisions:
# - Qualification is reduced through Strategy so all ranks agree whether a
#   monitored candidate enters the retained set
# - Scores are detached scalar CPU tensors, making ranking independent of
#   accelerator placement and autograd lifetime
# - NaN is converted to the worst possible score for the active direction; the
#   artifact remains representable without accidentally becoming best
# - Ranking mutations return the superseded path but never delete it. The
#   transaction layer prunes only after replacement persistence succeeds
# - snapshot and restore_snapshot provide in-process rollback, while state_dict
#   and load_state_dict provide strict cross-run continuation
#
# Author: Rahul Sawhney

from typing import Literal, cast

import syntheticmind
import torch

from syntheticmind.primitives.types import StateDict, StateValue

__all__: list[str] = []

type _RankingSnapshot = tuple[
    torch.Tensor | None,
    dict[str, torch.Tensor],
    str,
    torch.Tensor,
    torch.Tensor | None,
    str
]


class _CheckpointRanking:
    # Pure top-k state machine composed by the checkpoint transaction. It knows
    # scores and paths but has no filesystem, logger, or serialization access.
    def __init__(
        self,
        monitor: str | None,
        mode: Literal["min", "max"],
        save_top_k: int
    ) -> None:
        # Initializes an empty ranking with the direction-specific worst kth
        # value used before any candidate has been accepted.
        self.monitor: str | None = monitor
        self.mode: Literal["min", "max"] = mode
        self.save_top_k: int = save_top_k
        self.current_score: torch.Tensor | None = None
        self.best_k_models: dict[str, torch.Tensor] = {}
        self.kth_best_model_path: str = ""
        self.kth_value: torch.Tensor = self._initial_kth_value()
        self.best_model_score: torch.Tensor | None = None
        self.best_model_path: str = ""

    def qualifies(
        self,
        trainer: syntheticmind.Trainer,
        current: torch.Tensor | None
    ) -> bool:
        # Decides whether a synchronized scalar can enter the configured retained
        # set without mutating ranking state.
        if current is None or self.save_top_k == 0:
            return False
        if self.save_top_k == -1 or len(self.best_k_models) < self.save_top_k:
            return True
        if not self.kth_best_model_path:
            return True
        kth_value: torch.Tensor = self.best_k_models[self.kth_best_model_path]
        local_decision: bool = (
            bool(current < kth_value)
            if self.mode == "min"
            else bool(current > kth_value)
        )
        return trainer.strategy.reduce_boolean_decision(local_decision, all=False)

    def accept_ranked(
        self,
        current: torch.Tensor,
        filepath: str
    ) -> str | None:
        # Atomically installs a monitored candidate in logical state and returns
        # the prior kth path eligible for post-commit deletion.
        resolved_current: torch.Tensor = current.detach().cpu().squeeze()
        if bool(torch.isnan(resolved_current)):
            resolved_current: torch.Tensor = torch.tensor(
                torch.inf if self.mode == "min" else -torch.inf
            )
        delete_path: str | None = (
            self.kth_best_model_path
            if self.save_top_k > 0
            and len(self.best_k_models) >= self.save_top_k
            and self.kth_best_model_path
            else None
        )
        ranking: dict[str, torch.Tensor] = dict(self.best_k_models)
        if delete_path is not None:
            ranking.pop(delete_path, None)
        ranking[filepath] = resolved_current
        if self.mode == "min":
            best_model_path: str = min(ranking, key=ranking.__getitem__)
            kth_best_model_path: str = max(ranking, key=ranking.__getitem__)
        else:
            best_model_path: str = max(ranking, key=ranking.__getitem__)
            kth_best_model_path: str = min(ranking, key=ranking.__getitem__)
        self.current_score: torch.Tensor | None = resolved_current
        self.best_k_models: dict[str, torch.Tensor] = ranking
        self.best_model_path: str = best_model_path
        self.best_model_score: torch.Tensor | None = ranking[best_model_path]
        self.kth_best_model_path: str = kth_best_model_path
        self.kth_value: torch.Tensor = ranking[kth_best_model_path]
        return delete_path

    def accept_unmonitored(self, filepath: str) -> str | None:
        # Tracks the latest path for latest-only unmonitored retention.
        previous_path: str = self.best_model_path
        self.best_model_path: str = filepath
        return previous_path if self.save_top_k == 1 else None

    def reset(self) -> None:
        # Clears ranking paths after an incompatible directory change.
        self.current_score: torch.Tensor | None = None
        self.best_k_models: dict[str, torch.Tensor] = {}
        self.kth_best_model_path: str = ""
        self.kth_value: torch.Tensor = self._initial_kth_value()
        self.best_model_score: torch.Tensor | None = None
        self.best_model_path: str = ""

    def snapshot(self) -> _RankingSnapshot:
        # Captures an in-process rollback point before persistence begins.
        return (
            self.current_score,
            dict(self.best_k_models),
            self.kth_best_model_path,
            self.kth_value,
            self.best_model_score,
            self.best_model_path
        )

    def restore_snapshot(self, snapshot: _RankingSnapshot) -> None:
        # Reinstates the exact logical ranking after failed persistence.
        self.current_score: torch.Tensor | None = snapshot[0]
        self.best_k_models: dict[str, torch.Tensor] = snapshot[1]
        self.kth_best_model_path: str = snapshot[2]
        self.kth_value: torch.Tensor = snapshot[3]
        self.best_model_score: torch.Tensor | None = snapshot[4]
        self.best_model_path: str = snapshot[5]

    def state_dict(self) -> StateDict:
        # Serializes the full ranking trajectory for callback resumption.
        return {
            "monitor": self.monitor,
            "current_score": self.current_score,
            "best_k_models": cast(StateValue, self.best_k_models),
            "kth_best_model_path": self.kth_best_model_path,
            "kth_value": self.kth_value,
            "best_model_score": self.best_model_score,
            "best_model_path": self.best_model_path
        }

    def load_state_dict(self, state_dict: StateDict) -> None:
        # Validates scalar scores, path keys, and monitor identity before replacing
        # the live ranking state.
        monitor_value: StateValue | None = state_dict.get("monitor")
        if monitor_value != self.monitor:
            raise ValueError(
                f"Checkpoint monitor {monitor_value!r} does not match {self.monitor!r}."
            )
        current_score: torch.Tensor | None = self._optional_score(
            state_dict.get("current_score"),
            "current_score"
        )
        best_score: torch.Tensor | None = self._optional_score(
            state_dict.get("best_model_score"),
            "best_model_score"
        )
        kth_value: torch.Tensor | None = self._optional_score(
            state_dict.get("kth_value"),
            "kth_value"
        )
        if kth_value is None:
            raise TypeError("Checkpoint kth_value state must be scalar.")
        ranking_value: StateValue | None = state_dict.get("best_k_models")
        if not isinstance(ranking_value, dict):
            raise TypeError("Checkpoint best_k_models state must be a mapping.")
        ranking: dict[str, torch.Tensor] = {}
        path: str
        score_value: StateValue
        for path, score_value in ranking_value.items():
            if not isinstance(path, str):
                raise TypeError("Checkpoint ranking paths must be strings.")
            score: torch.Tensor | None = self._optional_score(
                score_value,
                f"best_k_models[{path!r}]"
            )
            if score is None:
                raise TypeError("Checkpoint ranking scores must be scalar.")
            ranking[path] = score
        self.current_score: torch.Tensor | None = current_score
        self.best_k_models: dict[str, torch.Tensor] = ranking
        self.kth_best_model_path: str = self._required_string(
            state_dict,
            "kth_best_model_path"
        )
        self.kth_value: torch.Tensor = kth_value
        self.best_model_score: torch.Tensor | None = best_score
        self.best_model_path: str = self._required_string(
            state_dict,
            "best_model_path"
        )

    def _initial_kth_value(self) -> torch.Tensor:
        # Returns the direction-specific identity for an empty retained set.
        return torch.tensor(torch.inf if self.mode == "min" else -torch.inf)

    def _optional_score(
        self,
        value: StateValue | None,
        name: str
    ) -> torch.Tensor | None:
        # Normalizes an optional Python or tensor scalar to a detached CPU tensor.
        if value is None:
            return None
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return value.detach().cpu().clone().squeeze()
        if type(value) is int or type(value) is float:
            return torch.tensor(float(value))
        raise TypeError(f"Checkpoint {name} state must be scalar or None.")

    def _required_string(self, state_dict: StateDict, name: str) -> str:
        # Reads one mandatory path field without coercing foreign checkpoint data.
        value: StateValue | None = state_dict.get(name)
        if not isinstance(value, str):
            raise TypeError(f"Checkpoint {name} state must be a string.")
        return value
