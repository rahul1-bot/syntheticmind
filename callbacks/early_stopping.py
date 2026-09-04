# This module:
# 1. Terminates training when a monitored validation metric stops improving for
#    a configured number of consecutive validation passes
# 2. Persists its patience counter and best score into checkpoints so a resumed
#    run continues the plateau measurement instead of restarting it
#
# Design decisions:
# - Sanity-check validation passes are excluded entirely, so a bounded sanity
#   run can neither consume patience nor establish a misleading best score
# - min_delta defines the smallest change that counts as improvement, so noise
#   within the metric's natural spread cannot reset the patience counter
# - strict=True turns a missing monitor key into an immediate configuration
#   error listing the available metric names, which catches monitor-name typos
#   that would otherwise disable early stopping silently
# - The stop decision passes through the strategy's boolean-decision reduction,
#   so under distributed execution every rank agrees to stop and no rank
#   continues training alone
# - The stop is requested cooperatively through the trainer state's stop flag;
#   the fit loop honors it at the epoch boundary
# - The state key is qualified by monitor and mode, so two instances watching
#   different metrics checkpoint their state independently
#
# Author: Rahul Sawhney

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, override

import torch
from loguru import logger as log

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.exceptions import MisconfigurationError
from syntheticmind.types import StateDict

if TYPE_CHECKING:
    from syntheticmind.trainer.trainer import Trainer

__all__: list[str] = ["EarlyStopping"]


class EarlyStopping(Callback):
    # Patience-based training termination on validation plateau. Tracks the
    # best monitored value seen so far and counts consecutive validation
    # passes without an improvement of at least min_delta; when the count
    # reaches patience, the trainer's cooperative stop flag is raised.
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: Literal["min", "max"] = "min",
        min_delta: float = 0.0,
        strict: bool = True
    ) -> None:
        # Binds the stopping criterion: the metric to watch, the plateau
        # length that triggers the stop, the improvement direction, the
        # minimum change that counts as improvement, and whether a missing
        # monitor is an error. The best score starts at the identity element
        # for the chosen direction so the first observation always improves.
        #
        # Args:
        #     monitor: Metric key watched in the trainer's callback metrics
        #         after each validation pass. Default: ``"val_loss"``.
        #     patience: Number of consecutive monitored evaluations without
        #         improvement tolerated before the stop is requested.
        #         Default: ``10``.
        #     mode: Improvement direction, ``"min"`` or ``"max"``.
        #         Default: ``"min"``.
        #     min_delta: Minimum absolute change over the best score that
        #         counts as improvement. Default: ``0.0``.
        #     strict: Whether a missing monitored metric raises a
        #         configuration error listing the available keys instead of
        #         being skipped silently. Default: ``True``.
        super().__init__()
        self.monitor: str = monitor
        self.patience: int = patience
        self.mode: Literal["min", "max"] = mode
        self.min_delta: float = min_delta
        self.strict: bool = strict

        self.wait_count: int = 0
        self.best_score: torch.Tensor = torch.tensor(
            float("inf") if mode == "min" else float("-inf")
        )
        self.stopped_epoch: int = 0

    @override
    def on_validation_epoch_end(self, trainer: Trainer, module: Module) -> None:
        # Core stopping logic, executed after every real validation pass.
        # Reads the monitored value from the reduced epoch metrics, resets the
        # patience counter on improvement, and otherwise advances it; when the
        # counter reaches patience, the decision is synchronized across ranks
        # and the trainer's stop flag is raised with the outcome logged.
        if trainer.fast_dev_run or trainer.sanity_checking:
            return
        metrics: dict[str, torch.Tensor] = trainer.callback_metrics
        current_score: torch.Tensor | None = metrics.get(self.monitor)
        if current_score is None:
            if self.strict:
                available: list[str] = list(metrics.keys())
                raise MisconfigurationError(
                    f"EarlyStopping monitor '{self.monitor}' not found in validation metrics. "
                    f"Available: {available}. Set strict=False to suppress this error."
                )
            log.warning(f"EarlyStopping: metric '{self.monitor}' not found in metrics")
            return

        if self._is_improvement(current_score):
            self.best_score = current_score.detach().clone()
            self.wait_count: int = 0
        else:
            self.wait_count += 1
            should_stop: bool = self.wait_count >= self.patience
            should_stop: bool = trainer.strategy.reduce_boolean_decision(should_stop)
            if should_stop:
                self.stopped_epoch: int = module.current_epoch
                trainer.should_stop = True

    def _is_improvement(self, current: torch.Tensor) -> bool:
        # Applies the improvement test in the configured direction, requiring
        # the current value to beat the best score by more than min_delta.
        best_score: torch.Tensor = self.best_score.to(current.device)
        if self.mode == "min":
            return bool(current < best_score - self.min_delta)
        return bool(current > best_score + self.min_delta)

    @property
    def state_key(self) -> str:
        # Qualifies the checkpoint identity with monitor and mode so multiple
        # instances watching different metrics persist independently.
        return self._generate_state_key(monitor=self.monitor, mode=self.mode)

    @override
    def state_dict(self) -> StateDict:
        # Persists the patience counter, best score, and the epoch at which a
        # stop triggered, so resumption continues the plateau measurement.
        state_dict: StateDict = {
            "wait_count": self.wait_count,
            "best_score": self.best_score,
            "stopped_epoch": self.stopped_epoch
        }
        return state_dict

    @override
    def load_state_dict(self, state_dict: StateDict) -> None:
        # Restores the persisted counters with None-guarded coercion; missing
        # entries fall back to the freshly initialized values.
        wait_count: object = state_dict.get("wait_count", 0)
        best_score: object = state_dict.get("best_score", self.best_score)
        stopped_epoch: object = state_dict.get("stopped_epoch", 0)
        scalar_types: tuple[type[bool], type[int], type[float], type[str]] = (
            bool,
            int,
            float,
            str
        )
        if wait_count is not None and not isinstance(wait_count, scalar_types):
            raise TypeError("EarlyStopping wait_count checkpoint state must be scalar.")
        if best_score is not None and not isinstance(
            best_score,
            (*scalar_types, torch.Tensor)
        ):
            raise TypeError("EarlyStopping best_score checkpoint state must be scalar.")
        if stopped_epoch is not None and not isinstance(stopped_epoch, scalar_types):
            raise TypeError("EarlyStopping stopped_epoch checkpoint state must be scalar.")
        self.wait_count: int = int(wait_count if wait_count is not None else 0)
        if isinstance(best_score, torch.Tensor):
            if best_score.numel() != 1:
                raise TypeError("EarlyStopping best_score checkpoint state must be scalar.")
            self.best_score = best_score.detach().clone().squeeze()
        elif best_score is not None:
            self.best_score = torch.tensor(float(best_score))
        self.stopped_epoch: int = int(stopped_epoch if stopped_epoch is not None else 0)

    @override
    def __repr__(self) -> str:
        # Compact configuration line for logs and interactive debugging.
        return (
            f"EarlyStopping("
            f"monitor={self.monitor!r}, "
            f"patience={self.patience}, "
            f"mode={self.mode!r}, "
            f"min_delta={self.min_delta}, "
            f"strict={self.strict})"
        )
