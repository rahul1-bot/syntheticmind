import math
from typing import Literal, override

import syntheticmind
import torch
from loguru import logger as log

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.exceptions import MisconfigurationError
from syntheticmind.types import StateDict, StateValue

__all__: list[str] = ["EarlyStopping"]

type StoppingReason = Literal[
    "stopping_threshold",
    "divergence_threshold",
    "patience_exhausted",
    "non_finite_metric"
]


class EarlyStopping(Callback):
    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: Literal["min", "max"] = "min",
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: float | None = None,
        divergence_threshold: float | None = None
    ) -> None:
        if not isinstance(monitor, str) or not monitor.strip():
            raise ValueError("`monitor` must be a non-empty string.")
        if (
            type(min_delta) not in (int, float)
            or not math.isfinite(float(min_delta))
            or min_delta < 0
        ):
            raise ValueError("`min_delta` must be a non-negative number.")
        if type(patience) is not int or patience < 0:
            raise ValueError("`patience` must be a non-negative integer.")
        if type(verbose) is not bool:
            raise TypeError("`verbose` must be a bool.")
        if mode not in ("min", "max"):
            raise ValueError("`mode` must be 'min' or 'max'.")
        if type(strict) is not bool:
            raise TypeError("`strict` must be a bool.")
        if type(check_finite) is not bool:
            raise TypeError("`check_finite` must be a bool.")
        self._validate_threshold("stopping_threshold", stopping_threshold)
        self._validate_threshold("divergence_threshold", divergence_threshold)
        self.monitor: str = monitor
        self.min_delta: float = float(min_delta)
        self.patience: int = patience
        self.verbose: bool = verbose
        self.mode: Literal["min", "max"] = mode
        self.strict: bool = strict
        self.check_finite: bool = check_finite
        self.stopping_threshold: float | None = (
            None if stopping_threshold is None else float(stopping_threshold)
        )
        self.divergence_threshold: float | None = (
            None if divergence_threshold is None else float(divergence_threshold)
        )
        self.wait_count: int = 0
        self.best_score: torch.Tensor = torch.tensor(
            float("inf") if mode == "min" else float("-inf")
        )
        self.stopped_epoch: int = 0
        self.stopping_reason: StoppingReason | None = None
        self.stopping_reason_message: str | None = None

    @property
    @override
    def state_key(self) -> str:
        return self._generate_state_key(
            monitor=self.monitor,
            min_delta=self.min_delta,
            patience=self.patience,
            mode=self.mode,
            check_finite=self.check_finite,
            stopping_threshold=self.stopping_threshold,
            divergence_threshold=self.divergence_threshold
        )

    @override
    def on_validation_epoch_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        if trainer.fast_dev_run or trainer.sanity_checking:
            return
        local_score: torch.Tensor | None = trainer.callback_metrics.get(self.monitor)
        broadcast_score: torch.Tensor | None = trainer.strategy.broadcast(
            local_score if trainer.is_global_zero else None
        )
        if broadcast_score is None:
            message: str = (
                f"EarlyStopping monitor {self.monitor!r} is unavailable. "
                f"Available metrics: {sorted(trainer.callback_metrics)}."
            )
            if self.strict:
                raise MisconfigurationError(message)
            if trainer.is_global_zero:
                log.warning(message)
            return
        if broadcast_score.numel() != 1:
            raise MisconfigurationError(
                f"EarlyStopping monitor {self.monitor!r} must be scalar, "
                f"got shape {tuple(broadcast_score.shape)}."
            )
        current_score: torch.Tensor = broadcast_score.detach().cpu().squeeze()
        evaluation: tuple[bool, StoppingReason | None, str | None] = (
            self._evaluate(current_score)
        )
        should_stop: bool = evaluation[0]
        reason: StoppingReason | None = evaluation[1]
        message: str | None = evaluation[2]
        synchronized_stop: bool = trainer.strategy.reduce_boolean_decision(
            should_stop,
            all=False
        )
        trainer.should_stop: bool = trainer.should_stop or synchronized_stop
        if not synchronized_stop:
            return
        self.stopped_epoch: int = module.current_epoch
        self.stopping_reason: StoppingReason | None = reason
        self.stopping_reason_message: str | None = message
        if self.verbose and trainer.is_global_zero and message is not None:
            log.info(message)

    @override
    def state_dict(self) -> StateDict:
        return {
            "wait_count": self.wait_count,
            "best_score": self.best_score.detach().cpu().clone(),
            "stopped_epoch": self.stopped_epoch,
            "stopping_reason": self.stopping_reason,
            "stopping_reason_message": self.stopping_reason_message
        }

    @override
    def load_state_dict(self, state_dict: StateDict) -> None:
        wait_value: StateValue | None = state_dict.get("wait_count")
        score_value: StateValue | None = state_dict.get("best_score")
        epoch_value: StateValue | None = state_dict.get("stopped_epoch")
        reason_value: StateValue | None = state_dict.get("stopping_reason")
        message_value: StateValue | None = state_dict.get("stopping_reason_message")
        if type(wait_value) is not int or wait_value < 0:
            raise TypeError("EarlyStopping wait_count must be a non-negative integer.")
        if type(epoch_value) is not int or epoch_value < 0:
            raise TypeError("EarlyStopping stopped_epoch must be a non-negative integer.")
        if reason_value not in (
            None,
            "stopping_threshold",
            "divergence_threshold",
            "patience_exhausted",
            "non_finite_metric"
        ):
            raise TypeError("EarlyStopping stopping_reason is invalid.")
        if message_value is not None and not isinstance(message_value, str):
            raise TypeError("EarlyStopping stopping_reason_message must be a string or None.")
        if isinstance(score_value, torch.Tensor):
            if score_value.numel() != 1:
                raise TypeError("EarlyStopping best_score must be scalar.")
            best_score: torch.Tensor = score_value.detach().cpu().clone().squeeze()
        elif type(score_value) is int or type(score_value) is float:
            best_score: torch.Tensor = torch.tensor(float(score_value))
        else:
            raise TypeError("EarlyStopping best_score must be a scalar tensor or number.")
        self.wait_count: int = wait_value
        self.best_score: torch.Tensor = best_score
        self.stopped_epoch: int = epoch_value
        self.stopping_reason: StoppingReason | None = reason_value
        self.stopping_reason_message: str | None = message_value

    def _evaluate(
        self,
        current_score: torch.Tensor
    ) -> tuple[bool, StoppingReason | None, str | None]:
        current_value: float = float(current_score.item())
        if self.check_finite and not bool(torch.isfinite(current_score)):
            return (
                True,
                "non_finite_metric",
                f"Early stopping because {self.monitor}={current_value} is non-finite."
            )
        if self._reached_stopping_threshold(current_value):
            return (
                True,
                "stopping_threshold",
                f"Early stopping because {self.monitor}={current_value} reached "
                f"the target {self.stopping_threshold}."
            )
        if self._reached_divergence_threshold(current_value):
            return (
                True,
                "divergence_threshold",
                f"Early stopping because {self.monitor}={current_value} crossed "
                f"the divergence limit {self.divergence_threshold}."
            )
        if self._is_improvement(current_value):
            self.best_score: torch.Tensor = torch.tensor(current_value)
            self.wait_count: int = 0
            self.stopping_reason: StoppingReason | None = None
            self.stopping_reason_message: str | None = None
            return False, None, None
        self.wait_count += 1
        if self.wait_count < self.patience:
            return False, None, None
        return (
            True,
            "patience_exhausted",
            f"Early stopping because {self.monitor} did not improve for "
            f"{self.wait_count} validation checks. Best={float(self.best_score)}."
        )

    def _is_improvement(self, current_value: float) -> bool:
        best_value: float = float(self.best_score)
        if self.mode == "min":
            return current_value < best_value - self.min_delta
        return current_value > best_value + self.min_delta

    def _reached_stopping_threshold(self, current_value: float) -> bool:
        if self.stopping_threshold is None:
            return False
        if self.mode == "min":
            return current_value <= self.stopping_threshold
        return current_value >= self.stopping_threshold

    def _reached_divergence_threshold(self, current_value: float) -> bool:
        if self.divergence_threshold is None:
            return False
        if self.mode == "min":
            return current_value >= self.divergence_threshold
        return current_value <= self.divergence_threshold

    def _validate_threshold(self, name: str, value: float | None) -> None:
        if value is None:
            return
        if type(value) not in (int, float):
            raise TypeError(f"`{name}` must be a finite number or None.")
        if not math.isfinite(float(value)):
            raise ValueError(f"`{name}` must be a finite number or None.")

    @override
    def __repr__(self) -> str:
        return (
            f"EarlyStopping(monitor={self.monitor!r}, min_delta={self.min_delta}, "
            f"patience={self.patience}, verbose={self.verbose}, "
            f"mode={self.mode!r}, strict={self.strict}, "
            f"check_finite={self.check_finite}, "
            f"stopping_threshold={self.stopping_threshold}, "
            f"divergence_threshold={self.divergence_threshold})"
        )
