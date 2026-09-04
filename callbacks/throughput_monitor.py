import time
from collections.abc import Callable
from typing import override

import syntheticmind
import torch

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.exceptions import MisconfigurationError
from syntheticmind.types import Batch, StepOutput

__all__: list[str] = ["ThroughputMonitor"]


class ThroughputMonitor(Callback):
    def __init__(
        self,
        batch_size_fn: Callable[[Batch], int] | None = None,
        log_every_n_steps: int = 10
    ) -> None:
        if batch_size_fn is not None and not callable(batch_size_fn):
            raise TypeError("`batch_size_fn` must be callable or None.")
        if type(log_every_n_steps) is not int or log_every_n_steps < 1:
            raise ValueError("`log_every_n_steps` must be a positive integer.")
        self._batch_size_fn: Callable[[Batch], int] | None = batch_size_fn
        self.log_every_n_steps: int = log_every_n_steps
        self._interval_started_at: float | None = None
        self._window_samples: int = 0
        self._window_batches: int = 0
        self._window_seconds: float = 0.0
        self._epoch_samples: int = 0
        self._epoch_batches: int = 0
        self._epoch_seconds: float = 0.0
        self._last_logged_step: int = -1

    @override
    def on_train_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        del module
        if not trainer.loggers:
            raise MisconfigurationError(
                "ThroughputMonitor requires at least one configured logger."
            )

    @override
    def on_train_epoch_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        del module
        if not trainer.is_global_zero:
            return
        self._window_samples: int = 0
        self._window_batches: int = 0
        self._window_seconds: float = 0.0
        self._epoch_samples: int = 0
        self._epoch_batches: int = 0
        self._epoch_seconds: float = 0.0
        self._last_logged_step: int = -1
        self._synchronize(trainer)
        self._interval_started_at: float | None = time.perf_counter()

    @override
    def on_train_batch_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int
    ) -> None:
        del module, outputs, batch_idx
        if not trainer.is_global_zero:
            return
        self._synchronize(trainer)
        current_time: float = time.perf_counter()
        if self._interval_started_at is None:
            self._interval_started_at: float | None = current_time
            return
        elapsed_seconds: float = max(0.0, current_time - self._interval_started_at)
        self._interval_started_at: float | None = current_time
        local_batch_size: int = self._infer_batch_size(batch)
        global_batch_size: int = local_batch_size * trainer.strategy.data_parallel_size
        self._window_samples += global_batch_size
        self._window_batches += 1
        self._window_seconds += elapsed_seconds
        self._epoch_samples += global_batch_size
        self._epoch_batches += 1
        self._epoch_seconds += elapsed_seconds
        if trainer.global_step <= 0:
            return
        if trainer.global_step == self._last_logged_step:
            return
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        self._emit_window(trainer)
        self._last_logged_step: int = trainer.global_step

    @override
    def on_train_epoch_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        del module
        if not trainer.is_global_zero or self._epoch_seconds <= 0.0:
            return
        metrics: dict[str, float] = {
            "throughput/epoch_samples_per_sec": self._epoch_samples / self._epoch_seconds,
            "throughput/epoch_batches_per_sec": self._epoch_batches / self._epoch_seconds
        }
        trainer._logger_connector.callback_metrics.update(
            {
                name: torch.tensor(value, device=trainer.strategy.root_device)
                for name, value in metrics.items()
            }
        )
        trainer._log_metrics(metrics, step=trainer.global_step)

    @override
    def on_validation_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        del module
        if not trainer.is_global_zero or trainer.state.stage != "fit":
            return
        self._synchronize(trainer)
        self._interval_started_at: float | None = None

    @override
    def on_validation_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        del module
        if not trainer.is_global_zero or trainer.state.stage != "fit":
            return
        self._synchronize(trainer)
        self._interval_started_at: float | None = time.perf_counter()

    def _emit_window(self, trainer: syntheticmind.Trainer) -> None:
        if self._window_seconds <= 0.0 or self._window_batches == 0:
            return
        metrics: dict[str, float] = {
            "throughput/samples_per_sec": self._window_samples / self._window_seconds,
            "throughput/batches_per_sec": self._window_batches / self._window_seconds,
            "throughput/batch_time_ms": self._window_seconds / self._window_batches * 1000.0
        }
        trainer._logger_connector.callback_metrics.update(
            {
                name: torch.tensor(value, device=trainer.strategy.root_device)
                for name, value in metrics.items()
            }
        )
        trainer._log_metrics(metrics, step=trainer.global_step)
        self._window_samples: int = 0
        self._window_batches: int = 0
        self._window_seconds: float = 0.0

    def _infer_batch_size(self, batch: Batch) -> int:
        if self._batch_size_fn is not None:
            batch_size: int = self._batch_size_fn(batch)
            if type(batch_size) is not int or batch_size < 0:
                raise TypeError("`batch_size_fn` must return a non-negative integer.")
            return batch_size
        sizes: list[int] = self._collect_batch_sizes(batch)
        if not sizes:
            return 1
        unique_sizes: set[int] = set(sizes)
        if len(unique_sizes) != 1:
            raise MisconfigurationError(
                f"ThroughputMonitor found inconsistent tensor batch sizes {sorted(unique_sizes)}."
            )
        return sizes[0]

    def _collect_batch_sizes(self, batch: Batch) -> list[int]:
        if isinstance(batch, torch.Tensor):
            return [1 if batch.ndim == 0 else int(batch.shape[0])]
        if isinstance(batch, dict):
            sizes: list[int] = []
            value: Batch
            for value in batch.values():
                sizes.extend(self._collect_batch_sizes(value))
            return sizes
        if isinstance(batch, list | tuple):
            sizes: list[int] = []
            value: Batch
            for value in batch:
                sizes.extend(self._collect_batch_sizes(value))
            if sizes:
                return sizes
            return [len(batch)]
        return []

    def _synchronize(self, trainer: syntheticmind.Trainer) -> None:
        device: torch.device = trainer.strategy.root_device
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elif device.type == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()

    @override
    def __repr__(self) -> str:
        return (
            f"ThroughputMonitor(batch_size_fn={self._batch_size_fn!r}, "
            f"log_every_n_steps={self.log_every_n_steps})"
        )
