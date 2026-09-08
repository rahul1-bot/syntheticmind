# This module:
# 1. Defines the small deterministic regression models, data builders, and
#    stream doubles shared by the test modules
#
# Design decisions:
# - Models log through the public Module.log contract so tests exercise the
#   same metric routing as client code rather than private accumulators
# - Every data builder seeds its own generator, so a test never depends on
#   global random state left behind by another test
# - Stream doubles subclass StringIO and only override isatty, keeping the
#   renderer on its real write and flush path
#
# Author: Rahul Sawhney

import io
import warnings
from typing import override

import torch
from loguru import logger as log
from torch.utils.data import DataLoader, TensorDataset

from syntheticmind import Module
from syntheticmind.primitives.types import Batch, TrainingStepOutput

__all__: list[str] = ["LinearRegressor", "ProbeData", "TerminalStringIO"]


class LinearRegressor(Module):
    # One linear layer with configurable batch skipping and failure injection
    # so optimization, presentation, and checkpoint contracts can be probed
    # without a domain model.
    def __init__(
        self,
        input_width: int = 8,
        learning_rate: float = 0.1,
        skip_batches: tuple[int, ...] = (),
        fail_at_batch: int | None = None,
        scheduler_step_size: int | None = None,
        emit_at_batch: int | None = None
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.network: torch.nn.Linear = torch.nn.Linear(input_width, 1)
        self.learning_rate: float = learning_rate
        self.skip_batches: tuple[int, ...] = skip_batches
        self.fail_at_batch: int | None = fail_at_batch
        self.scheduler_step_size: int | None = scheduler_step_size
        self.emit_at_batch: int | None = emit_at_batch
        self.seen_batches: list[int] = []

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

    def _loss(self, batch: Batch) -> torch.Tensor:
        # Computes the mean-squared error for one input and target pair.
        if not isinstance(batch, list | tuple) or len(batch) != 2:
            raise TypeError("LinearRegressor expects an input-target pair.")
        inputs, targets = batch
        if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise TypeError("LinearRegressor expects tensor inputs and targets.")
        return torch.nn.functional.mse_loss(self(inputs), targets)

    @override
    def training_step(self, batch: Batch, batch_idx: int) -> TrainingStepOutput:
        self.seen_batches.append(batch_idx)
        if self.fail_at_batch is not None and batch_idx == self.fail_at_batch:
            raise RuntimeError("injected training failure")
        if self.emit_at_batch is not None and batch_idx == self.emit_at_batch:
            log.warning("loguru record from training_step")
            warnings.warn("python warning from training_step", UserWarning, stacklevel=1)
            self.print("module print from training_step")
        if batch_idx in self.skip_batches:
            return None
        loss: torch.Tensor = self._loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    @override
    def validation_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        del batch_idx, dataloader_idx
        self.log("val_loss", self._loss(batch), on_epoch=True, prog_bar=True)

    @override
    def test_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        del batch_idx, dataloader_idx
        self.log("test_loss", self._loss(batch), on_epoch=True, prog_bar=True)

    @override
    def predict_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        if not isinstance(batch, list | tuple):
            raise TypeError("LinearRegressor expects an input-target pair.")
        return self(batch[0])

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer | dict[str, object]:
        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate
        )
        if self.scheduler_step_size is None:
            return optimizer
        scheduler: torch.optim.lr_scheduler.StepLR = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_step_size,
            gamma=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }


class ProbeData:
    # Seeded tensor datasets and loaders sized for sub-second CPU runs.
    def __init__(
        self,
        samples: int = 64,
        input_width: int = 8,
        batch_size: int = 8,
        seed: int = 17
    ) -> None:
        generator: torch.Generator = torch.Generator().manual_seed(seed)
        self.inputs: torch.Tensor = torch.randn(samples, input_width, generator=generator)
        self.targets: torch.Tensor = torch.randn(samples, 1, generator=generator)
        self.batch_size: int = batch_size

    def loader(self, shuffle: bool = False, batch_size: int | None = None) -> DataLoader:
        # Builds one deterministic loader over the seeded tensors.
        return DataLoader(
            TensorDataset(self.inputs, self.targets),
            batch_size=self.batch_size if batch_size is None else batch_size,
            shuffle=shuffle
        )

    @property
    def batches(self) -> int:
        # Number of batches one epoch of the default loader produces.
        return (len(self.inputs) + self.batch_size - 1) // self.batch_size


class TerminalStringIO(io.StringIO):
    # In-memory stream that reports itself as an interactive terminal.
    @override
    def isatty(self) -> bool:
        return True
