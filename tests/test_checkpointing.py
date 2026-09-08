# This module:
# 1. Pins checkpoint durability contracts: exact continuation, ranked
#    retention, and asynchronous transport failure visibility
#
# Design decisions:
# - Exact continuation is asserted as bit-identical parameters between an
#   uninterrupted run and a resumed run because the framework claims exact,
#   not approximate, continuation
# - The asynchronous failure test pins only guaranteed behavior: the earlier
#   artifact survives, the failed artifact never appears, and the original
#   exception surfaces from the stage that requested the save
#
# Author: Rahul Sawhney

from pathlib import Path
from typing import override

import pytest
import torch

from syntheticmind import Trainer
from syntheticmind.callbacks.checkpointing.model_checkpoint import ModelCheckpoint
from syntheticmind.plugins.checkpoint_io import (
    AsyncCheckpointIO,
    CheckpointPath,
    CheckpointStorageOptions,
    TorchCheckpointIO
)
from syntheticmind.primitives.types import CheckpointDict

from tests.support import LinearRegressor, ProbeData


class _FailOnSecondSave(TorchCheckpointIO):
    # Transport double whose second save fails after the first succeeded.
    def __init__(self) -> None:
        super().__init__()
        self.saves: int = 0

    @override
    def save_checkpoint(
        self,
        checkpoint: CheckpointDict,
        path: CheckpointPath,
        storage_options: CheckpointStorageOptions = None
    ) -> None:
        self.saves += 1
        if self.saves == 2:
            raise OSError("injected transport failure")
        super().save_checkpoint(checkpoint, path, storage_options)


class TestExactContinuation:
    # Resuming from a durable checkpoint reproduces the uninterrupted trajectory.
    def test_resumed_fit_matches_uninterrupted_fit(self, tmp_path: Path) -> None:
        data: ProbeData = ProbeData(samples=64, batch_size=8)
        torch.manual_seed(11)
        reference: LinearRegressor = LinearRegressor()
        Trainer(
            accelerator="cpu",
            max_epochs=2,
            seed=11,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=str(tmp_path / "reference")
        ).fit(reference, train_dataloaders=data.loader())

        torch.manual_seed(11)
        first: LinearRegressor = LinearRegressor()
        first_trainer: Trainer = Trainer(
            accelerator="cpu",
            max_epochs=1,
            seed=11,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=str(tmp_path / "first")
        )
        first_trainer.fit(first, train_dataloaders=data.loader())
        checkpoint_path: Path = tmp_path / "first" / "epoch_one.ckpt"
        first_trainer.save_checkpoint(checkpoint_path)

        resumed: LinearRegressor = LinearRegressor()
        resumed_trainer: Trainer = Trainer(
            accelerator="cpu",
            max_epochs=2,
            seed=11,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=str(tmp_path / "resumed")
        )
        resumed_trainer.fit(resumed, train_dataloaders=data.loader(), ckpt_path=checkpoint_path)
        assert resumed_trainer.current_epoch == 2
        assert resumed_trainer.global_step == 16
        for name, parameter in reference.state_dict().items():
            assert torch.equal(parameter, resumed.state_dict()[name]), name


class TestRankedRetention:
    # Top-k retention keeps the best artifact and a last artifact.
    def test_top_one_keeps_best_and_last(self, tmp_path: Path) -> None:
        data: ProbeData = ProbeData(samples=32, batch_size=8)
        checkpoint: ModelCheckpoint = ModelCheckpoint(
            dirpath=tmp_path / "checkpoints",
            monitor="val_loss",
            save_top_k=1,
            save_last=True
        )
        trainer: Trainer = Trainer(
            accelerator="cpu",
            max_epochs=3,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[checkpoint],
            default_root_dir=str(tmp_path)
        )
        trainer.fit(
            LinearRegressor(),
            train_dataloaders=data.loader(),
            val_dataloaders=data.loader()
        )
        artifacts: list[str] = sorted(path.name for path in (tmp_path / "checkpoints").iterdir())
        assert len(artifacts) == 2
        assert "last.ckpt" in artifacts
        assert Path(checkpoint.best_model_path).name in artifacts
        assert checkpoint.best_model_score is not None


class TestAsynchronousFailure:
    # A failed background save cannot destroy the prior artifact and must
    # surface its original exception from the requesting stage.
    def test_failure_preserves_previous_artifact(self, tmp_path: Path) -> None:
        data: ProbeData = ProbeData(samples=64, batch_size=8)
        transport: _FailOnSecondSave = _FailOnSecondSave()
        checkpoint: ModelCheckpoint = ModelCheckpoint(
            dirpath=tmp_path / "checkpoints",
            filename="step-{step}",
            every_n_train_steps=1,
            save_top_k=-1
        )
        trainer: Trainer = Trainer(
            accelerator="cpu",
            max_epochs=1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            plugins=[AsyncCheckpointIO(transport)],
            callbacks=[checkpoint],
            default_root_dir=str(tmp_path)
        )
        with pytest.raises(OSError, match="injected transport failure"):
            trainer.fit(LinearRegressor(), train_dataloaders=data.loader())
        artifacts: list[str] = sorted(path.name for path in (tmp_path / "checkpoints").iterdir())
        assert artifacts == ["step-step=1.ckpt"]
