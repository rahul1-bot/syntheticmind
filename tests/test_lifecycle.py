# This module:
# 1. Pins the public stage lifecycle: fit, validate, test, predict, cooperative
#    early stopping, multiple evaluation loaders, and durable metric records
#
# Design decisions:
# - Stage results are asserted through returned values and public views only
# - The early-stopping test uses a stopping threshold instead of patience so
#   the outcome does not depend on the optimization trajectory
#
# Author: Rahul Sawhney

import csv
from pathlib import Path

import torch

from syntheticmind import DatasetDataModule, Trainer
from syntheticmind.callbacks.early_stopping import EarlyStopping
from syntheticmind.loggers.csv_logger import CSVLogger
from syntheticmind.primitives.types import ModelOutput
from torch.utils.data import TensorDataset

from tests.support import LinearRegressor, ProbeData


class TestStages:
    # Every public stage runs on CPU and returns its documented shape.
    def test_fit_validate_test_predict(self, tmp_path: Path) -> None:
        data: ProbeData = ProbeData(samples=32, batch_size=8)
        model: LinearRegressor = LinearRegressor()
        trainer: Trainer = Trainer(
            accelerator="cpu",
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=str(tmp_path)
        )
        trainer.fit(model, train_dataloaders=data.loader(), val_dataloaders=data.loader())
        validation: list[dict[str, float]] = trainer.validate(
            model,
            dataloaders=data.loader(),
            verbose=False
        )
        testing: list[dict[str, float]] = trainer.test(
            model,
            dataloaders=data.loader(),
            verbose=False
        )
        predictions: list[ModelOutput] | list[list[ModelOutput]] | None = trainer.predict(
            model,
            dataloaders=data.loader()
        )
        assert trainer.current_epoch == 1
        assert trainer.global_step == data.batches
        assert list(validation[0]) == ["val_loss"]
        assert list(testing[0]) == ["test_loss"]
        assert isinstance(predictions, list)
        assert len(predictions) == data.batches
        assert all(isinstance(batch, torch.Tensor) for batch in predictions)

    def test_datamodule_and_multiple_validation_loaders(self, tmp_path: Path) -> None:
        data: ProbeData = ProbeData(samples=32, batch_size=8)
        dataset: TensorDataset = TensorDataset(data.inputs, data.targets)
        datamodule: DatasetDataModule = DatasetDataModule.from_datasets(
            train_dataset=dataset,
            val_dataset=[dataset, dataset],
            batch_size=8
        )
        trainer: Trainer = Trainer(
            accelerator="cpu",
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=str(tmp_path)
        )
        trainer.fit(LinearRegressor(), datamodule=datamodule)
        names: set[str] = set(trainer.callback_metrics)
        assert "val_loss/dataloader_idx_0" in names
        assert "val_loss/dataloader_idx_1" in names


class TestCooperativeStopping:
    # A stopping threshold ends fit after the first qualifying validation.
    def test_early_stopping_threshold(self, tmp_path: Path) -> None:
        data: ProbeData = ProbeData(samples=32, batch_size=8)
        stopper: EarlyStopping = EarlyStopping(
            monitor="val_loss",
            stopping_threshold=1.0e9
        )
        trainer: Trainer = Trainer(
            accelerator="cpu",
            max_epochs=5,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[stopper],
            default_root_dir=str(tmp_path)
        )
        trainer.fit(
            LinearRegressor(),
            train_dataloaders=data.loader(),
            val_dataloaders=data.loader()
        )
        assert trainer.current_epoch == 1
        assert stopper.stopped_epoch == 0
        assert stopper.stopping_reason is not None


class TestMetricRecords:
    # CSV records widen their schema across stages and keep earlier rows.
    def test_csv_schema_widens_across_stages(self, tmp_path: Path) -> None:
        data: ProbeData = ProbeData(samples=32, batch_size=8)
        logger: CSVLogger = CSVLogger(tmp_path, name="run", version=0)
        model: LinearRegressor = LinearRegressor()
        trainer: Trainer = Trainer(
            accelerator="cpu",
            max_epochs=1,
            logger=logger,
            log_every_n_steps=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=str(tmp_path)
        )
        trainer.fit(model, train_dataloaders=data.loader(), val_dataloaders=data.loader())
        trainer.test(model, dataloaders=data.loader(), verbose=False)
        metrics_path: Path = Path(logger.log_dir) / "metrics.csv"
        with metrics_path.open(newline="") as handle:
            rows: list[dict[str, str]] = list(csv.DictReader(handle))
        header: list[str] = list(rows[0])
        assert header[0] == "step"
        assert "train_loss_step" in header
        assert "val_loss" in header
        assert "test_loss" in header
        assert any(row["train_loss_step"] for row in rows)
        assert any(row["test_loss"] for row in rows)
