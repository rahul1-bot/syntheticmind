# This module:
# 1. Pins automatic-optimization progress semantics: skipped batches, gradient
#    accumulation windows, step ceilings, and scheduler cadence
#
# Design decisions:
# - Every assertion reads public Trainer and Module progress views so the tests
#   describe the client-visible contract rather than loop internals
# - Parameter snapshots are compared exactly because SGD without momentum on a
#   skipped window must leave weights bit-identical
#
# Author: Rahul Sawhney

import torch

from syntheticmind import Trainer

from tests.support import LinearRegressor, ProbeData


class TestSkippedBatches:
    # A training_step that returns None must not count as optimizer progress.
    def test_skipped_window_does_not_advance_global_step(self, tmp_path) -> None:
        data: ProbeData = ProbeData(samples=32, batch_size=8)
        model: LinearRegressor = LinearRegressor(skip_batches=(1, 2))
        before: torch.Tensor = model.network.weight.detach().clone()
        trainer: Trainer = Trainer(
            accelerator="cpu",
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=str(tmp_path)
        )
        trainer.fit(model, train_dataloaders=data.loader())
        assert model.seen_batches == [0, 1, 2, 3]
        assert trainer.global_step == 2
        assert not torch.equal(model.network.weight.detach(), before)

    def test_skipped_window_does_not_step_schedulers(self, tmp_path) -> None:
        data: ProbeData = ProbeData(samples=32, batch_size=8)
        model: LinearRegressor = LinearRegressor(
            learning_rate=1.0,
            skip_batches=(1, 2),
            scheduler_step_size=1
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
        trainer.fit(model, train_dataloaders=data.loader())
        learning_rate: float = float(trainer.optimizers[0].param_groups[0]["lr"])
        assert trainer.global_step == 2
        assert learning_rate == 0.25

    def test_skipped_microbatch_inside_window_keeps_window_progress(self, tmp_path) -> None:
        data: ProbeData = ProbeData(samples=32, batch_size=8)
        model: LinearRegressor = LinearRegressor(skip_batches=(1,))
        trainer: Trainer = Trainer(
            accelerator="cpu",
            max_epochs=1,
            accumulate_grad_batches=2,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=str(tmp_path)
        )
        trainer.fit(model, train_dataloaders=data.loader())
        assert trainer.global_step == 2


class TestStepCeilings:
    # max_steps counts completed managed optimizer mutations only.
    def test_max_steps_stops_at_a_window_boundary(self, tmp_path) -> None:
        data: ProbeData = ProbeData(samples=96, batch_size=8)
        model: LinearRegressor = LinearRegressor()
        trainer: Trainer = Trainer(
            accelerator="cpu",
            max_epochs=-1,
            max_steps=3,
            accumulate_grad_batches=3,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=str(tmp_path)
        )
        trainer.fit(model, train_dataloaders=data.loader())
        assert trainer.global_step == 3
        assert len(model.seen_batches) == 9

    def test_epoch_tail_flushes_partial_window(self, tmp_path) -> None:
        data: ProbeData = ProbeData(samples=40, batch_size=8)
        model: LinearRegressor = LinearRegressor()
        trainer: Trainer = Trainer(
            accelerator="cpu",
            max_epochs=1,
            accumulate_grad_batches=3,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=str(tmp_path)
        )
        trainer.fit(model, train_dataloaders=data.loader())
        assert len(model.seen_batches) == 5
        assert trainer.global_step == 2
