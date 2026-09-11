# This module:
# 1. Pins the spawned two-process CPU data-parallel path: launcher state
#    transfer, rank-zero-only presentation, and parent-side progress
#    reconciliation
#
# Design decisions:
# - The test carries the distributed marker so environments that cannot spawn
#   worker processes deselect it with ``-m "not distributed"``
# - Gloo on CPU is the minimum proof for collective ordering; the assertions
#   read parent-side views the launcher must reconcile from the workers
# - Terminal output is captured at the file-descriptor level because the
#   rendering rank is a spawned worker that inherits the process stderr
#
# Author: Rahul Sawhney

from pathlib import Path

import pytest

from syntheticmind import Trainer

from tests.support import LinearRegressor, ProbeData


@pytest.mark.distributed
class TestSpawnedDataParallel:
    # Two spawned CPU workers train one epoch and report through rank zero.
    def test_two_process_fit_reconciles_progress(
        self,
        tmp_path: Path,
        capfd: pytest.CaptureFixture[str]
    ) -> None:
        data: ProbeData = ProbeData(samples=64, batch_size=8)
        trainer: Trainer = Trainer(
            accelerator="cpu",
            devices=2,
            strategy="ddp_spawn",
            max_epochs=1,
            seed=3,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            default_root_dir=str(tmp_path)
        )
        trainer.fit(LinearRegressor(), train_dataloaders=data.loader())
        assert trainer.world_size == 2
        assert trainer.current_epoch == 1
        assert trainer.global_step == data.batches // 2
        lines: list[str] = capfd.readouterr().err.splitlines()
        assert lines.count("syntheticmind · fit") == 1
        assert sum(line.startswith("✓ epoch 1/1") for line in lines) == 1
