# Compiler, weighted-loss, and real restart verification

The workflow extracts the following block as `runtime_failure_probe.py` beside
`parallel_probe.py`. `local` exercises compiler diagnostics; `parallel` compares
unequally weighted context shards against a full-sequence reference. `baseline`
and `restart` are separate two-worker torchrun launches using one shared artifact
root. Only the `restart` fixture deliberately terminates its own rank-1 worker
at step 2 with exit code 86. TorchElastic must actually restart the worker group;
the second attempt must automatically restore a durable checkpoint. Final model,
optimizer, step, and rank-local torch RNG must equal the uninterrupted baseline.

This is an isolated CPU/Gloo process-failure test, not a simulation of a failed
physical node, network partition, CUDA context, or remote object store. It never
terminates a process outside its own explicitly selected fixture.

```python
import argparse
import os
from datetime import timedelta
from pathlib import Path
from typing import ClassVar, Literal, cast, override
from unittest.mock import patch

import torch
import torch.distributed as dist
from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError
from torch import nn
from torch._dynamo.exc import BackendCompilerFailed
from torch.utils.data import DataLoader, TensorDataset

from syntheticmind import Module, Trainer
from syntheticmind.callbacks.callback import Callback
from syntheticmind.callbacks.checkpointing import ModelCheckpoint
from syntheticmind.core.optimizer import SyntheticOptimizer
from syntheticmind.plugins import TorchElasticEnvironment
from syntheticmind.primitives.types import Batch, CheckpointDict, CheckpointValue, TrainingStepOutput
from syntheticmind.strategies import (
    ContextParallelConfiguration, ContextParallelStrategy,
    PipelineParallelConfiguration, PipelineParallelStrategy, PipelineStageShape,
    TorchCompiler
)
from syntheticmind.strategies.compiler import CompilationEvent, CompilationReport
from parallel_probe import ContextModule, PipelineModule

__all__: list[str] = []


class VerificationConfiguration(BaseModel):
    mode: Literal["local", "parallel", "baseline", "restart"]
    artifact_root: Path
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", strict=True, validate_default=True
    )


class GraphBreakModule(nn.Module):
    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        intermediate: torch.Tensor = inputs.sin()
        torch._dynamo.graph_break()
        return intermediate.cos()


class WeightedContextModule(ContextModule):
    @override
    def training_step(self, batch: Batch, batch_idx: int) -> TrainingStepOutput:
        del batch_idx
        inputs: torch.Tensor
        targets: torch.Tensor
        inputs, targets = cast(tuple[torch.Tensor, torch.Tensor], batch)
        inputs: torch.Tensor = inputs.chunk(2, dim=1)[self.global_rank].contiguous()
        targets: torch.Tensor = targets.chunk(2, dim=1)[self.global_rank].contiguous()
        valid_positions: int = 4 if self.global_rank == 0 else 1
        optimizer: SyntheticOptimizer = cast(SyntheticOptimizer, self.optimizers())
        optimizer.zero_grad()
        output: torch.Tensor = self(inputs)
        loss_sum: torch.Tensor = (output[:, :valid_positions] - targets[:, :valid_positions]).square().sum()
        strategy: ContextParallelStrategy = cast(ContextParallelStrategy, self.trainer.strategy)
        loss: torch.Tensor = strategy.normalize_loss(loss_sum, targets[:, :valid_positions].numel())
        self.manual_backward(loss)
        optimizer.step()
        return {"loss": loss}


class CrashOnce(Callback):
    def __init__(self, mode: Literal["baseline", "restart"]) -> None:
        self._mode: Literal["baseline", "restart"] = mode
        self.restored_step: int = 0

    @override
    def on_fit_start(self, trainer: Trainer, module: Module) -> None:
        del module
        self.restored_step: int = trainer.global_step
        environment: TorchElasticEnvironment = cast(TorchElasticEnvironment, trainer.strategy.cluster_environment)
        if environment.restart_count:
            assert self.restored_step > 0, "Restart did not restore a checkpoint."
            logger.info(f"Restored rank {trainer.global_rank} at step {self.restored_step}, attempt {environment.restart_count}")

    @override
    def on_train_batch_start(self, trainer: Trainer, module: Module, batch: Batch, batch_idx: int) -> None:
        del module, batch, batch_idx
        environment: TorchElasticEnvironment = cast(TorchElasticEnvironment, trainer.strategy.cluster_environment)
        if self._mode == "restart" and environment.restart_count == 0 and trainer.global_rank == 1 and trainer.global_step == 2:
            logger.warning("Deliberately terminating fixture worker rank 1 at step 2.")
            os._exit(86)


class RuntimeFailureProbe:
    def __init__(self) -> None:
        parser: argparse.ArgumentParser = argparse.ArgumentParser(suggest_on_error=True, color=False)
        parser.add_argument("--mode", choices=["local", "parallel", "baseline", "restart"], required=True)
        parser.add_argument("--artifact-root", type=Path, required=True)
        arguments: argparse.Namespace = parser.parse_args()
        self._configuration: VerificationConfiguration = VerificationConfiguration(
            mode=arguments.mode, artifact_root=arguments.artifact_root
        )

    def run(self) -> None:
        self._configuration.artifact_root.mkdir(parents=True, exist_ok=True)
        if self._configuration.mode == "local":
            self._compiler_reports()
            return
        if self._configuration.mode == "parallel":
            dist.init_process_group("gloo", timeout=timedelta(seconds=45))
            try:
                self._weighted_context("mean")
                self._weighted_context("sum")
                logger.info(f"PASS rank {dist.get_rank()}: unequal context loss weights under mean and sum gradient reduction")
            finally:
                dist.destroy_process_group()
            return
        self._restart_trajectory(self._configuration.mode)

    def _compiler_reports(self) -> None:
        compiler: TorchCompiler = TorchCompiler(backend="eager", shape_policy="static", diagnostics_limit=1)
        model: nn.Linear = nn.Linear(4, 4)
        compiler.compile(model)
        try:
            model(torch.ones(2, 4))
            model(torch.ones(3, 4))
            report: CompilationReport = compiler.report
            assert len(report.events) == 1 and report.dropped_event_count >= 1
            event: CompilationEvent = report.events[0]
            assert event.frame_compile_id is not None and event.frame_compile_id > 0
            assert event.backend_duration_seconds >= 0 and event.failure is None
        finally:
            compiler.teardown()
        destination: Path = self._configuration.artifact_root / "compiler.json"
        compiler.save_report(destination)
        assert CompilationReport.model_validate_json(destination.read_text()) == compiler.report
        broken: GraphBreakModule = GraphBreakModule()
        observed: TorchCompiler = TorchCompiler(backend="eager")
        observed.compile(broken)
        try:
            torch.testing.assert_close(broken(torch.ones(4)), torch.ones(4).sin().cos())
            assert any(event.graph_break_reason is not None for event in observed.report.events)
        finally:
            observed.teardown()
        failed: TorchCompiler = TorchCompiler(backend="eager")
        failed_model: nn.Linear = nn.Linear(4, 4)
        failed.compile(failed_model)
        try:
            with patch("syntheticmind.strategies.compiler.lookup_backend", side_effect=RuntimeError("injected compiler failure")):
                try:
                    failed_model(torch.ones(7, 4))
                except BackendCompilerFailed:
                    assert "injected compiler failure" in str(failed.report.events[-1].failure)
                else:
                    raise AssertionError("Compiler failure was swallowed.")
        finally:
            failed.teardown()
        assert "forward" not in vars(failed_model)
        try:
            CompilationReport.model_validate(compiler.report.model_dump() | {"schema_version": 999})
        except ValidationError:
            pass
        else:
            raise AssertionError("Unknown compiler report schema accepted.")
        logger.info("PASS: native recompilation IDs, bounded events, graph-break reasons, JSON, backend failure and teardown")

    def _weighted_context(self, reduction: Literal["mean", "sum"]) -> None:
        inputs: torch.Tensor = torch.arange(64, dtype=torch.float32).reshape(2, 8, 4) / 64
        targets: torch.Tensor = inputs * 0.2
        reference: ContextModule = ContextModule()
        optimizer: torch.optim.Optimizer = reference.configure_optimizers()
        optimizer.zero_grad()
        (reference(inputs)[:, :5] - targets[:, :5]).square().mean().backward()
        optimizer.step()
        model: WeightedContextModule = WeightedContextModule()
        strategy: ContextParallelStrategy = ContextParallelStrategy(
            ContextParallelConfiguration(context_parallel_size=2, gradient_reduction=reduction),
            process_group_backend="gloo"
        )
        trainer: Trainer = Trainer(
            accelerator="cpu", devices=2, strategy=strategy, max_steps=1, max_epochs=1,
            use_distributed_sampler=False, enable_checkpointing=False,
            logger=False, enable_model_summary=False, enable_progress_bar=False,
            num_sanity_val_steps=0, limit_val_batches=0
        )
        trainer.fit(model, train_dataloaders=DataLoader(TensorDataset(inputs, targets), batch_size=2))
        actual: torch.Tensor
        expected: torch.Tensor
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def _restart_trajectory(self, mode: Literal["baseline", "restart"]) -> None:
        environment: TorchElasticEnvironment = TorchElasticEnvironment()
        rank: int = environment.global_rank()
        directory: Path = self._configuration.artifact_root / mode
        directory.mkdir(parents=True, exist_ok=True)
        policy: PipelineParallelConfiguration = PipelineParallelConfiguration(
            stages=(
                PipelineStageShape(input_shape=(2, 4), output_shape=(2, 6)),
                PipelineStageShape(input_shape=(2, 6), output_shape=(2, 3))
            ),
            n_microbatches=4, schedule="1f1b", loss_id="mse:v1"
        )
        crash: CrashOnce = CrashOnce(mode)
        model: PipelineModule = PipelineModule(rank, 0.2)
        trainer: Trainer = Trainer(
            accelerator="cpu", devices=2,
            strategy=PipelineParallelStrategy(
                policy, nn.MSELoss(), checkpoint_write="asynchronous",
                process_group_backend="gloo", timeout=timedelta(seconds=45)
            ),
            max_steps=4, max_epochs=2, seed=700 + rank, use_distributed_sampler=False,
            callbacks=[crash, ModelCheckpoint(
                dirpath=directory / "checkpoints", save_last=True,
                save_top_k=-1, every_n_train_steps=1
            )],
            logger=False, enable_model_summary=False, enable_progress_bar=False,
            num_sanity_val_steps=0, limit_val_batches=0
        )
        inputs: torch.Tensor = torch.arange(128, dtype=torch.float32).reshape(32, 4) / 128
        loader: DataLoader = DataLoader(TensorDataset(inputs, inputs[:, :3] * 0.4), batch_size=8)
        try:
            trainer.fit(model, train_dataloaders=loader)
            state: CheckpointDict = {
                "model": cast(CheckpointValue, model.state_dict()),
                "optimizer": cast(CheckpointValue, trainer.optimizers[0].state_dict()),
                "rng": torch.get_rng_state(), "global_step": trainer.global_step
            }
            if mode == "baseline":
                torch.save(state, directory / f"rank-{rank}.pt")
            else:
                assert environment.restart_count == 1 and crash.restored_step > 0
                expected: CheckpointDict = cast(CheckpointDict, torch.load(
                    self._configuration.artifact_root / "baseline" / f"rank-{rank}.pt",
                    weights_only=True
                ))
                torch.testing.assert_close(state, expected, atol=0, rtol=0)
                logger.info(f"PASS rank {rank}: real worker exit, automatic restart, exact model/optimizer/RNG/step continuation")
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    probe: RuntimeFailureProbe = RuntimeFailureProbe()
    probe.run()
```
