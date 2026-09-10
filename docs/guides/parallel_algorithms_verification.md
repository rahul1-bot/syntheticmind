# Parallel algorithm runtime probes

These executable examples exercise two real ranks. The workflow extracts the
single Python block below into `parallel_probe.py` outside the tracked package.
Its runtime result, not the existence of this file, is evidence of a pass.

The probe compares context attention outputs and input gradients against ordinary
full-sequence attention, and replicated-parameter training against an unsharded
reference. It exercises native GPipe and 1F1B, checks the local stage weights
against sequential training, and checks interrupted/resumed stochastic pipeline
training with per-rank RNG. All fixtures are created in a fresh explicit artifact
root. These are CPU/Gloo checks, not GPU throughput benchmarks.

Reference and harness paths deliberately use the same sequential sample order.
The probe disables automatic DistributedSampler injection: CP/PP ranks cooperate
on each complete sample, and the client partitions its sequence or computation.
Comparing a sequential reference against a shuffled training order is not an
algorithm-equivalence test. Numerical and bit-exact resume tolerances are unchanged.

```python
import argparse
from datetime import timedelta
from pathlib import Path
from typing import ClassVar, Literal, cast, override

import torch
import torch.distributed as dist
from loguru import logger
from pydantic import BaseModel, ConfigDict
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader, TensorDataset

from syntheticmind import Module, Trainer
from syntheticmind.callbacks.checkpointing import ModelCheckpoint
from syntheticmind.core.context_parallel_attention import ContextParallelAttention
from syntheticmind.core.optimizer import SyntheticOptimizer
from syntheticmind.primitives.types import Batch, TrainingStepOutput
from syntheticmind.strategies import (
    ContextParallelConfiguration, ContextParallelStrategy,
    PipelineParallelConfiguration, PipelineParallelStrategy, PipelineStageShape
)

__all__: list[str] = []


class ProbeConfiguration(BaseModel):
    artifact_root: Path
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", strict=True, validate_default=True
    )


class PipelineModule(Module):
    def __init__(self, rank: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.automatic_optimization: bool = False
        self.rank_index: int = rank
        torch.manual_seed(1000 + rank)
        self.network: nn.Linear = nn.Linear(4, 6) if rank == 0 else nn.Linear(6, 3)
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

    @override
    def configure_model(self) -> None:
        # Partition construction is explicit and already complete.
        pass

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.network(inputs)
        return self.dropout(output.tanh() if self.rank_index == 0 else output)

    @override
    def training_step(self, batch: Batch, batch_idx: int) -> TrainingStepOutput:
        del batch_idx
        inputs: torch.Tensor
        targets: torch.Tensor
        inputs, targets = cast(tuple[torch.Tensor, torch.Tensor], batch)
        optimizer: SyntheticOptimizer = cast(SyntheticOptimizer, self.optimizers())
        optimizer.zero_grad()
        strategy: PipelineParallelStrategy = cast(PipelineParallelStrategy, self.trainer.strategy)
        loss: torch.Tensor = strategy.pipeline_step(
            inputs if self.global_rank == 0 else None,
            targets if self.global_rank == self.trainer.world_size - 1 else None
        )
        optimizer.step()
        self.log("loss", loss, on_step=True, on_epoch=True, batch_size=inputs.shape[0])
        return {"loss": loss}

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=0.003)


class ContextModule(Module):
    def __init__(self) -> None:
        super().__init__()
        self.automatic_optimization: bool = False
        torch.manual_seed(333)
        self.projection: nn.Linear = nn.Linear(4, 12)
        self.attention: ContextParallelAttention | None = None

    @override
    def configure_model(self) -> None:
        self.attention: ContextParallelAttention | None = ContextParallelAttention(
            self.device_mesh["context_parallel"]
        )

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query: torch.Tensor
        key: torch.Tensor
        value: torch.Tensor
        query, key, value = self.projection(inputs).chunk(3, dim=-1)
        if self.attention is None:
            return torch.nn.functional.scaled_dot_product_attention(
                query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1), is_causal=True
            ).squeeze(1)
        return self.attention(query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1)).squeeze(1)

    @override
    def training_step(self, batch: Batch, batch_idx: int) -> TrainingStepOutput:
        del batch_idx
        inputs: torch.Tensor
        targets: torch.Tensor
        inputs, targets = cast(tuple[torch.Tensor, torch.Tensor], batch)
        inputs: torch.Tensor = inputs.chunk(self.trainer.world_size, dim=1)[self.global_rank].contiguous()
        targets: torch.Tensor = targets.chunk(self.trainer.world_size, dim=1)[self.global_rank].contiguous()
        optimizer: SyntheticOptimizer = cast(SyntheticOptimizer, self.optimizers())
        optimizer.zero_grad()
        loss: torch.Tensor = torch.nn.functional.mse_loss(self(inputs), targets)
        self.manual_backward(loss)
        optimizer.step()
        self.log("loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=inputs.shape[0])
        return {"loss": loss}

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=0.003)


class ParallelAlgorithmProbe:
    def __init__(self) -> None:
        parser: argparse.ArgumentParser = argparse.ArgumentParser(suggest_on_error=True, color=False)
        parser.add_argument("--artifact-root", type=Path, required=True)
        arguments: argparse.Namespace = parser.parse_args()
        self._configuration: ProbeConfiguration = ProbeConfiguration(artifact_root=arguments.artifact_root)

    def run(self) -> None:
        dist.init_process_group("gloo", timeout=timedelta(seconds=45))
        try:
            self._configuration.artifact_root.mkdir(parents=True, exist_ok=True)
            self._attention_gradients()
            self._context_training()
            self._pipeline_training("gpipe")
            self._pipeline_training("1f1b")
            logger.info(f"PASS rank {dist.get_rank()}: context attention/gradients/training; GPipe and 1F1B sequential equivalence, stochastic resume and rank RNG")
        finally:
            dist.destroy_process_group()

    def _attention_gradients(self) -> None:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("context",))
        torch.manual_seed(2026)
        query: torch.Tensor = torch.randn(2, 2, 8, 4, requires_grad=True)
        key: torch.Tensor = torch.randn(2, 2, 8, 4, requires_grad=True)
        value: torch.Tensor = torch.randn(2, 2, 8, 4, requires_grad=True)
        reference: torch.Tensor = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
        reference.square().sum().backward()
        rank: int = dist.get_rank()
        tensor: torch.Tensor
        local: list[torch.Tensor] = [
            tensor.detach().chunk(2, dim=2)[rank].clone().requires_grad_()
            for tensor in (query, key, value)
        ]
        attention: ContextParallelAttention = ContextParallelAttention(mesh)
        output: torch.Tensor = attention(*local)
        torch.testing.assert_close(output, reference.detach().chunk(2, dim=2)[rank], atol=1e-6, rtol=1e-5)
        output.square().sum().backward()
        source: torch.Tensor
        shard: torch.Tensor
        for source, shard in zip((query, key, value), local, strict=True):
            torch.testing.assert_close(shard.grad, source.grad.chunk(2, dim=2)[rank], atol=2e-6, rtol=2e-5)

    def _loader(self, context: bool = False) -> DataLoader:
        if context:
            inputs: torch.Tensor = torch.arange(256, dtype=torch.float32).reshape(8, 8, 4) / 256
            return DataLoader(TensorDataset(inputs, inputs * 0.2), batch_size=2)
        inputs: torch.Tensor = torch.arange(128, dtype=torch.float32).reshape(32, 4) / 128
        return DataLoader(TensorDataset(inputs, inputs[:, :3] * 0.4), batch_size=8)

    def _trainer(
        self,
        strategy: ContextParallelStrategy | PipelineParallelStrategy,
        steps: int,
        directory: Path
    ) -> Trainer:
        return Trainer(
            accelerator="cpu", devices=2, strategy=strategy, max_epochs=2, max_steps=steps,
            callbacks=[ModelCheckpoint(dirpath=directory, save_last=True, save_top_k=-1, every_n_train_steps=1)],
            logger=False, enable_progress_bar=False, enable_model_summary=False,
            num_sanity_val_steps=0, limit_val_batches=0, seed=700 + dist.get_rank(),
            use_distributed_sampler=False
        )

    def _context_training(self) -> None:
        reference: ContextModule = ContextModule()
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(reference.parameters(), lr=0.003)
        batch: tuple[torch.Tensor, torch.Tensor]
        for batch in self._loader(True):
            optimizer.zero_grad()
            torch.nn.functional.mse_loss(reference(batch[0]), batch[1]).backward()
            optimizer.step()
        policy: ContextParallelConfiguration = ContextParallelConfiguration(context_parallel_size=2)
        baseline: ContextModule = ContextModule()
        trainer: Trainer = self._trainer(
            ContextParallelStrategy(policy, checkpoint_write="asynchronous", process_group_backend="gloo"),
            4, self._configuration.artifact_root / "context-reference"
        )
        trainer.fit(baseline, train_dataloaders=self._loader(True))
        expected: torch.Tensor
        actual: torch.Tensor
        for expected, actual in zip(reference.parameters(), baseline.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        random_reference: torch.Tensor = torch.rand(8)
        partial: ContextModule = ContextModule()
        directory: Path = self._configuration.artifact_root / "context-resume"
        self._trainer(
            ContextParallelStrategy(policy, checkpoint_write="asynchronous", process_group_backend="gloo"), 2, directory
        ).fit(partial, train_dataloaders=self._loader(True))
        restored: ContextModule = ContextModule()
        resumed: Trainer = self._trainer(
            ContextParallelStrategy(policy, checkpoint_write="asynchronous", process_group_backend="gloo"), 4, directory
        )
        resumed.fit(restored, train_dataloaders=self._loader(True), ckpt_path="last")
        assert resumed.global_step == 4
        for expected, actual in zip(baseline.parameters(), restored.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(torch.rand(8), random_reference, atol=0, rtol=0)

    def _pipeline_policy(self, schedule: Literal["gpipe", "1f1b"]) -> PipelineParallelConfiguration:
        return PipelineParallelConfiguration(
            stages=(
                PipelineStageShape(input_shape=(2, 4), output_shape=(2, 6)),
                PipelineStageShape(input_shape=(2, 6), output_shape=(2, 3))
            ),
            n_microbatches=4, schedule=schedule, loss_id="mse:v1"
        )

    def _pipeline_training(self, schedule: Literal["gpipe", "1f1b"]) -> None:
        reference: nn.Sequential = nn.Sequential(PipelineModule(0), PipelineModule(1))
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(reference.parameters(), lr=0.003)
        batch: tuple[torch.Tensor, torch.Tensor]
        for batch in self._loader():
            optimizer.zero_grad()
            torch.nn.functional.mse_loss(reference(batch[0]), batch[1]).backward()
            optimizer.step()
        policy: PipelineParallelConfiguration = self._pipeline_policy(schedule)
        module: PipelineModule = PipelineModule(dist.get_rank())
        strategy: PipelineParallelStrategy = PipelineParallelStrategy(
            policy, nn.MSELoss(), checkpoint_write="asynchronous", process_group_backend="gloo"
        )
        trainer: Trainer = self._trainer(strategy, 4, self._configuration.artifact_root / f"{schedule}-sequential")
        trainer.fit(module, train_dataloaders=self._loader())
        expected: torch.Tensor
        actual: torch.Tensor
        for expected, actual in zip(reference[dist.get_rank()].parameters(), module.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        baseline: PipelineModule = PipelineModule(dist.get_rank(), 0.2)
        self._trainer(
            PipelineParallelStrategy(policy, nn.MSELoss(), process_group_backend="gloo"),
            4, self._configuration.artifact_root / f"{schedule}-stochastic"
        ).fit(baseline, train_dataloaders=self._loader())
        random_reference: torch.Tensor = torch.rand(8)
        directory: Path = self._configuration.artifact_root / f"{schedule}-resume"
        partial: PipelineModule = PipelineModule(dist.get_rank(), 0.2)
        self._trainer(
            PipelineParallelStrategy(policy, nn.MSELoss(), process_group_backend="gloo"), 2, directory
        ).fit(partial, train_dataloaders=self._loader())
        restored: PipelineModule = PipelineModule(dist.get_rank(), 0.2)
        resumed: Trainer = self._trainer(
            PipelineParallelStrategy(policy, nn.MSELoss(), process_group_backend="gloo"), 4, directory
        )
        resumed.fit(restored, train_dataloaders=self._loader(), ckpt_path="last")
        assert resumed.global_step == 4
        for expected, actual in zip(baseline.parameters(), restored.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(torch.rand(8), random_reference, atol=0, rtol=0)


if __name__ == "__main__":
    probe: ParallelAlgorithmProbe = ParallelAlgorithmProbe()
    probe.run()
```
