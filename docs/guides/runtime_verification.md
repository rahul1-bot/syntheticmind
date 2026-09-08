# Runtime control verification

These focused runtime probes are executable source examples, not a second test
framework. The `Runtime control plane` workflow extracts the three Python blocks
into its temporary workspace as `runtime_probe_1.py`, `runtime_probe_2.py`, and
`runtime_probe_3.py`. Run them from a directory containing the `syntheticmind/`
package. They use Python 3.14, PyTorch 2.14 CPU, Pydantic 2.13.5, and TorchAO 0.18.
No linter or static type checker is invoked.

```sh
python runtime_probe_1.py
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 runtime_probe_2.py --artifact-root /tmp/syntheticmind-probe
python runtime_probe_3.py
```

The first probe checks real single-process stages and CPU Inductor execution.
The second checks real two-rank Gloo DDP, tensor-parallel continuation,
asynchronous DCP, snapshot isolation, failure publication, retention, and mesh
admission. The third checks scheduler input boundaries and the optional TorchAO
adapter without pretending CPU execution proves CUDA FP8 correctness.

Use a fresh artifact directory for each distributed probe invocation. Injected
failures are intentional. The workflow log, commit SHA, and recorded dependencies
are the authoritative result; the presence of these examples alone is not a pass.

## Single-process stages

```python
import tempfile
from pathlib import Path
from typing import cast, override

import torch
from loguru import logger
from pydantic import ValidationError
from torch.utils.data import DataLoader, TensorDataset

from syntheticmind import Module, Trainer
from syntheticmind.callbacks.callback import Callback
from syntheticmind.callbacks.checkpointing import ModelCheckpoint
from syntheticmind.callbacks.cuda_memory_snapshot import CUDAMemorySnapshot
from syntheticmind.callbacks.run_manifest import RunManifest, RunManifestCallback
from syntheticmind.core.activation_checkpointing import ActivationCheckpointing
from syntheticmind.primitives.exceptions import MisconfigurationError
from syntheticmind.primitives.types import Batch, CheckpointDict, ModelOutput, StepArgument, StepOutput, TrainingStepOutput
from syntheticmind.strategies import CompilationConfiguration, SingleDeviceStrategy, TorchCompiler

__all__: list[str] = []


class Regressor(Module):
    def __init__(self) -> None:
        super().__init__()
        self.network: torch.nn.Linear = torch.nn.Linear(4, 4)
        self.calls: int = 0
        self.register_forward_hook(self._observe)

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

    @override
    def training_step(self, batch: Batch, batch_idx: int) -> TrainingStepOutput:
        del batch_idx
        inputs: torch.Tensor
        targets: torch.Tensor
        inputs, targets = cast(tuple[torch.Tensor, torch.Tensor], batch)
        loss: torch.Tensor = torch.nn.functional.mse_loss(self(inputs), targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss}

    @override
    def validation_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> StepOutput:
        del batch_idx, dataloader_idx
        inputs: torch.Tensor
        targets: torch.Tensor
        inputs, targets = cast(tuple[torch.Tensor, torch.Tensor], batch)
        loss: torch.Tensor = torch.nn.functional.mse_loss(self(inputs), targets)
        self.log("val_loss", loss, on_epoch=True)
        return {"val_loss": loss}

    @override
    def test_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> StepOutput:
        del batch_idx, dataloader_idx
        inputs: torch.Tensor
        targets: torch.Tensor
        inputs, targets = cast(tuple[torch.Tensor, torch.Tensor], batch)
        loss: torch.Tensor = torch.nn.functional.mse_loss(self(inputs), targets)
        self.log("test_loss", loss, on_epoch=True)
        return {"test_loss": loss}

    @override
    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> ModelOutput:
        del batch_idx, dataloader_idx
        inputs: torch.Tensor
        targets: torch.Tensor
        inputs, targets = cast(tuple[torch.Tensor, torch.Tensor], batch)
        del targets
        return self(inputs)

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=0.01)

    def _observe(self, module: torch.nn.Module, inputs: tuple[StepArgument, ...], output: ModelOutput) -> None:
        del module, inputs, output
        self.calls: int = self.calls + 1


class SingleProcessProbe:
    def run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root: Path = Path(directory)
            self._compiler_and_resume(root)
            self._activation()
            self._validation(root)
        logger.info("PASS: CPU fit/validate/test/predict, compiler identity/hooks/resume, activation gradients, manifest, guards")

    def _trainer(self, steps: int, compiler: TorchCompiler | None, callbacks: list[Callback] | None = None) -> Trainer:
        strategy: SingleDeviceStrategy = SingleDeviceStrategy()
        strategy.attach_compiler(compiler)
        return Trainer(
            accelerator="cpu", strategy=strategy, max_epochs=2, max_steps=steps,
            callbacks=callbacks, logger=False, enable_checkpointing=False,
            enable_progress_bar=False, enable_model_summary=False,
            num_sanity_val_steps=0, seed=731
        )

    def _loader(self) -> DataLoader:
        inputs: torch.Tensor = torch.arange(64, dtype=torch.float32).reshape(16, 4) / 64
        return DataLoader(TensorDataset(inputs, inputs * 0.4), batch_size=4)

    def _model(self) -> Regressor:
        torch.manual_seed(123)
        return Regressor()

    def _compiler_and_resume(self, root: Path) -> None:
        loader: DataLoader = self._loader()
        baseline: Regressor = self._model()
        eager: Trainer = self._trainer(4, None)
        eager.fit(baseline, train_dataloaders=loader)
        compiler: TorchCompiler = TorchCompiler(backend="eager", graph_break_policy="forbid")
        module: Regressor = self._model()
        manifest: RunManifestCallback = RunManifestCallback(root / "records", data_identity="synthetic-v1")
        trainer: Trainer = self._trainer(2, compiler, [manifest])
        parameter: torch.nn.Parameter
        name: str
        identities: list[int] = [id(parameter) for parameter in module.parameters()]
        trainer.fit(module, train_dataloaders=loader)
        assert not compiler.is_active and "forward" not in vars(module)
        assert identities == [id(parameter) for parameter in module.parameters()]
        assert module.calls == 2
        checkpoint: Path = root / "partial.ckpt"
        trainer.save_checkpoint(checkpoint)
        saved: CheckpointDict = cast(CheckpointDict, torch.load(checkpoint, weights_only=False))
        assert isinstance(saved["compiler_configuration"], dict)
        assert isinstance(saved["state_dict"], dict)
        assert saved["compiler_configuration"]["backend"] == "eager"
        assert all("_orig_mod" not in name for name in saved["state_dict"])
        restored: Regressor = self._model()
        resumed: Trainer = self._trainer(4, TorchCompiler(backend="eager", graph_break_policy="forbid"))
        resumed.fit(restored, train_dataloaders=loader, ckpt_path=checkpoint)
        assert resumed.global_step == 4
        expected: torch.Tensor
        actual: torch.Tensor
        for expected, actual in zip(baseline.parameters(), restored.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert resumed.validate(restored, dataloaders=loader)
        assert resumed.test(restored, dataloaders=loader)
        assert len(resumed.predict(restored, dataloaders=loader)) == 4
        rejected: Regressor = self._model()
        original: torch.Tensor = rejected.network.weight.detach().clone()
        try:
            self._trainer(4, None).fit(rejected, train_dataloaders=loader, ckpt_path=checkpoint)
        except MisconfigurationError:
            torch.testing.assert_close(rejected.network.weight, original)
        else:
            raise AssertionError("Changed compiler policy was not rejected.")
        assert manifest.manifest_path is not None
        record: RunManifest = RunManifest.model_validate_json(manifest.manifest_path.read_text())
        assert record.schema_version == 2 and record.data_identity == "synthetic-v1"
        assert record.compiler_configuration == compiler.configuration
        manifest_trainer: Trainer = Trainer(
            accelerator="cpu", max_steps=1, max_epochs=1, logger=False,
            callbacks=[manifest, ModelCheckpoint(
                dirpath=root / "manifest-checkpoints", save_last=True,
                every_n_train_steps=1, save_top_k=-1
            )],
            enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0
        )
        old_path: Path = manifest.manifest_path
        manifest_trainer.fit(self._model(), train_dataloaders=loader)
        assert old_path.exists() and manifest.manifest_path != old_path
        updated: RunManifest = RunManifest.model_validate_json(manifest.manifest_path.read_text())
        assert updated.resolved_trainer_configuration is not None
        native: Regressor = self._model()
        compiled: Trainer = self._trainer(4, TorchCompiler(backend="inductor"))
        compiled.fit(native, train_dataloaders=loader)
        for expected, actual in zip(baseline.parameters(), native.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def _activation(self) -> None:
        torch.manual_seed(29)
        plain: torch.nn.Sequential = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 4))
        inputs: torch.Tensor = torch.randn(2, 4, requires_grad=True)
        plain(inputs).square().sum().backward()
        parameter: torch.nn.Parameter
        reference: list[torch.Tensor] = [parameter.grad.clone() for parameter in plain.parameters()]
        plain.zero_grad(set_to_none=True)
        wrapped: ActivationCheckpointing = ActivationCheckpointing(
            plain, saved_operations=tuple([torch.ops.aten.mm.default])
        )
        wrapped(inputs).square().sum().backward()
        expected: torch.Tensor
        parameter: torch.nn.Parameter
        for expected, parameter in zip(reference, plain.parameters(), strict=True):
            torch.testing.assert_close(parameter.grad, expected)
        with torch.no_grad():
            torch.testing.assert_close(wrapped(inputs), plain(inputs))

    def _validation(self, root: Path) -> None:
        try:
            TorchCompiler(module_paths=("network", "network.child"))
        except ValidationError:
            pass
        else:
            raise AssertionError("Overlapping compiler regions were accepted.")
        policy: CompilationConfiguration = TorchCompiler().configuration
        assert CompilationConfiguration.model_validate_json(policy.model_dump_json()) == policy
        try:
            RunManifestCallback(root, filename="..")
        except ValueError:
            pass
        else:
            raise AssertionError("Manifest path traversal was accepted.")
        guarded: CUDAMemorySnapshot = CUDAMemorySnapshot(root / "cuda")
        try:
            self._trainer(1, None, [guarded]).fit(self._model(), train_dataloaders=self._loader())
        except RuntimeError as exception:
            assert "CUDA" in str(exception)
        else:
            raise AssertionError("CUDA callback accepted a CPU Strategy.")
        assert not (root / "cuda").exists()


if __name__ == "__main__":
    probe: SingleProcessProbe = SingleProcessProbe()
    probe.run()
```


## Two-rank execution and checkpoint transactions

```python
import argparse
from datetime import timedelta
from pathlib import Path
from threading import Event
from typing import ClassVar, cast, override
from unittest.mock import patch

import torch
import torch.distributed as dist
from loguru import logger
from pydantic import BaseModel, ConfigDict
from torch.distributed.checkpoint import StorageWriter
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module
from torch.utils.data import DataLoader, TensorDataset

from syntheticmind import Trainer
from syntheticmind.callbacks.checkpointing import ModelCheckpoint
from syntheticmind.primitives.exceptions import CheckpointError, MisconfigurationError
from syntheticmind.primitives.types import CheckpointDict, DistributedCheckpointState
from syntheticmind.strategies import DDPStrategy, ModelParallelStrategy, TorchCompiler
from syntheticmind.strategies.model_parallel_checkpointing.asynchronous import _AsyncShardedCheckpointStore
from syntheticmind.strategies.model_parallel_checkpointing.sharded_store import _ShardedCheckpointStore
from runtime_probe_1 import Regressor

__all__: list[str] = []


class ProbeConfiguration(BaseModel):
    artifact_root: Path
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", strict=True, validate_default=True
    )


class ParallelRegressor(Regressor):
    @override
    def __init__(self) -> None:
        super().__init__()
        self._parallelized: bool = False

    @override
    def configure_model(self) -> None:
        if self._parallelized:
            return
        parallelize_module(
            self, self.device_mesh["tensor_parallel"],
            {"network": ColwiseParallel(output_layouts=Replicate())}
        )
        self._parallelized: bool = True


class GatedStore(_AsyncShardedCheckpointStore):
    @override
    def __init__(self) -> None:
        super().__init__()
        self.release: Event = Event()

    @override
    def _write_and_publish(
        self,
        state: DistributedCheckpointState,
        writer: StorageWriter,
        checkpoint_path: str,
        generation_name: str,
        global_rank: int
    ) -> None:
        if not self.release.wait(timeout=20):
            raise TimeoutError("Probe did not release checkpoint writer.")
        super()._write_and_publish(state, writer, checkpoint_path, generation_name, global_rank)


class DistributedProbe:
    def __init__(self) -> None:
        parser: argparse.ArgumentParser = argparse.ArgumentParser(suggest_on_error=True, color=False)
        parser.add_argument("--artifact-root", type=Path, required=True)
        arguments: argparse.Namespace = parser.parse_args()
        self._configuration: ProbeConfiguration = ProbeConfiguration(artifact_root=arguments.artifact_root)

    def run(self) -> None:
        dist.init_process_group("gloo", timeout=timedelta(seconds=45))
        try:
            assert dist.get_world_size() == 2
            self._configuration.artifact_root.mkdir(parents=True, exist_ok=True)
            self._ddp()
            self._model_parallel_resume()
            self._staging_and_failure()
            self._extended_mesh()
            logger.info(f"PASS rank {dist.get_rank()}: DDP, TP resume, async staging/collectives/failure/retention, mesh guards")
        finally:
            dist.destroy_process_group()

    def _loader(self) -> DataLoader:
        inputs: torch.Tensor = torch.arange(64, dtype=torch.float32).reshape(16, 4) / 64
        return DataLoader(TensorDataset(inputs, inputs * 0.4), batch_size=4)

    def _ddp(self) -> None:
        torch.manual_seed(123)
        module: Regressor = Regressor()
        strategy: DDPStrategy = DDPStrategy(process_group_backend="gloo")
        compiler: TorchCompiler = TorchCompiler(backend="eager", graph_break_policy="forbid")
        strategy.attach_compiler(compiler)
        trainer: Trainer = Trainer(
            accelerator="cpu", devices=2, strategy=strategy, max_steps=2, max_epochs=1,
            logger=False, enable_checkpointing=False, enable_model_summary=False,
            enable_progress_bar=False, num_sanity_val_steps=0, seed=731
        )
        trainer.fit(module, train_dataloaders=self._loader())
        assert trainer.global_step == 2 and module.calls >= 2 and not compiler.is_active
        rank: int
        gathered: list[torch.Tensor] = [torch.empty_like(module.network.weight) for rank in range(2)]
        dist.all_gather(gathered, module.network.weight.detach())
        torch.testing.assert_close(gathered[0], gathered[1], atol=0, rtol=0)
        assert trainer.validate(module, dataloaders=self._loader())
        assert trainer.test(module, dataloaders=self._loader())
        assert trainer.predict(module, dataloaders=self._loader())

    def _parallel_trainer(self, steps: int, path: Path) -> Trainer:
        strategy: ModelParallelStrategy = ModelParallelStrategy(
            1, 2, True, "gloo", checkpoint_write="asynchronous"
        )
        checkpoint: ModelCheckpoint = ModelCheckpoint(
            dirpath=path, save_last="link", save_top_k=-1, every_n_train_steps=1
        )
        return Trainer(
            accelerator="cpu", devices=2, strategy=strategy, max_epochs=2, max_steps=steps,
            callbacks=[checkpoint], logger=False, enable_progress_bar=False,
            enable_model_summary=False, num_sanity_val_steps=0, seed=731
        )

    def _model_parallel_resume(self) -> None:
        root: Path = self._configuration.artifact_root
        torch.manual_seed(123)
        baseline: ParallelRegressor = ParallelRegressor()
        reference_trainer: Trainer = self._parallel_trainer(4, root / "baseline")
        reference_trainer.fit(baseline, train_dataloaders=self._loader())
        reference: list[torch.Tensor] = self._full_parameters(baseline)
        torch.manual_seed(123)
        interrupted: ParallelRegressor = ParallelRegressor()
        first: Trainer = self._parallel_trainer(2, root / "resume")
        first.fit(interrupted, train_dataloaders=self._loader())
        assert (root / "resume" / "last.ckpt").is_symlink()
        restored: ParallelRegressor = ParallelRegressor()
        continuation: Trainer = self._parallel_trainer(4, root / "resume")
        continuation.fit(restored, train_dataloaders=self._loader(), ckpt_path="last")
        assert continuation.global_step == 4
        actual: list[torch.Tensor] = self._full_parameters(restored)
        expected_tensor: torch.Tensor
        actual_tensor: torch.Tensor
        for expected_tensor, actual_tensor in zip(reference, actual, strict=True):
            torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)

    def _full_parameters(self, model: ParallelRegressor) -> list[torch.Tensor]:
        result: list[torch.Tensor] = []
        parameter: torch.nn.Parameter
        for parameter in model.parameters():
            result.append(parameter.full_tensor().detach().clone() if isinstance(parameter, DTensor) else parameter.detach().clone())
        return result

    def _staging_and_failure(self) -> None:
        root: Path = self._configuration.artifact_root / "transaction"
        source: torch.Tensor = torch.tensor([1.0, 2.0])
        state: DistributedCheckpointState = {"state_dict": {"weight": source}}
        metadata: CheckpointDict = {"step": 1}
        synchronous: _ShardedCheckpointStore = _ShardedCheckpointStore()
        synchronous.save(state, metadata, root, dist.get_rank(), self._barrier, self._broadcast)
        previous: str = (root / "active_generation").read_text()
        asynchronous: GatedStore = GatedStore()
        asynchronous.save(state, {"step": 2}, root, dist.get_rank(), self._barrier, self._broadcast)
        source.fill_(99)
        training_value: torch.Tensor = torch.ones(1)
        dist.all_reduce(training_value)
        assert training_value.item() == 2
        assert (root / "active_generation").read_text() == previous
        asynchronous.release.set()
        assert asynchronous.is_checkpoint(root)
        loaded: DistributedCheckpointState = {"state_dict": {"weight": torch.zeros(2)}}
        asynchronous.load_state(loaded, root)
        torch.testing.assert_close(cast(torch.Tensor, loaded["state_dict"]["weight"]), torch.tensor([1.0, 2.0]))
        committed: str = (root / "active_generation").read_text()
        assert committed != previous
        # Only rank zero removes a retired artifact; other ranks can use the
        # training group while the checkpoint worker completes independently.
        retired: Path = self._configuration.artifact_root / "retired.txt"
        if dist.get_rank() == 0:
            retired.write_text("retired")
        dist.barrier()
        asynchronous.save(state, {"step": 3}, root, dist.get_rank(), self._barrier, self._broadcast)
        if dist.get_rank() == 0:
            asynchronous.remove(retired)
            assert not retired.exists()
        dist.barrier()
        asynchronous.teardown()
        committed: str = (root / "active_generation").read_text()
        failing: _AsyncShardedCheckpointStore = _AsyncShardedCheckpointStore()
        with patch.object(_AsyncShardedCheckpointStore, "_publish_generation", side_effect=OSError("injected publication failure")):
            failing.save(state, {"step": 4}, root, dist.get_rank(), self._barrier, self._broadcast)
            try:
                failing.is_checkpoint(root)
            except CheckpointError:
                pass
            else:
                raise AssertionError("Publication failure was swallowed.")
            try:
                failing.teardown()
            except CheckpointError:
                pass
        failing.teardown()
        assert (root / "active_generation").read_text() == committed
        assert synchronous.load_metadata(root, False)["step"] == 3
        assert dist.is_initialized()

    def _extended_mesh(self) -> None:
        strategy: ModelParallelStrategy = ModelParallelStrategy(1, 2)
        trainer: Trainer = Trainer(accelerator="cpu", devices=2, strategy=strategy, logger=False)
        module: ParallelRegressor = ParallelRegressor()
        strategy.connect(module)
        strategy.setup_environment()
        assert strategy.device_mesh.mesh_dim_names == ("data_parallel", "tensor_parallel")
        strategy.teardown()
        extended: ModelParallelStrategy = ModelParallelStrategy(1, 1, context_parallel_size=2)
        extended_trainer: Trainer = Trainer(
            accelerator="cpu", devices=2, strategy=extended, max_steps=1,
            logger=False, enable_checkpointing=False, enable_model_summary=False,
            enable_progress_bar=False, num_sanity_val_steps=0
        )
        try:
            extended_trainer.fit(ParallelRegressor(), train_dataloaders=self._loader())
        except MisconfigurationError as exception:
            assert "manual" in str(exception)
        else:
            raise AssertionError("Extended mesh silently accepted automatic optimization.")
        del trainer

    def _barrier(self) -> None:
        dist.barrier()

    def _broadcast(self, value: str | None, source_rank: int) -> str | None:
        values: list[str | None] = [value]
        dist.broadcast_object_list(values, src=source_rank)
        return values[0]


if __name__ == "__main__":
    probe: DistributedProbe = DistributedProbe()
    probe.run()
```


## Scheduler and optional-dependency boundaries

```python
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from syntheticmind import Trainer
from syntheticmind.callbacks.checkpointing import ModelCheckpoint
from syntheticmind.plugins import SLURMEnvironment, TorchElasticEnvironment
from syntheticmind.plugins.precision.torchao import TorchAOFloat8Precision
from syntheticmind.primitives.exceptions import MisconfigurationError
from syntheticmind.primitives.types import StateDict
from syntheticmind.strategies import DDPStrategy, ModelParallelStrategy
from runtime_probe_1 import Regressor

__all__: list[str] = []


class BoundaryProbe:
    def run(self) -> None:
        self._environments()
        self._elastic_resume_guard()
        self._precision()
        self._mesh_inputs()
        logger.info("PASS: Slurm/torchrun precedence, restart failure guard, float8 adapter CPU rejection/state, mesh inputs")

    def _environments(self) -> None:
        slurm: dict[str, str] = {
            "SLURM_NTASKS": "2", "SLURM_PROCID": "1", "SLURM_LOCALID": "1",
            "SLURM_NODEID": "0", "SLURM_JOB_ID": "42", "MASTER_ADDR": "localhost"
        }
        with patch.dict(os.environ, slurm, clear=True):
            environment: SLURMEnvironment = SLURMEnvironment()
            environment.validate_settings(2, 1)
            assert environment.global_rank() == 1 and environment.local_rank() == 1
            assert environment.main_port == 10042
            strategy: DDPStrategy = DDPStrategy()
            assert isinstance(strategy.cluster_environment, SLURMEnvironment)
        elastic: dict[str, str] = {
            "TORCHELASTIC_RUN_ID": "probe", "LOCAL_RANK": "0", "RANK": "0",
            "WORLD_SIZE": "2", "MASTER_ADDR": "localhost", "MASTER_PORT": "29598",
            "TORCHELASTIC_RESTART_COUNT": "1", "TORCHELASTIC_MAX_RESTARTS": "2"
        }
        with patch.dict(os.environ, slurm | elastic, clear=True):
            chosen: DDPStrategy = DDPStrategy()
            assert isinstance(chosen.cluster_environment, TorchElasticEnvironment)
            chosen.cluster_environment.validate_settings(2, 1)
            assert chosen.cluster_environment.restart_count == 1
        with patch.dict(os.environ, elastic | {"TORCHELASTIC_RESTART_COUNT": "-1"}, clear=True):
            try:
                TorchElasticEnvironment().validate_settings(2, 1)
            except ValueError:
                pass
            else:
                raise AssertionError("Negative restart count accepted.")

    def _elastic_resume_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment: dict[str, str] = {
                "TORCHELASTIC_RUN_ID": "probe", "LOCAL_RANK": "0", "RANK": "0",
                "WORLD_SIZE": "1", "MASTER_ADDR": "localhost", "MASTER_PORT": "29597",
                "TORCHELASTIC_RESTART_COUNT": "1", "TORCHELASTIC_MAX_RESTARTS": "2"
            }
            with patch.dict(os.environ, environment, clear=True):
                trainer: Trainer = Trainer(
                    accelerator="cpu", devices=1, strategy=DDPStrategy(), max_steps=1,
                    callbacks=[ModelCheckpoint(dirpath=Path(directory), save_last=True)],
                    logger=False, enable_progress_bar=False, enable_model_summary=False
                )
                inputs: torch.Tensor = torch.ones(4, 4)
                loader: DataLoader = DataLoader(TensorDataset(inputs, inputs), batch_size=4)
                try:
                    trainer.fit(Regressor(), train_dataloaders=loader)
                except FileNotFoundError as exception:
                    assert "restart" in str(exception)
                else:
                    raise AssertionError("Restart silently started from scratch.")
                assert not torch.distributed.is_initialized()

    def _precision(self) -> None:
        precision: TorchAOFloat8Precision = TorchAOFloat8Precision()
        state: StateDict = precision.state_dict()
        precision.load_state_dict(state)
        try:
            precision.load_state_dict({"schema_version": 1, "recipe": "tensorwise"})
        except ValueError:
            pass
        else:
            raise AssertionError("Changed float8 recipe accepted.")
        module: torch.nn.Linear = torch.nn.Linear(16, 16)
        original: torch.Tensor = module.weight.detach().clone()
        try:
            precision.convert_module(module)
        except RuntimeError:
            torch.testing.assert_close(module.weight, original)
            assert module.weight.dtype == torch.float32
        else:
            raise AssertionError("CPU conversion silently claimed float8 support.")
        try:
            Trainer(accelerator="cpu", plugins=[precision])
        except MisconfigurationError:
            pass
        else:
            raise AssertionError("CPU Trainer accepted CUDA float8 precision.")

    def _mesh_inputs(self) -> None:
        try:
            ModelParallelStrategy(context_parallel_size=2)
        except ValueError:
            pass
        else:
            raise AssertionError("Extended topology guessed automatic factors.")
        try:
            ModelParallelStrategy(1, 1, context_parallel_size=True)
        except ValueError:
            pass
        else:
            raise AssertionError("Bool accepted as a mesh size.")
        try:
            ModelParallelStrategy(1, 1, False, checkpoint_write="asynchronous")
        except ValueError:
            pass
        else:
            raise AssertionError("Asynchronous full checkpoint accepted.")


if __name__ == "__main__":
    probe: BoundaryProbe = BoundaryProbe()
    probe.run()
```
