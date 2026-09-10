# This module:
# 1. Records versioned run provenance without copying environment secrets or code
# 2. Publishes local JSON atomically and preserves separate fit/restart attempts
#
# Design decisions:
# - Rank zero owns records; each fit receives an independent UUIDv7 directory
# - Compiler policy is the same immutable contract used by checkpoint validation
# - Complete Trainer policy is taken from an assembled checkpoint, not guessed
# - A manifest observes configuration; its presence does not prove checkpoint commit
# - Data identity is supplied by the researcher; dataset contents are not scanned
#
# Author: Rahul Sawhney

import importlib.metadata as metadata
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, ClassVar, Literal, override
from uuid import UUID, uuid7

import syntheticmind
import torch
from pydantic import BaseModel, ConfigDict, Field

from syntheticmind.callbacks.callback import Callback
from syntheticmind.plugins.torch_elastic import TorchElasticEnvironment
from syntheticmind.primitives.types import CheckpointDict, CheckpointValue, PrecisionType
from syntheticmind.strategies.compiler import CompilationConfiguration, TorchCompiler
from syntheticmind.strategies.parallel import ParallelStrategy
from syntheticmind.trainer.configuration import _TrainerConfigurationSnapshot

__all__: list[str] = ["RunManifest", "RunManifestCallback"]


class RunManifest(BaseModel):
    # Durable observation, not live Trainer state or a complete reconstruction of
    # client code and datasets. Null fields denote unavailable observations only.
    schema_version: Literal[2] = 2
    run_id: UUID
    parent_run_id: UUID | None
    created_at_utc: datetime
    python_version: str
    torch_version: str
    operating_system: str
    machine_architecture: str
    module_class: str
    accelerator_class: str
    strategy_class: str
    devices: tuple[str, ...]
    world_size: Annotated[int, Field(ge=1)]
    precision: PrecisionType
    maximum_epochs: Annotated[int, Field(ge=-1)]
    minimum_epochs: Annotated[int, Field(ge=0)]
    maximum_steps: Annotated[int, Field(ge=-1)]
    minimum_steps: Annotated[int, Field(ge=0)] | None
    accumulate_grad_batches: Annotated[int, Field(ge=1)]
    compiler_configuration: CompilationConfiguration | None
    elastic_run_id: str | None
    elastic_restart_count: Annotated[int, Field(ge=0)] | None
    elastic_maximum_restarts: Annotated[int, Field(ge=0)] | None
    git_commit: str | None
    git_is_dirty: bool | None
    data_identity: str | None
    dependency_versions: tuple[str, ...]
    resolved_trainer_configuration: _TrainerConfigurationSnapshot | None = None
    checkpoint_schema_version: Literal[2] | None = None
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", strict=True, validate_default=True
    )


class RunManifestCallback(Callback):
    # Supply an artifact directory and optional client Git/data identities. The
    # manifest_path property identifies this instance's latest rank-zero record.
    def __init__(
        self,
        dirpath: Path,
        repository_root: Path | None = None,
        filename: str = "run_manifest.json",
        *,
        data_identity: str | None = None
    ) -> None:
        if not isinstance(filename, str) or not filename or filename in {".", ".."}:
            raise ValueError("Run manifest filename must be a non-empty file name.")
        if Path(filename).name != filename or "/" in filename or "\\" in filename:
            raise ValueError("Run manifest filename cannot contain path separators.")
        if data_identity is not None and (not isinstance(data_identity, str) or not data_identity.strip()):
            raise ValueError("data_identity must be a non-empty identifier or None.")
        self._dirpath: Path = Path(dirpath)
        self._repository_root: Path = Path.cwd() if repository_root is None else Path(repository_root)
        self._filename: str = filename
        self._data_identity: str | None = data_identity
        self._manifest: RunManifest | None = None
        self._manifest_path: Path | None = None
        self._parent_run_id: UUID | None = None

    @override
    def on_fit_start(self, trainer: syntheticmind.Trainer, module: syntheticmind.Module) -> None:
        if not trainer.strategy.is_global_zero:
            return
        manifest: RunManifest = self._build_manifest(trainer, module)
        self._manifest_path: Path | None = self._dirpath / str(manifest.run_id) / self._filename
        self._manifest: RunManifest | None = manifest
        self._write_manifest(manifest)

    @override
    def on_save_checkpoint(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module,
        checkpoint: CheckpointDict
    ) -> None:
        # This is an assembly hook, not a durability notification. The existing
        # checkpoint schema owns the resolved configuration being projected here.
        del module
        if not trainer.is_global_zero or self._manifest is None:
            return
        configuration: _TrainerConfigurationSnapshot = _TrainerConfigurationSnapshot.model_validate(
            checkpoint.get("trainer_configuration")
        )
        manifest: RunManifest = RunManifest.model_validate(
            self._manifest.model_dump(mode="python")
            | {"resolved_trainer_configuration": configuration, "checkpoint_schema_version": 2}
        )
        self._write_manifest(manifest)
        self._manifest: RunManifest | None = manifest
        checkpoint["run_manifest_id"]: CheckpointValue = str(manifest.run_id)

    @override
    def on_load_checkpoint(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module,
        checkpoint: CheckpointDict
    ) -> None:
        del trainer, module
        value: CheckpointValue | None = checkpoint.get("run_manifest_id")
        if value is not None and not isinstance(value, str):
            raise TypeError("Checkpoint run_manifest_id must be a UUID string.")
        self._parent_run_id: UUID | None = UUID(value) if value is not None else None

    @override
    def teardown(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module,
        stage: Literal["fit", "validate", "test", "predict"]
    ) -> None:
        del trainer, module, stage
        self._manifest: RunManifest | None = None
        self._parent_run_id: UUID | None = None

    @property
    def manifest_path(self) -> Path | None:
        return self._manifest_path

    def _build_manifest(self, trainer: syntheticmind.Trainer, module: syntheticmind.Module) -> RunManifest:
        compiler: TorchCompiler | None = trainer.strategy.compiler
        elastic: TorchElasticEnvironment | None = (
            trainer.strategy.cluster_environment
            if isinstance(trainer.strategy, ParallelStrategy)
            and isinstance(trainer.strategy.cluster_environment, TorchElasticEnvironment)
            else None
        )
        commit: str | None = self._git_output(["rev-parse", "HEAD"])
        status: str | None = self._git_output(["status", "--porcelain"])
        device: torch.device
        return RunManifest(
            run_id=uuid7(), parent_run_id=self._parent_run_id,
            created_at_utc=datetime.now(timezone.utc), python_version=platform.python_version(),
            torch_version=str(torch.__version__), operating_system=platform.platform(),
            machine_architecture=platform.machine(),
            module_class=f"{type(module).__module__}.{type(module).__qualname__}",
            accelerator_class=type(trainer.accelerator).__name__, strategy_class=type(trainer.strategy).__name__,
            devices=tuple(str(device) for device in trainer.strategy.parallel_devices),
            world_size=trainer.strategy.world_size, precision=trainer.precision,
            maximum_epochs=trainer.max_epochs, minimum_epochs=trainer.min_epochs,
            maximum_steps=trainer.max_steps, minimum_steps=trainer.min_steps,
            accumulate_grad_batches=trainer.accumulate_grad_batches,
            compiler_configuration=compiler.configuration if compiler is not None else None,
            elastic_run_id=elastic.run_id if elastic is not None else None,
            elastic_restart_count=elastic.restart_count if elastic is not None else None,
            elastic_maximum_restarts=elastic.maximum_restarts if elastic is not None else None,
            git_commit=commit or None, git_is_dirty=bool(status) if status is not None else None,
            data_identity=self._data_identity, dependency_versions=self._dependency_versions()
        )

    def _write_manifest(self, manifest: RunManifest) -> None:
        path: Path | None = self._manifest_path
        if path is None:
            raise RuntimeError("Run manifest path is unavailable before fit starts.")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
                temporary_path: Path | None = Path(stream.name)
                stream.write(manifest.model_dump_json(indent=2))
                stream.flush()
                os.fsync(stream.fileno())
            temporary_path.replace(path)
            if os.name == "posix":
                descriptor: int = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def _git_output(self, arguments: list[str]) -> str | None:
        try:
            result: subprocess.CompletedProcess[str] = subprocess.run(
                ["git", *arguments], cwd=self._repository_root,
                check=True, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    def _dependency_versions(self) -> tuple[str, ...]:
        package_names: tuple[str, ...] = (
            "fsspec", "loguru", "numpy", "packaging", "pydantic", "pyyaml",
            "rich", "tensorboard", "torch", "torchao", "torchmetrics"
        )
        versions: list[str] = []
        package_name: str
        for package_name in package_names:
            try:
                package_version: str = metadata.version(package_name)
            except metadata.PackageNotFoundError:
                continue
            versions.append(f"{package_name}=={package_version}")
        return tuple(versions)
