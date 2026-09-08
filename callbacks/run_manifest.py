import importlib.metadata as metadata
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, Literal, override

import syntheticmind
import torch
from pydantic import BaseModel, ConfigDict

from syntheticmind.callbacks.callback import Callback
from syntheticmind.plugins.torch_elastic import TorchElasticEnvironment
from syntheticmind.strategies.compiler import TorchCompiler
from syntheticmind.strategies.parallel import ParallelStrategy

__all__: list[str] = ["RunManifest", "RunManifestCallback"]


class RunManifest(BaseModel):
    schema_version: Literal[1] = 1
    created_at_utc: datetime
    python_version: str
    torch_version: str
    operating_system: str
    machine_architecture: str
    module_class: str
    accelerator_class: str
    strategy_class: str
    devices: tuple[str, ...]
    world_size: int
    precision: str
    maximum_epochs: int
    minimum_epochs: int
    maximum_steps: int
    minimum_steps: int | None
    accumulate_grad_batches: int
    compiler_backend: str | None
    compiler_mode: str | None
    compiler_graph_break_policy: str | None
    compiler_shape_policy: str | None
    elastic_run_id: str | None
    elastic_restart_count: int | None
    elastic_maximum_restarts: int | None
    git_commit: str | None
    git_is_dirty: bool | None
    dependency_versions: tuple[str, ...]
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        validate_default=True
    )


class RunManifestCallback(Callback):
    def __init__(
        self,
        dirpath: Path,
        repository_root: Path | None = None,
        filename: str = "run_manifest.json"
    ) -> None:
        if not filename or Path(filename).name != filename:
            raise ValueError(
                f"Run manifest filename must be one file name, got {filename!r}."
            )
        self._dirpath: Path = Path(dirpath)
        self._repository_root: Path = (
            Path.cwd() if repository_root is None else Path(repository_root)
        )
        self._filename: str = filename

    @override
    def on_fit_start(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module
    ) -> None:
        if not trainer.strategy.is_global_zero:
            return
        self._dirpath.mkdir(parents=True, exist_ok=True)
        manifest: RunManifest = self._build_manifest(trainer, module)
        manifest_path: Path = self._dirpath / self._filename
        manifest_path.write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8"
        )

    def _build_manifest(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module
    ) -> RunManifest:
        compiler: TorchCompiler | None = trainer.strategy.compiler
        elastic_environment: TorchElasticEnvironment | None = (
            trainer.strategy.cluster_environment
            if isinstance(trainer.strategy, ParallelStrategy)
            and isinstance(
                trainer.strategy.cluster_environment,
                TorchElasticEnvironment
            )
            else None
        )
        return RunManifest(
            created_at_utc=datetime.now(timezone.utc),
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            operating_system=platform.platform(),
            machine_architecture=platform.machine(),
            module_class=f"{type(module).__module__}.{type(module).__qualname__}",
            accelerator_class=type(trainer.accelerator).__name__,
            strategy_class=type(trainer.strategy).__name__,
            devices=tuple(str(device) for device in trainer.strategy.parallel_devices),
            world_size=trainer.strategy.world_size,
            precision=trainer.precision,
            maximum_epochs=trainer.max_epochs,
            minimum_epochs=trainer.min_epochs,
            maximum_steps=trainer.max_steps,
            minimum_steps=trainer.min_steps,
            accumulate_grad_batches=trainer.accumulate_grad_batches,
            compiler_backend=compiler.backend if compiler is not None else None,
            compiler_mode=compiler.mode if compiler is not None else None,
            compiler_graph_break_policy=(
                compiler.graph_break_policy if compiler is not None else None
            ),
            compiler_shape_policy=(
                compiler.shape_policy if compiler is not None else None
            ),
            elastic_run_id=(
                elastic_environment.run_id
                if elastic_environment is not None
                else None
            ),
            elastic_restart_count=(
                elastic_environment.restart_count
                if elastic_environment is not None
                else None
            ),
            elastic_maximum_restarts=(
                elastic_environment.maximum_restarts
                if elastic_environment is not None
                else None
            ),
            git_commit=self._git_commit(),
            git_is_dirty=self._git_is_dirty(),
            dependency_versions=self._dependency_versions()
        )

    def _git_commit(self) -> str | None:
        try:
            result: subprocess.CompletedProcess[str] = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self._repository_root,
                check=True,
                capture_output=True,
                text=True
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        commit: str = result.stdout.strip()
        return commit or None

    def _git_is_dirty(self) -> bool | None:
        try:
            result: subprocess.CompletedProcess[str] = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self._repository_root,
                check=True,
                capture_output=True,
                text=True
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return bool(result.stdout.strip())

    def _dependency_versions(self) -> tuple[str, ...]:
        package_names: tuple[str, ...] = (
            "fsspec",
            "numpy",
            "pydantic",
            "torch",
            "torchao"
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
