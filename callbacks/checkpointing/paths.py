import re
from pathlib import Path
from typing import ClassVar

import syntheticmind
import torch

from syntheticmind.exceptions import MisconfigurationError
from syntheticmind.storage.filesystem import (
    exists,
    is_local_path,
    is_path_within,
    join_path,
    normalize_path,
    same_path
)

__all__: list[str] = []


class _CheckpointPaths:
    _join_character: ClassVar[str] = "-"
    _equals_character: ClassVar[str] = "="
    _extension: ClassVar[str] = ".ckpt"
    _starting_version: ClassVar[int] = 1

    def __init__(
        self,
        dirpath: str | Path | None,
        filename: str | None,
        auto_insert_metric_name: bool,
        enable_version_counter: bool
    ) -> None:
        self.dirpath: str | None = self._normalize_directory(dirpath)
        self.filename: str | None = filename
        self.auto_insert_metric_name: bool = auto_insert_metric_name
        self.enable_version_counter: bool = enable_version_counter

    def resolve_directory(self, trainer: syntheticmind.Trainer) -> str:
        if self.dirpath is None:
            root: str = (
                trainer.log_dir
                if trainer.loggers and not trainer.fast_dev_run
                else trainer.default_root_dir
            )
            self.dirpath: str | None = join_path(root, "checkpoints")
        return self.dirpath

    def format_checkpoint_name(
        self,
        metrics: dict[str, torch.Tensor],
        filename: str | None = None,
        version: int | None = None,
        prefix: str | None = None
    ) -> str:
        resolved_filename: str = self._format_name(
            filename or self.filename,
            dict(metrics),
            prefix
        )
        if version is not None:
            resolved_filename: str = self._join_character.join(
                (resolved_filename, f"v{version}")
            )
        relative_path: Path = Path(f"{resolved_filename}{self._extension}")
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise MisconfigurationError(
                f"Checkpoint filename escapes its directory: {relative_path}."
            )
        filepath: str = (
            join_path(self.dirpath, str(relative_path))
            if self.dirpath is not None
            else str(relative_path)
        )
        if (
            self.dirpath is not None
            and is_local_path(self.dirpath)
            and not is_path_within(filepath, self.dirpath)
        ):
            raise MisconfigurationError(
                f"Checkpoint path {filepath!r} escapes directory {self.dirpath!r}."
            )
        return filepath

    def ranked_filepath(
        self,
        candidates: dict[str, torch.Tensor],
        trainer: syntheticmind.Trainer | None = None
    ) -> str:
        filepath: str = self.format_checkpoint_name(candidates)
        if not self.enable_version_counter:
            return filepath
        version: int = self._starting_version
        while self._exists(filepath, trainer):
            filepath: str = self.format_checkpoint_name(
                candidates,
                version=version
            )
            version += 1
        return filepath

    def last_filepath(
        self,
        candidates: dict[str, torch.Tensor],
        previous_path: str,
        trainer: syntheticmind.Trainer | None = None
    ) -> str:
        filepath: str = self.format_checkpoint_name(candidates, "last")
        if not self.enable_version_counter:
            return filepath
        version: int = self._starting_version
        while self._exists(filepath, trainer) and filepath != previous_path:
            filepath: str = self.format_checkpoint_name(
                candidates,
                "last",
                version
            )
            version += 1
        return filepath

    def should_remove(
        self,
        trainer: syntheticmind.Trainer,
        previous: str,
        current: str
    ) -> bool:
        if previous == current:
            return False
        if not is_local_path(previous):
            return True
        resume_path: str | Path | None = (
            trainer.ckpt_path
            if trainer.ckpt_path is not None
            else trainer.resume_from_checkpoint
        )
        if resume_path is not None and same_path(previous, str(resume_path)):
            return False
        return self.dirpath is not None and is_path_within(previous, self.dirpath)

    def link_checkpoint(
        self,
        trainer: syntheticmind.Trainer,
        filepath: str | Path,
        linkpath: str | Path,
        synchronize: bool = True
    ) -> None:
        if not is_local_path(filepath) or not is_local_path(linkpath):
            raise ValueError("Checkpoint links require local paths.")
        source_path: Path = Path(normalize_path(filepath))
        target_path: Path = Path(normalize_path(linkpath))
        if not trainer.is_global_zero:
            if synchronize:
                trainer.strategy.barrier()
            return
        try:
            if source_path.resolve() == target_path.resolve():
                return
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists() and target_path.is_dir() and not target_path.is_symlink():
                raise FileExistsError(
                    f"Cannot replace checkpoint directory {target_path} with a link."
                )
            if target_path.is_symlink() or target_path.is_file():
                target_path.unlink()
            relative_source: Path = source_path.relative_to(
                target_path.parent,
                walk_up=True
            )
            try:
                target_path.symlink_to(
                    relative_source,
                    target_is_directory=source_path.is_dir()
                )
            except OSError:
                source_path.copy(target_path)
        finally:
            if synchronize:
                trainer.strategy.barrier()

    def _format_name(
        self,
        filename: str | None,
        metrics: dict[str, torch.Tensor],
        prefix: str | None
    ) -> str:
        resolved_filename: str = filename or "{epoch}-{step}"
        groups: list[str] = sorted(
            re.findall(r"(\{.*?)[:\}]", resolved_filename),
            key=len,
            reverse=True
        )
        for group in groups:
            name: str = group[1:]
            if self.auto_insert_metric_name:
                resolved_filename: str = resolved_filename.replace(
                    group,
                    name + self._equals_character + "{" + name
                )
            resolved_filename: str = resolved_filename.replace(
                group,
                f"{{0[{name}]"
            )
            if name not in metrics:
                metrics[name] = torch.tensor(0)
        resolved_filename: str = resolved_filename.format(metrics)
        if prefix is not None:
            resolved_filename: str = self._join_character.join(
                (prefix, resolved_filename)
            )
        return resolved_filename

    def _exists(
        self,
        filepath: str | Path,
        trainer: syntheticmind.Trainer | None
    ) -> bool:
        if trainer is None:
            return exists(filepath)
        path_exists: bool = exists(filepath) if trainer.is_global_zero else False
        return trainer.strategy.broadcast(path_exists)

    def _normalize_directory(self, dirpath: str | Path | None) -> str | None:
        if dirpath is None:
            return None
        normalized_path: str = normalize_path(dirpath)
        if is_local_path(normalized_path):
            return str(Path(normalized_path).expanduser().resolve())
        return normalized_path
