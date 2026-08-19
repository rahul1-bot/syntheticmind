from collections.abc import Iterable
from importlib.util import find_spec
from pathlib import Path

from syntheticmind.loggers.csv_logger import CSVLogger
from syntheticmind.loggers.logger import Logger, _NoOpLogger
from syntheticmind.loggers.tensorboard import TensorBoardLogger
from syntheticmind.utilities.filesystem import basename, is_directory, join_path, list_directory

__all__: list[str] = ["resolve_loggers"]


def resolve_loggers(
    logger: bool | Logger | Iterable[Logger] | None,
    default_root_dir: str | Path,
    fast_dev_run: bool
) -> list[Logger]:
    if logger is None or logger is True:
        if fast_dev_run:
            return [_NoOpLogger()]
        return [_default_logger(default_root_dir)]
    if logger is False:
        return []
    if isinstance(logger, Logger):
        return [_NoOpLogger()] if fast_dev_run else [logger]
    if not isinstance(logger, Iterable):
        raise TypeError(
            "`logger` must be a Logger, an iterable of Logger instances, "
            "True, False, or None."
        )
    loggers: list[Logger] = list(logger)
    for index, configured_logger in enumerate(loggers):
        if not isinstance(configured_logger, Logger):
            raise TypeError(
                f"`logger` item {index} must be a Logger, got "
                f"{type(configured_logger).__name__}."
            )
    if fast_dev_run and loggers:
        return [_NoOpLogger()]
    return loggers


def _default_logger(default_root_dir: str | Path) -> Logger:
    name: str = "syntheticmind_logs"
    version: str = _next_version(join_path(default_root_dir, name))
    if find_spec("tensorboard") is not None:
        return TensorBoardLogger(
            save_dir=default_root_dir,
            name=name,
            version=version
        )
    return CSVLogger(
        save_dir=default_root_dir,
        name=name,
        version=version
    )


def _next_version(root_dir: str | Path) -> str:
    versions: list[int] = []
    for path in list_directory(root_dir):
        name: str = basename(path)
        if not is_directory(path) or not name.startswith("version_"):
            continue
        suffix: str = name.removeprefix("version_")
        if suffix.isdigit():
            versions.append(int(suffix))
    return f"version_{max(versions, default=-1) + 1}"
