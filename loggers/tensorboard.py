import json
from pathlib import Path
from typing import Protocol, override

from syntheticmind.loggers.logger import Logger, _JsonValue
from syntheticmind.storage.filesystem import is_local_path
from syntheticmind.types import HyperparameterDict

__all__: list[str] = ["TensorBoardLogger"]

type _TensorBoardHyperparameter = bool | int | float | str


class _TensorBoardWriter(Protocol):
    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None
    ) -> None:
        ...

    def add_hparams(
        self,
        hparam_dict: dict[str, _TensorBoardHyperparameter],
        metric_dict: dict[str, float]
    ) -> None:
        ...

    def flush(self) -> None:
        ...

    def close(self) -> None:
        ...


class TensorBoardLogger(Logger):
    def __init__(
        self,
        save_dir: str | Path,
        name: str = "default",
        version: int | str | None = None
    ) -> None:
        super().__init__(save_dir=save_dir, name=name, version=version)
        if not is_local_path(self.log_dir):
            raise ValueError("TensorBoardLogger requires a local filesystem directory.")
        self._writer: _TensorBoardWriter | None = None

    @override
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        validated_metrics: dict[str, float] = self._validated_metrics(metrics, step)
        writer: _TensorBoardWriter = self._get_writer()
        key: str
        value: float
        for key, value in validated_metrics.items():
            writer.add_scalar(key, value, global_step=step)

    @override
    def log_hyperparams(self, params: HyperparameterDict) -> None:
        normalized_params: dict[str, _JsonValue] = self._normalize_hyperparameters(
            params
        )
        tensorboard_params: dict[str, _TensorBoardHyperparameter] = {}
        name: str
        value: _JsonValue
        for name, value in normalized_params.items():
            if isinstance(value, bool | int | float | str):
                tensorboard_params[name]: _TensorBoardHyperparameter = value
            elif value is None:
                tensorboard_params[name]: _TensorBoardHyperparameter = "null"
            else:
                tensorboard_params[name]: _TensorBoardHyperparameter = json.dumps(
                    value,
                    allow_nan=False,
                    ensure_ascii=False,
                    sort_keys=True
                )
        self._get_writer().add_hparams(tensorboard_params, {})

    @override
    def finalize(self) -> None:
        writer: _TensorBoardWriter | None = self._writer
        self._writer: _TensorBoardWriter | None = None
        if writer is None:
            return
        try:
            writer.flush()
        finally:
            writer.close()

    def _get_writer(self) -> _TensorBoardWriter:
        if self._writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self._writer: _TensorBoardWriter | None = SummaryWriter(
                log_dir=self.log_dir
            )
        return self._writer
