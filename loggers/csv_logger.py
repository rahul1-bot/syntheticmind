import csv
import json
import os
import tempfile
import warnings
from io import StringIO, TextIOBase
from pathlib import Path
from typing import override

from fsspec.spec import AbstractFileSystem
from syntheticmind.loggers.logger import Logger, _JsonValue
from syntheticmind.storage.filesystem import StoragePath
from syntheticmind.types import HyperparameterDict

__all__: list[str] = ["CSVLogger"]

type _CSVRow = dict[str, str | None]
class CSVLogger(Logger):
    @override
    def __init__(
        self,
        save_dir: str | Path,
        name: str = "default",
        version: int | str | None = None
    ) -> None:
        super().__init__(save_dir=save_dir, name=name, version=version)
        self._metrics_path: str = StoragePath(self.log_dir).join("metrics.csv")
        self._hyperparameters_path: str = StoragePath(self.log_dir).join(
            "hparams.json"
        )
        self._writer: csv.DictWriter | None = None
        self._file_handle: TextIOBase | None = None
        self._fieldnames: list[str] = self._read_existing_fieldnames()

    @override
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        validated_metrics: dict[str, float] = self._validated_metrics(metrics, step)
        if "step" in validated_metrics:
            raise ValueError("CSV metric name 'step' conflicts with the logging coordinate.")
        if self._writer is None:
            self._merge_existing_fieldnames()
        new_fieldnames: list[str] = sorted(
            name for name in validated_metrics if name not in self._fieldnames
        )
        if new_fieldnames:
            previous_fieldnames: list[str] = list(self._fieldnames)
            self._fieldnames.extend(new_fieldnames)
            try:
                self._rewrite_metrics()
            except BaseException:
                self._fieldnames: list[str] = previous_fieldnames
                raise
        row: dict[str, int | float] = {"step": step, **validated_metrics}
        if not StoragePath(self._metrics_path).is_local:
            self._append_remote_row(row)
            return
        if self._writer is None:
            self._open_writer()
        assert self._writer is not None
        self._writer.writerow(row)
        assert self._file_handle is not None
        self._file_handle.flush()

    @override
    def log_hyperparams(self, params: HyperparameterDict) -> None:
        normalized_params: dict[str, _JsonValue] = self._normalize_hyperparameters(
            params
        )
        serialized_params: str = json.dumps(
            normalized_params,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True
        )
        self._replace_text(self._hyperparameters_path, serialized_params)

    @override
    def finalize(self) -> None:
        file_handle: TextIOBase | None = self._file_handle
        self._file_handle: TextIOBase | None = None
        self._writer: csv.DictWriter | None = None
        if file_handle is None:
            return
        try:
            file_handle.flush()
        finally:
            file_handle.close()

    def _read_existing_fieldnames(self) -> list[str]:
        metrics_path: StoragePath = StoragePath(self._metrics_path)
        if not metrics_path.exists() or metrics_path.size == 0:
            return ["step"]
        reader: csv.DictReader = csv.DictReader(
            StringIO(metrics_path.read_text())
        )
        fieldnames: list[str] | None = reader.fieldnames
        if fieldnames is None or not fieldnames:
            raise ValueError(f"CSV metrics file has no header: {self._metrics_path}.")
        if fieldnames[0] != "step":
            raise ValueError(
                f"CSV metrics header must begin with 'step': {self._metrics_path}."
            )
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(
                f"CSV metrics header contains duplicate fields: {self._metrics_path}."
            )
        return list(fieldnames)

    def _read_existing_rows(self) -> list[_CSVRow]:
        metrics_path: StoragePath = StoragePath(self._metrics_path)
        if not metrics_path.exists() or metrics_path.size == 0:
            return []
        reader: csv.DictReader = csv.DictReader(
            StringIO(metrics_path.read_text())
        )
        rows: list[_CSVRow] = []
        row: _CSVRow
        for row in reader:
            if None in row:
                raise ValueError(
                    f"CSV metrics row exceeds its header: {self._metrics_path}."
                )
            rows.append(dict(row))
        return rows

    def _merge_existing_fieldnames(self) -> None:
        existing_fieldnames: list[str] = self._read_existing_fieldnames()
        fieldname: str
        for fieldname in existing_fieldnames:
            if fieldname not in self._fieldnames:
                self._fieldnames.append(fieldname)

    def _rewrite_metrics(self) -> None:
        self.finalize()
        existing_rows: list[_CSVRow] = self._read_existing_rows()
        stream: StringIO = StringIO()
        writer: csv.DictWriter = csv.DictWriter(
            stream,
            fieldnames=self._fieldnames,
            extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(existing_rows)
        self._replace_text(self._metrics_path, stream.getvalue())

    def _open_writer(self) -> None:
        metrics_path: StoragePath = StoragePath(self._metrics_path)
        file_exists: bool = metrics_path.exists()
        file_is_empty: bool = not file_exists or metrics_path.size == 0
        mode: str = "a" if file_exists else "w"
        file_handle: TextIOBase = metrics_path.open_text(
            mode,
            newline=""
        )
        writer: csv.DictWriter = csv.DictWriter(
            file_handle,
            fieldnames=self._fieldnames,
            extrasaction="raise"
        )
        self._file_handle: TextIOBase | None = file_handle
        self._writer: csv.DictWriter | None = writer
        if file_is_empty:
            writer.writeheader()

    def _append_remote_row(self, row: dict[str, int | float]) -> None:
        stream: StringIO = StringIO()
        writer: csv.DictWriter = csv.DictWriter(
            stream,
            fieldnames=self._fieldnames,
            extrasaction="raise"
        )
        metrics_path: StoragePath = StoragePath(self._metrics_path)
        if not metrics_path.exists() or metrics_path.size == 0:
            writer.writeheader()
        writer.writerow(row)
        existing_content: str = (
            metrics_path.read_text()
            if metrics_path.exists()
            else ""
        )
        self._replace_text(
            self._metrics_path,
            f"{existing_content}{stream.getvalue()}"
        )

    def _replace_text(self, path: str, content: str) -> None:
        storage_path: StoragePath = StoragePath(path)
        if not storage_path.is_local:
            StoragePath(storage_path.parent).make_directory()
            filesystem: AbstractFileSystem = storage_path.filesystem
            with filesystem.transaction:
                with storage_path.open_text("w", newline="") as opened_file:
                    opened_file.write(content)
            return
        local_path: Path = Path(storage_path.normalized)
        StoragePath(local_path.parent).make_directory()
        temporary_file: tuple[int, str] = tempfile.mkstemp(
            dir=str(local_path.parent),
            prefix=f".{local_path.name}-"
        )
        file_descriptor: int = temporary_file[0]
        temporary_path: Path = Path(temporary_file[1])
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            temporary_path.replace(local_path)
            try:
                directory_descriptor: int = os.open(local_path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            except OSError as exception:
                warnings.warn(
                    f"Logger file replaced but directory synchronization failed: {exception}",
                    RuntimeWarning,
                    stacklevel=3
                )
        except BaseException:
            if temporary_path.exists():
                temporary_path.unlink()
            raise
