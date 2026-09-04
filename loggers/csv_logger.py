# This module:
# 1. Persists metric rows to a metrics.csv file whose column set widens as new
#    metric keys appear during the run
# 2. Resumes cleanly into an existing metrics.csv by adopting the union of its
#    header and the live key set
# 3. Persists hyperparameters as a JSON sidecar next to the metrics file
#
# Design decisions:
# - The file is opened in append mode and flushed after every row, so a crash
#   loses at most the row being written and resumed runs keep appending to the
#   same file
# - When a batch introduces unseen metric keys, the whole file is rewritten once
#   with the widened header and its existing rows preserved; rewriting on schema
#   growth keeps every row parseable under one final header
# - DictWriter runs with extrasaction="ignore" so a transiently missing key
#   produces an empty cell instead of an exception mid-run
# - Hyperparameters are serialized with default=str because run configurations
#   can contain paths and other non-JSON scalars that should degrade to text
#   rather than fail the dump
#
# Author: Rahul Sawhney

import csv
from io import StringIO, TextIOWrapper
from pathlib import Path
from typing import override

from syntheticmind.loggers.logger import Logger
from syntheticmind.storage.filesystem import (
    exists,
    file_size,
    is_local_path,
    join_path,
    open_text,
    read_text,
    write_text
)
from syntheticmind.types import HyperparameterDict

__all__: list[str] = ["CSVLogger"]


class CSVLogger(Logger):
    # Concrete logger that writes one CSV row per log_metrics call, with a step
    # column first and one column per metric key seen so far. The writer is
    # created lazily and recreated whenever the column set widens.
    def __init__(
        self,
        save_dir: str | Path,
        name: str = "default",
        version: str | None = None
    ) -> None:
        # Resolves the metrics file inside log_dir and seeds the column set from
        # any existing file so resumed runs extend rather than clobber history.
        super().__init__(save_dir=save_dir, name=name, version=version)
        self._metrics_file: str = join_path(self._log_directory(), "metrics.csv")
        self._writer: csv.DictWriter | None = None
        self._file_handle: TextIOWrapper | None = None
        self._fieldnames: list[str] = self._read_existing_fieldnames()

    @override
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        # Appends one row. Unseen keys first widen the schema (rewriting the
        # file with the new header); the writer is then created lazily and the
        # handle flushed so the row is durable immediately.
        new_keys: list[str] = [k for k in metrics if k not in self._fieldnames]
        if new_keys:
            self._fieldnames.extend(new_keys)
            self._reset_writer()

        if not is_local_path(self._metrics_file):
            self._write_remote_metrics(metrics, step)
            return
        if self._writer is None:
            self._init_writer()

        row: dict[str, int | float] = {"step": step, **metrics}
        assert self._writer is not None
        self._writer.writerow(row)
        assert self._file_handle is not None
        self._file_handle.flush()

    @override
    def log_hyperparams(self, params: HyperparameterDict) -> None:
        # Writes the hyperparameter mapping as pretty-printed JSON beside the
        # metrics file; non-JSON scalars degrade to their string form.
        import json
        hparams_file: str = join_path(self.log_dir, "hparams.json")
        serializable_params: dict[str, object] = {
            str(key): value for key, value in params.items()
        }
        write_text(
            hparams_file,
            json.dumps(serializable_params, indent=2, default=str)
        )

    @override
    def finalize(self) -> None:
        # Closes the file handle and drops the writer so a later log_metrics
        # call would reopen cleanly; safe to call repeatedly.
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle: TextIOWrapper | None = None
            self._writer: csv.DictWriter | None = None

    def _read_existing_fieldnames(self) -> list[str]:
        # Seeds the column set with the mandatory step column, then adopts any
        # columns already present in an existing non-empty metrics file so
        # resumption preserves prior schema.
        fieldnames: list[str] = ["step"]
        if not exists(self._metrics_file) or file_size(self._metrics_file) == 0:
            return fieldnames
        with open_text(self._metrics_file, "r", newline="") as f:
            reader: csv.DictReader = csv.DictReader(f)
            if reader.fieldnames is not None:
                for col in reader.fieldnames:
                    if col not in fieldnames:
                        fieldnames.append(col)
        return fieldnames

    def _init_writer(self) -> None:
        # Opens the metrics file for appending and binds a DictWriter over the
        # current column set, emitting the header only when the file is empty.
        file_exists: bool = exists(self._metrics_file)
        file_is_empty: bool = not file_exists or file_size(self._metrics_file) == 0
        mode: str = "a" if file_exists else "w"
        self._file_handle: TextIOWrapper | None = open_text(  # noqa: SIM115
            self._metrics_file,
            mode,
            newline=""
        )
        self._writer: csv.DictWriter | None = csv.DictWriter(
            self._file_handle,  # type: ignore[arg-type]
            fieldnames=self._fieldnames,
            extrasaction="ignore"
        )
        if file_is_empty:
            self._writer.writeheader()

    def _reset_writer(self) -> None:
        # Rebuilds the file under the widened schema: closes the current writer,
        # merges any columns present on disk, rewrites the file with the full
        # header and existing rows, then reopens for appending.
        self.finalize()
        if exists(self._metrics_file):
            existing_data: list[dict[str, str]] = []
            with open_text(self._metrics_file, "r", newline="") as f:
                reader: csv.DictReader = csv.DictReader(f)
                if reader.fieldnames is not None:
                    for col in reader.fieldnames:
                        if col not in self._fieldnames:
                            self._fieldnames.append(col)
                existing_data: list[dict[str, str]] = list(reader)
            with open_text(self._metrics_file, "w", newline="") as f:
                writer: csv.DictWriter = csv.DictWriter(
                    f, fieldnames=self._fieldnames, extrasaction="ignore"
                )
                writer.writeheader()
                writer.writerows(existing_data)
        if is_local_path(self._metrics_file):
            self._init_writer()

    def _write_remote_metrics(
        self,
        metrics: dict[str, float],
        step: int
    ) -> None:
        stream: StringIO = StringIO()
        writer: csv.DictWriter = csv.DictWriter(
            stream,
            fieldnames=self._fieldnames,
            extrasaction="ignore"
        )
        if not exists(self._metrics_file) or file_size(self._metrics_file) == 0:
            writer.writeheader()
        row: dict[str, int | float] = {"step": step, **metrics}
        writer.writerow(row)
        existing_data: str = (
            read_text(self._metrics_file)
            if exists(self._metrics_file)
            else ""
        )
        write_text(self._metrics_file, f"{existing_data}{stream.getvalue()}")
