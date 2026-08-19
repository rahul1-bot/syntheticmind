# This module:
# 1. Writes checkpoint payloads to disk atomically through a same-directory
#    temporary file followed by a rename onto the final path
# 2. Loads checkpoint payloads with an explicit map_location and a clear
#    missing-file error
#
# Design decisions:
# - The temporary file is created inside the destination directory rather than the
#   system temp root because Path.replace is atomic only within one filesystem
# - The mkstemp descriptor is closed immediately since torch.save reopens the path
#   itself; mkstemp still guarantees the temporary name is exclusively owned
# - A failed save unlinks the temporary file and re-raises, so an interrupted write
#   can never leave a truncated payload under the final checkpoint name
# - Loading defaults to map_location="cpu" so restore works on hosts without the
#   accelerator that produced the file
# - torch.load runs with weights_only=False because checkpoint payloads carry
#   structured component state beyond bare tensors; checkpoint files are treated
#   as trusted project artifacts, not untrusted inputs
#
# Author: Rahul Sawhney

import os
import tempfile
from pathlib import Path
from typing import cast

import torch

from syntheticmind.utilities.types import CheckpointDict, MapLocation

__all__: list[str] = ["save_checkpoint", "load_checkpoint"]


def save_checkpoint(checkpoint: CheckpointDict, filepath: Path) -> None:
    # Serializes the payload to a temporary file beside the destination and
    # promotes it with an atomic rename, creating parent directories on demand.
    # Readers therefore observe either the previous complete file or the new
    # complete file, never a partial write.
    filepath.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor: int
    tmp_path_str: str
    file_descriptor, tmp_path_str = tempfile.mkstemp(
        dir=str(filepath.parent), suffix=".tmp"
    )
    os.close(file_descriptor)
    tmp_path: Path = Path(tmp_path_str)
    try:
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(filepath)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def load_checkpoint(
    filepath: Path,
    map_location: MapLocation = "cpu",
    weights_only: bool | None = None
) -> CheckpointDict:
    # Deserializes a checkpoint payload, raising FileNotFoundError with the
    # offending path instead of letting torch.load fail with a less direct
    # message. map_location defaults to CPU so restoration never requires the
    # original device.
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    checkpoint: object = torch.load(
        filepath,
        map_location=map_location,
        weights_only=False if weights_only is None else weights_only
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint payload must be a dictionary, got {type(checkpoint).__name__}."
        )
    return cast(CheckpointDict, checkpoint)
