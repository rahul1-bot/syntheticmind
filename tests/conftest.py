# This module:
# 1. Makes the repository importable as the syntheticmind package for a source
#    checkout that has not been installed, and pins CPU-only, non-interactive,
#    color-free execution for every test process
#
# Design decisions:
# - The repository directory is the package itself, so its parent enters
#   sys.path only when the package is not already importable; an installed or
#   editable distribution takes precedence
# - NO_COLOR and a fixed terminal width keep console assertions deterministic
#   across developer terminals and continuous-integration runners
# - One session fixture restores loguru's process default handler afterward,
#   because a renderer release rebinds it to whichever stream was current
#
# Author: Rahul Sawhney

import importlib.util
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from loguru import logger as log

repository_root: Path = Path(__file__).resolve().parents[1]
if importlib.util.find_spec("syntheticmind") is None:
    if repository_root.name != "syntheticmind":
        raise RuntimeError(
            "The checkout directory must be named 'syntheticmind' or the package "
            "must be installed before running the test suite."
        )
    sys.path.insert(0, str(repository_root.parent))

os.environ["NO_COLOR"] = "1"
os.environ["COLUMNS"] = "120"


@pytest.fixture(scope="session", autouse=True)
def restore_process_diagnostics() -> Iterator[None]:
    # Leaves the interpreter with loguru's ordinary stderr handler after every
    # test has run, whatever stream a renderer last released it onto.
    yield
    log.remove()
    log.add(sys.__stderr__)
