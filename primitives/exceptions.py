# This module:
# 1. Defines the exception vocabulary shared by every SyntheticMind subsystem
# 2. Separates configuration failures, checkpoint failures, and process
#    termination from ordinary Python and PyTorch exceptions
#
# Design decisions:
# - FrameworkError is the common catch boundary for recoverable framework
#   failures without swallowing unrelated user-code or backend exceptions
# - MisconfigurationError and CheckpointError prefix their messages at the
#   exception boundary, keeping raise sites focused on the corrective detail
# - SIGTERMException inherits SystemExit rather than FrameworkError so normal
#   exception handlers cannot accidentally convert process termination into a
#   recoverable training failure; its exit status follows the shell convention
#   of 128 plus the received signal number
#
# Author: Rahul Sawhney

import signal
from typing import override

__all__: list[str] = [
    "FrameworkError",
    "MisconfigurationError",
    "CheckpointError",
    "SIGTERMException"
]


class FrameworkError(Exception):
    # Base class for SyntheticMind failures that callers may catch without
    # intercepting arbitrary model, dataloader, or backend exceptions.
    pass


class MisconfigurationError(FrameworkError):
    # Raised when a requested framework configuration is internally invalid or
    # incompatible with the selected runtime components.
    @override
    def __init__(self, message: str) -> None:
        # Applies one stable category prefix while preserving the corrective
        # detail supplied by the owning validation boundary.
        super().__init__(f"Misconfiguration: {message}")


class CheckpointError(FrameworkError):
    # Raised when a checkpoint cannot satisfy its declared storage, schema, or
    # atomic-publication contract.
    @override
    def __init__(self, message: str) -> None:
        # Applies one stable persistence-category prefix to the causal detail.
        super().__init__(f"Checkpoint error: {message}")


class SIGTERMException(SystemExit):
    # Process-termination signal represented as a standard shell exit status.
    # Callbacks may observe it through on_exception, but ordinary framework
    # error recovery must allow the SystemExit semantics to propagate.
    @override
    def __init__(self) -> None:
        # Encodes SIGTERM using the standard shell status of 128 plus signal.
        super().__init__(128 + signal.SIGTERM)
