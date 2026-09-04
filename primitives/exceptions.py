import signal
from typing import override

__all__: list[str] = [
    "FrameworkError",
    "MisconfigurationError",
    "CheckpointError",
    "SIGTERMException"
]


class FrameworkError(Exception):
    pass


class MisconfigurationError(FrameworkError):
    @override
    def __init__(self, message: str) -> None:
        super().__init__(f"Misconfiguration: {message}")


class CheckpointError(FrameworkError):
    @override
    def __init__(self, message: str) -> None:
        super().__init__(f"Checkpoint error: {message}")


class SIGTERMException(SystemExit):
    @override
    def __init__(self) -> None:
        super().__init__(128 + signal.SIGTERM)
