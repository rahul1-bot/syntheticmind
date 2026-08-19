from __future__ import annotations

import signal
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import FrameType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from syntheticmind.core.trainer import Trainer

__all__: list[str] = []

type _SignalHandler = (
    Callable[[int, FrameType | None], object]
    | signal.Handlers
    | int
    | None
)


class _SignalConnector:
    def __init__(self, trainer: Trainer) -> None:
        self.trainer: Trainer = trainer
        self.received_sigterm: bool = False
        self._original_handler: _SignalHandler = None
        self._installed_handler: Callable[[int, FrameType | None], object] | None = None
        self._registration_depth: int = 0

    def register_signal_handlers(self) -> None:
        if self._registration_depth > 0:
            self._registration_depth += 1
            return
        self.received_sigterm = False
        if threading.current_thread() is not threading.main_thread():
            return
        self._original_handler = signal.getsignal(signal.SIGTERM)
        self._installed_handler = self._handle_sigterm
        signal.signal(signal.SIGTERM, self._installed_handler)
        self._registration_depth = 1

    def teardown(self) -> None:
        if self._registration_depth == 0:
            return
        self._registration_depth -= 1
        if self._registration_depth > 0:
            return
        if (
            self._installed_handler is not None
            and signal.getsignal(signal.SIGTERM) == self._installed_handler
            and self._original_handler is not None
        ):
            signal.signal(signal.SIGTERM, self._original_handler)
        self._original_handler = None
        self._installed_handler = None

    @contextmanager
    def launch_context(self) -> Iterator[None]:
        self.register_signal_handlers()
        try:
            yield
        finally:
            self.teardown()

    def _handle_sigterm(
        self,
        signum: int,
        frame: FrameType | None
    ) -> None:
        first_signal: bool = not self.received_sigterm
        self.received_sigterm = True
        if first_signal:
            launcher = self.trainer.strategy.launcher
            if launcher is not None:
                try:
                    launcher.kill(signum)
                except OSError:
                    pass
        original_handler: _SignalHandler = self._original_handler
        if callable(original_handler) and original_handler != self._installed_handler:
            original_handler(signum, frame)

    def __getstate__(self) -> dict[str, object]:
        state: dict[str, object] = dict(self.__dict__)
        state["_original_handler"] = None
        state["_installed_handler"] = None
        state["_registration_depth"] = 0
        return state
