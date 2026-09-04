import signal
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import FrameType
from typing import cast, override

from syntheticmind.strategies.launchers.launcher import _ProcessLauncher
from syntheticmind.strategies.strategy import Strategy

__all__: list[str] = []

type _SignalHandler = (
    Callable[[int, FrameType | None], None]
    | signal.Handlers
    | int
    | None
)
type _SignalStateValue = Strategy | _SignalHandler | bool | int


class _SignalConnector:
    def __init__(self, strategy: Strategy) -> None:
        self._strategy: Strategy = strategy
        self.received_sigterm: bool = False
        self._original_handler: _SignalHandler = None
        self._installed_handler: Callable[[int, FrameType | None], None] | None = None
        self._registration_depth: int = 0

    def register_signal_handlers(self) -> None:
        if self._registration_depth > 0:
            self._registration_depth: int = self._registration_depth + 1
            return
        self.received_sigterm: bool = False
        if threading.current_thread() is not threading.main_thread():
            return
        self._original_handler: _SignalHandler = signal.getsignal(signal.SIGTERM)
        self._installed_handler: Callable[[int, FrameType | None], None] | None = (
            self._handle_sigterm
        )
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        self._registration_depth: int = 1

    def teardown(self) -> None:
        if self._registration_depth == 0:
            return
        self._registration_depth: int = self._registration_depth - 1
        if self._registration_depth > 0:
            return
        if (
            self._installed_handler is not None
            and signal.getsignal(signal.SIGTERM) == self._installed_handler
            and self._original_handler is not None
        ):
            signal.signal(signal.SIGTERM, self._original_handler)
        self._original_handler: _SignalHandler = None
        self._installed_handler: Callable[[int, FrameType | None], None] | None = None

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
        self.received_sigterm: bool = True
        if first_signal:
            launcher: _ProcessLauncher | None = self._strategy.launcher
            if launcher is not None:
                try:
                    launcher.kill(signum)
                except OSError:
                    pass
        original_handler: _SignalHandler = self._original_handler
        if callable(original_handler) and original_handler != self._installed_handler:
            original_handler(signum, frame)

    @override
    def __getstate__(self) -> dict[str, _SignalStateValue]:
        state: dict[str, _SignalStateValue] = cast(
            dict[str, _SignalStateValue],
            dict(self.__dict__)
        )
        state["_original_handler"] = None
        state["_installed_handler"] = None
        state["_registration_depth"] = 0
        return state
