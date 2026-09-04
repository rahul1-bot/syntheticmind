# This module:
# 1. Defines the single invocation boundary for callback, Module, DataModule, and
#    Strategy hooks
# 2. Owns profiler action names, nested Module hook context, callback grouping,
#    and robust override discovery through decorators, mocks, and partials
#
# Design decisions:
# - Trainer and loops own when hooks run; this class owns only how each call is
#   contextualized and measured
# - Module and callback calls install the active hook name so Module.log validates
#   lifecycle semantics even under nested framework operations
# - Profiler contexts are omitted entirely while inactive, avoiding measurement
#   overhead and false empty actions
# - Callback state_key enters action identity so several configured instances
#   remain distinguishable in profiler output
# - Override discovery compares unwrapped code identity rather than catching
#   TypeError during invocation, preserving exceptions raised inside client code
# - A generic Trainer type avoids circular import and a fabricated host Protocol
#
# Author: Rahul Sawhney

from collections.abc import Callable, Iterable
from functools import partial
from types import CodeType
from typing import cast

from syntheticmind.callbacks.callback import Callback
from syntheticmind.data.datamodule import DataModule
from syntheticmind.core.hooks import DataHooks, ModelHooks
from syntheticmind.core.module import Module
from syntheticmind.profilers.profiler import Profiler
from syntheticmind.strategies.strategy import Strategy
from syntheticmind.trainer.connectors.callback_connector import _CallbackConnector
from syntheticmind.primitives.types import DataLoaderSource, ModelOutput

__all__: list[str] = []

type _HookOwner = Module | DataModule | DataHooks | ModelHooks
type _HookCallable = Callable[..., ModelOutput | DataLoaderSource | None]


class _HookDispatcher[TrainerT]:
    # Private invocation collaborator bound to one Trainer, callback connector,
    # profiler, and Strategy.
    def __init__(
        self,
        trainer: TrainerT,
        callback_connector: _CallbackConnector,
        profiler: Profiler,
        strategy: Strategy
    ) -> None:
        self._trainer: TrainerT = trainer
        self._callback_connector: _CallbackConnector = callback_connector
        self._profiler: Profiler = profiler
        self._strategy: Strategy = strategy

    def call_callback[**Parameters, ReturnT](
        self,
        callback: Callback,
        module: Module,
        hook_name: str,
        hook: Callable[..., ReturnT],
        *args: Parameters.args,
        **kwargs: Parameters.kwargs
    ) -> ReturnT:
        # Invokes one callback with Trainer and Module, active hook context, and an
        # optional profiler action qualified by callback state identity.
        action_name: str = f"[Callback]{callback.state_key}.{hook_name}"
        if not self._profiler.is_active:
            with module._hook_context(hook_name):
                return hook(self._trainer, module, *args, **kwargs)
        with module._hook_context(hook_name), self._profiler.profile(action_name):
            return hook(self._trainer, module, *args, **kwargs)

    def call_callbacks[**Parameters](
        self,
        module: Module,
        hook_name: str,
        *args: Parameters.args,
        callback_group: Iterable[Callback] | None = None,
        **kwargs: Parameters.kwargs
    ) -> None:
        # Invokes one hook on the supplied callback group or current configured
        # callback order.
        callbacks: Iterable[Callback] = (
            self._callback_connector.callbacks
            if callback_group is None
            else callback_group
        )
        callback: Callback
        for callback in callbacks:
            hook: Callable[..., None] = cast(
                Callable[..., None],
                getattr(callback, hook_name)
            )
            self.call_callback(
                callback,
                module,
                hook_name,
                hook,
                *args,
                **kwargs
            )

    def call_module[**Parameters, ReturnT](
        self,
        module: Module,
        hook_name: str,
        hook: Callable[Parameters, ReturnT],
        *args: Parameters.args,
        **kwargs: Parameters.kwargs
    ) -> ReturnT:
        # Invokes one Module hook with nested-safe logging context and optional
        # profiler ownership.
        action_name: str = f"[Module]{type(module).__name__}.{hook_name}"
        if not self._profiler.is_active:
            with module._hook_context(hook_name):
                return hook(*args, **kwargs)
        with self._profiler.profile(action_name), module._hook_context(hook_name):
            return hook(*args, **kwargs)

    def call_datamodule[**Parameters, ReturnT](
        self,
        datamodule: DataModule,
        hook_name: str,
        hook: Callable[Parameters, ReturnT],
        *args: Parameters.args,
        **kwargs: Parameters.kwargs
    ) -> ReturnT:
        # Invokes one DataModule hook under profiler ownership without installing
        # Module logging context.
        action_name: str = f"[DataModule]{type(datamodule).__name__}.{hook_name}"
        if not self._profiler.is_active:
            return hook(*args, **kwargs)
        with self._profiler.profile(action_name):
            return hook(*args, **kwargs)

    def call_strategy[**Parameters, ReturnT](
        self,
        hook_name: str,
        hook: Callable[Parameters, ReturnT],
        *args: Parameters.args,
        **kwargs: Parameters.kwargs
    ) -> ReturnT:
        # Invokes one Strategy operation under its concrete policy identity.
        action_name: str = f"[Strategy]{type(self._strategy).__name__}.{hook_name}"
        if not self._profiler.is_active:
            return hook(*args, **kwargs)
        with self._profiler.profile(action_name):
            return hook(*args, **kwargs)

    def is_overridden(
        self,
        method_name: str,
        instance: _HookOwner,
        parent: type[_HookOwner]
    ) -> bool:
        # Resolves instance and declared parent callables, unwraps decoration, and
        # compares code identity or callable identity when code is unavailable.
        instance_attribute: _HookCallable | None = cast(
            _HookCallable | None,
            getattr(instance, method_name, None)
        )
        if instance_attribute is None:
            return False
        parent_attribute: _HookCallable | None = cast(
            _HookCallable | None,
            getattr(parent, method_name, None)
        )
        if parent_attribute is None:
            raise ValueError(f"{parent.__name__} does not define {method_name}().")
        instance_callable: _HookCallable = self._unwrap(instance_attribute)
        parent_callable: _HookCallable = self._unwrap(parent_attribute)
        instance_code: CodeType | None = cast(
            CodeType | None,
            getattr(instance_callable, "__code__", None)
        )
        parent_code: CodeType | None = cast(
            CodeType | None,
            getattr(parent_callable, "__code__", None)
        )
        if instance_code is None or parent_code is None:
            return instance_callable is not parent_callable
        return instance_code is not parent_code

    def _unwrap(self, hook: _HookCallable) -> _HookCallable:
        # Peels __wrapped__, unittest mock delegation, and functools.partial layers
        # until override comparison reaches the implementation callable.
        unwrapped_hook: _HookCallable = hook
        while True:
            wrapped_hook: _HookCallable | None = cast(
                _HookCallable | None,
                getattr(unwrapped_hook, "__wrapped__", None)
            )
            if wrapped_hook is not None:
                unwrapped_hook: _HookCallable = wrapped_hook
                continue
            mock_wrap: _HookCallable | None = cast(
                _HookCallable | None,
                getattr(unwrapped_hook, "_mock_wraps", None)
            )
            if mock_wrap is not None:
                unwrapped_hook: _HookCallable = mock_wrap
                continue
            if isinstance(unwrapped_hook, partial):
                unwrapped_hook: _HookCallable = cast(
                    _HookCallable,
                    unwrapped_hook.func
                )
                continue
            return unwrapped_hook
