from collections.abc import Callable, Iterable
from functools import partial
from types import CodeType
from typing import cast

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.datamodule import DataModule
from syntheticmind.core.hooks import DataHooks, ModelHooks
from syntheticmind.core.module import Module
from syntheticmind.profilers.profiler import Profiler
from syntheticmind.strategies.strategy import Strategy
from syntheticmind.trainer.connectors.callback_connector import _CallbackConnector
from syntheticmind.types import DataLoaderSource, ModelOutput

__all__: list[str] = []

type _HookOwner = Module | DataModule | DataHooks | ModelHooks
type _HookCallable = Callable[..., ModelOutput | DataLoaderSource | None]


class _HookDispatcher[TrainerT]:
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
