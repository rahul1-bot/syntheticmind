import inspect
from collections.abc import Callable
from types import FrameType
from typing import override

import torch
from torch.overrides import TorchFunctionMode

__all__: list[str] = []


class _EmptyInit(TorchFunctionMode):
    def __init__(self, enabled: bool = True) -> None:
        super().__init__()
        self.enabled: bool = enabled

    @override
    def __torch_function__(
        self,
        func: Callable[..., object],
        types: tuple[type[object], ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None
    ) -> object:
        del types
        call_kwargs: dict[str, object] = {} if kwargs is None else kwargs
        function_name: object = getattr(func, "__name__", None)
        if (
            self.enabled
            and getattr(func, "__module__", None) == "torch.nn.init"
            and isinstance(function_name, str)
            and function_name.endswith("_")
        ):
            if "tensor" in call_kwargs:
                return call_kwargs["tensor"]
            return args[0]
        called_from_init: bool = self.enabled and self._called_from_torch_nn_init()
        if (
            called_from_init
            and function_name == "randperm"
            and args
            and isinstance(args[0], int)
        ):
            return torch.arange(args[0])
        if called_from_init and "out" in call_kwargs:
            return call_kwargs["out"]
        if called_from_init and function_name == "__setitem__":
            return None
        if (
            called_from_init
            and isinstance(function_name, str)
            and function_name.endswith("_")
            and not function_name.endswith("__")
            and args
        ):
            return args[0]
        return func(*args, **call_kwargs)

    def _called_from_torch_nn_init(self) -> bool:
        frame: FrameType | None = inspect.currentframe()
        try:
            while frame is not None:
                if frame.f_globals.get("__name__") == "torch.nn.init":
                    return True
                frame: FrameType | None = frame.f_back
        finally:
            del frame
        return False
