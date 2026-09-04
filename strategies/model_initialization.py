import inspect
from collections.abc import Callable, Iterator
from itertools import chain
from types import FrameType
from typing import override

import torch
from torch.overrides import TorchFunctionMode

__all__: list[str] = []


class _ModuleInitializer(TorchFunctionMode):
    def __init__(self, enabled: bool = True) -> None:
        super().__init__()
        self.enabled: bool = enabled

    def materialize_distributed(
        self,
        module: torch.nn.Module,
        device: torch.device
    ) -> set[str]:
        tensors: Iterator[torch.Tensor] = chain(module.parameters(), module.buffers())
        if not any(tensor.is_meta for tensor in tensors):
            return set()
        module.to_empty(device=device)
        uninitialized_modules: set[str] = set()
        for submodule in module.modules():
            local_tensors: Iterator[torch.Tensor] = chain(
                submodule.parameters(recurse=False),
                submodule.buffers(recurse=False)
            )
            if not any(True for _ in local_tensors):
                continue
            reset_parameters: object = getattr(submodule, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()
            else:
                uninitialized_modules.add(type(submodule).__name__)
        return uninitialized_modules

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
