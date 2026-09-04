import inspect
from collections.abc import Callable, Iterator
from itertools import chain
from types import FrameType
from typing import cast, override

import torch
from torch.overrides import TorchFunctionMode

__all__: list[str] = []

type _TorchFunctionValue = (
    bool
    | int
    | float
    | complex
    | str
    | bytes
    | torch.Tensor
    | torch.device
    | torch.dtype
    | torch.layout
    | torch.memory_format
    | torch.Generator
    | torch.Size
    | list[_TorchFunctionValue]
    | tuple[_TorchFunctionValue, ...]
    | dict[str, _TorchFunctionValue]
    | None
)
type _TorchFunction = Callable[..., _TorchFunctionValue]
type _ResetParameters = Callable[[], None]


class _ModuleInitializer(TorchFunctionMode):
    @override
    def __init__(self, enabled: bool = True) -> None:
        super().__init__()
        if type(enabled) is not bool:
            raise TypeError(f"Module initialization mode must be a bool, got {enabled!r}.")
        self.enabled: bool = enabled

    def materialize_distributed(
        self,
        module: torch.nn.Module,
        device: torch.device
    ) -> set[str]:
        tensors: Iterator[torch.Tensor] = chain(
            module.parameters(),
            module.buffers()
        )
        if not any(tensor.is_meta for tensor in tensors):
            return set()
        module.to_empty(device=device)
        uninitialized_modules: set[str] = set()
        submodule: torch.nn.Module
        for submodule in module.modules():
            local_tensors: Iterator[torch.Tensor] = chain(
                submodule.parameters(recurse=False),
                submodule.buffers(recurse=False)
            )
            if not any(True for _ in local_tensors):
                continue
            reset_parameters: _ResetParameters | None = cast(
                _ResetParameters | None,
                getattr(submodule, "reset_parameters", None)
            )
            if reset_parameters is None:
                uninitialized_modules.add(type(submodule).__qualname__)
                continue
            reset_parameters()
        return uninitialized_modules

    def _called_from_torch_nn_init(self) -> bool:
        frame: FrameType | None = inspect.currentframe()
        try:
            while frame is not None:
                module_name: str | None = cast(
                    str | None,
                    frame.f_globals.get("__name__")
                )
                if module_name == "torch.nn.init":
                    return True
                frame: FrameType | None = frame.f_back
        finally:
            del frame
        return False

    @override
    def __torch_function__(
        self,
        func: _TorchFunction,
        types: tuple[type[torch.Tensor], ...],
        args: tuple[_TorchFunctionValue, ...] = (),
        kwargs: dict[str, _TorchFunctionValue] | None = None
    ) -> _TorchFunctionValue:
        del types
        call_kwargs: dict[str, _TorchFunctionValue] = (
            {} if kwargs is None else kwargs
        )
        function_name: str | None = cast(
            str | None,
            getattr(func, "__name__", None)
        )
        function_module: str | None = cast(
            str | None,
            getattr(func, "__module__", None)
        )
        if (
            self.enabled
            and function_module == "torch.nn.init"
            and function_name is not None
            and function_name.endswith("_")
        ):
            if "tensor" in call_kwargs:
                return call_kwargs["tensor"]
            if not args:
                raise RuntimeError(
                    f"Initialization function {function_name!r} received no tensor."
                )
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
            and function_name is not None
            and function_name.endswith("_")
            and not function_name.endswith("__")
            and args
        ):
            return args[0]
        return func(*args, **call_kwargs)
