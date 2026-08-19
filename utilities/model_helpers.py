from functools import partial

from torch import nn

__all__: list[str] = ["ModuleMode", "is_overridden"]


class ModuleMode:
    def __init__(self) -> None:
        self._training_modes: dict[str, bool] = {}

    def capture(self, module: nn.Module) -> None:
        self._training_modes: dict[str, bool] = {
            name: submodule.training for name, submodule in module.named_modules()
        }

    def restore(self, module: nn.Module) -> None:
        for name, submodule in module.named_modules():
            if name in self._training_modes:
                submodule.training: bool = self._training_modes[name]


def is_overridden(method_name: str, instance: object, parent: type[object]) -> bool:
    instance_attribute: object | None = getattr(instance, method_name, None)
    if instance_attribute is None:
        return False
    parent_attribute: object | None = getattr(parent, method_name, None)
    if parent_attribute is None:
        raise ValueError(f"{parent.__name__} does not define {method_name}().")

    instance_attribute: object = _unwrap_callable(instance_attribute)
    parent_attribute: object = _unwrap_callable(parent_attribute)
    instance_code: object | None = getattr(instance_attribute, "__code__", None)
    parent_code: object | None = getattr(parent_attribute, "__code__", None)
    if instance_code is None or parent_code is None:
        return instance_attribute is not parent_attribute
    return instance_code is not parent_code


def _unwrap_callable(callable_object: object) -> object:
    unwrapped_callable: object = callable_object
    while True:
        wrapped_callable: object | None = getattr(unwrapped_callable, "__wrapped__", None)
        if wrapped_callable is not None:
            unwrapped_callable: object = wrapped_callable
            continue
        mock_wrap: object | None = getattr(unwrapped_callable, "_mock_wraps", None)
        if mock_wrap is not None:
            unwrapped_callable: object = mock_wrap
            continue
        if isinstance(unwrapped_callable, partial):
            unwrapped_callable: object = unwrapped_callable.func
            continue
        return unwrapped_callable
