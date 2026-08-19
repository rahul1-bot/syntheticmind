from collections.abc import Callable
from types import TracebackType
from typing import Literal, cast

import torch

from syntheticmind.utilities.types import Batch

__all__: list[str] = []


class _DtypeContext:
    def __init__(self, dtype: torch.dtype) -> None:
        self._dtype: torch.dtype = dtype
        self._original_dtype: torch.dtype | None = None

    def __enter__(self) -> None:
        self._original_dtype: torch.dtype | None = torch.get_default_dtype()
        torch.set_default_dtype(self._dtype)

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None
    ) -> Literal[False]:
        del exception_type, exception, traceback
        original_dtype: torch.dtype | None = self._original_dtype
        assert original_dtype is not None
        torch.set_default_dtype(original_dtype)
        self._original_dtype: torch.dtype | None = None
        return False


class _FloatingPointConverter:
    def convert(self, data: Batch, dtype: torch.dtype) -> Batch:
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype) if data.is_floating_point() else data
        if isinstance(data, dict):
            converted_mapping: dict[str, Batch] = {}
            key: str
            value: Batch
            for key, value in data.items():
                converted_mapping[key] = self.convert(value, dtype)
            return converted_mapping
        if isinstance(data, tuple):
            converted_items: list[Batch] = []
            item: Batch
            for item in data:
                converted_items.append(self.convert(item, dtype))
            if hasattr(data, "_fields"):
                namedtuple_factory: Callable[[list[Batch]], Batch] = cast(
                    Callable[[list[Batch]], Batch],
                    getattr(type(data), "_make")
                )
                return namedtuple_factory(converted_items)
            return tuple(converted_items)
        if isinstance(data, list):
            converted_items: list[Batch] = []
            item: Batch
            for item in data:
                converted_items.append(self.convert(item, dtype))
            return converted_items
        return data
