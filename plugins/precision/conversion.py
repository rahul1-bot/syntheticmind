from types import TracebackType
from typing import Literal

import torch

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
        if original_dtype is None:
            raise RuntimeError("The dtype context was not entered.")
        torch.set_default_dtype(original_dtype)
        self._original_dtype: torch.dtype | None = None
        return False
