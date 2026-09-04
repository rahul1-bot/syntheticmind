# This module:
# 1. Defines the reversible default-dtype context used by true precision policies
#    during tensor, module, and forward construction
#
# Design decisions:
# - PyTorch default dtype is process-global state, so the prior value is captured
#   on entry and restored on every exit including exceptions
# - The context is a small stateful owner rather than a generator function because
#   an invalid exit-before-entry transition must be detected explicitly
#
# Author: Rahul Sawhney

from types import TracebackType
from typing import Literal

import torch

__all__: list[str] = []


class _DtypeContext:
    # Reversible process-global torch default-dtype transition.
    def __init__(self, dtype: torch.dtype) -> None:
        self._dtype: torch.dtype = dtype
        self._original_dtype: torch.dtype | None = None

    def __enter__(self) -> None:
        # Captures the current default immediately before applying the requested
        # construction dtype.
        self._original_dtype: torch.dtype | None = torch.get_default_dtype()
        torch.set_default_dtype(self._dtype)

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None
    ) -> Literal[False]:
        # Restores the captured dtype, clears context state for safe reuse, and
        # returns False so any enclosed exception propagates unchanged.
        del exception_type, exception, traceback
        original_dtype: torch.dtype | None = self._original_dtype
        if original_dtype is None:
            raise RuntimeError("The dtype context was not entered.")
        torch.set_default_dtype(original_dtype)
        self._original_dtype: torch.dtype | None = None
        return False
