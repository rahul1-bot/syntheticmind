# This module:
# 1. Defines the lookahead-aware data fetcher used by sized and unsized loader
#    iteration
# 2. Captures stateful-loader position after the returned batch but before the
#    next unseen prefetch for exact checkpoint continuation
#
# Design decisions:
# - Sized loaders derive final-batch status from their bound and need no lookahead;
#   unsized loaders prefetch one value to know whether the current value is final
# - For unsized stateful loaders, the snapshot is taken after exposing the current
#   batch and before requesting another, so a checkpoint never advances its
#   cursor beyond data the loop actually consumed
# - Explicit max_batches stops without probing another value, preserving stream
#   state and preventing blocking iterables from reading past the configured run
# - Iterator state resets on each __iter__ call and no underlying loader behavior
#   is wrapped or replaced
#
# Author: Rahul Sawhney

from collections.abc import Callable, Iterable, Iterator
from typing import cast, override

from syntheticmind.primitives.types import DataLoaderState

__all__: list[str] = []


class _DataFetcher[T](Iterator[T]):
    # Internal bounded iterator with optional unsized lookahead and cursor
    # snapshot ownership.
    def __init__(
        self,
        iterable: Iterable[T],
        max_batches: int | None,
        sized: bool,
        state_snapshot: Callable[[], list[DataLoaderState]] | None = None
    ) -> None:
        # Selects lookahead only for unsized data and starts at an explicit zero-
        # batch completion state when max_batches is zero.
        self._iterable: Iterable[T] = iterable
        self._max_batches: int | None = max_batches
        self._prefetch: bool = not sized
        self._iterator: Iterator[T] | None = None
        self._next_batch: T | None = None
        self._has_next_batch: bool = False
        self._state_snapshot: Callable[[], list[DataLoaderState]] | None = state_snapshot
        self.state: list[DataLoaderState] | None = None
        self.fetched: int = 0
        self.done: bool = max_batches == 0

    @override
    def __iter__(self) -> _DataFetcher[T]:
        # Recreates the underlying iterator, clears counters and snapshots, and
        # primes one value only when unsized lookahead is required.
        self._iterator: Iterator[T] | None = iter(self._iterable)
        self._next_batch: T | None = None
        self._has_next_batch: bool = False
        self.state: list[DataLoaderState] | None = None
        self.fetched: int = 0
        self.done: bool = self._max_batches == 0
        if self._prefetch and not self.done:
            self._prefetch_next()
        return self

    @override
    def __next__(self) -> T:
        # Returns one bounded value and updates finality. Unsized state snapshots
        # describe the cursor immediately after the returned value.
        if self.done:
            raise StopIteration
        if self._iterator is None:
            raise RuntimeError("The data fetcher must be iterated before requesting a batch.")
        if not self._prefetch:
            try:
                batch: T = next(self._iterator)
            except StopIteration:
                self.done: bool = True
                raise
            self.fetched: int = self.fetched + 1
            self.done: bool = (
                self._max_batches is not None
                and self.fetched >= self._max_batches
            )
            return batch
        if not self._has_next_batch:
            self.done: bool = True
            raise StopIteration
        batch: T = cast(T, self._next_batch)
        self._next_batch: T | None = None
        self._has_next_batch: bool = False
        self.fetched: int = self.fetched + 1
        if self._state_snapshot is not None:
            self.state: list[DataLoaderState] | None = self._state_snapshot()
        if (
            self._max_batches is not None
            and self.fetched >= self._max_batches
        ):
            self.done: bool = True
            return batch
        self._prefetch_next()
        return batch

    def _prefetch_next(self) -> None:
        # Reads exactly one lookahead value or marks natural exhaustion without
        # exposing the unseen value to checkpoint state.
        if self._iterator is None:
            raise RuntimeError("The data fetcher must be iterated before prefetching.")
        try:
            self._next_batch: T | None = next(self._iterator)
            self._has_next_batch: bool = True
        except StopIteration:
            self._next_batch: T | None = None
            self._has_next_batch: bool = False
            self.done: bool = True
