from collections.abc import Callable, Iterable, Iterator
from typing import cast, override

from syntheticmind.utilities.types import DataLoaderState

__all__: list[str] = []


class _DataFetcher[T](Iterator[T]):
    def __init__(
        self,
        iterable: Iterable[T],
        max_batches: int | None,
        sized: bool,
        state_snapshot: Callable[[], list[DataLoaderState]] | None = None
    ) -> None:
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
        if self._iterator is None:
            raise RuntimeError("The data fetcher must be iterated before prefetching.")
        try:
            self._next_batch: T | None = next(self._iterator)
            self._has_next_batch: bool = True
        except StopIteration:
            self._next_batch: T | None = None
            self._has_next_batch: bool = False
            self.done: bool = True
