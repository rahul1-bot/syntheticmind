from collections.abc import Iterable, Iterator
from typing import cast, override

from syntheticmind.utilities.types import Batch

__all__: list[str] = []

_MISSING_BATCH: object = object()


class _DataFetcher(Iterator[Batch]):
    def __init__(
        self,
        iterable: Iterable[Batch],
        max_batches: int | None,
        sized: bool
    ) -> None:
        self._iterable: Iterable[Batch] = iterable
        self._max_batches: int | None = max_batches
        self._prefetch: bool = not sized
        self._iterator: Iterator[Batch] | None = None
        self._next_batch: Batch | object = _MISSING_BATCH
        self.fetched: int = 0
        self.done: bool = max_batches == 0

    @override
    def __iter__(self) -> _DataFetcher:
        self._iterator = iter(self._iterable)
        self._next_batch = _MISSING_BATCH
        self.fetched = 0
        self.done = self._max_batches == 0
        if self._prefetch and not self.done:
            self._prefetch_next()
        return self

    @override
    def __next__(self) -> Batch:
        if self.done:
            raise StopIteration
        if self._iterator is None:
            raise RuntimeError("The data fetcher must be iterated before requesting a batch.")
        if not self._prefetch:
            try:
                batch: Batch = next(self._iterator)
            except StopIteration:
                self.done = True
                raise
            self.fetched += 1
            self.done = (
                self._max_batches is not None
                and self.fetched >= self._max_batches
            )
            return batch
        if self._next_batch is _MISSING_BATCH:
            self.done = True
            raise StopIteration
        batch = cast(Batch, self._next_batch)
        self._next_batch = _MISSING_BATCH
        self.fetched += 1
        if (
            self._max_batches is not None
            and self.fetched >= self._max_batches
        ):
            self.done = True
            return batch
        self._prefetch_next()
        return batch

    def _prefetch_next(self) -> None:
        if self._iterator is None:
            raise RuntimeError("The data fetcher must be iterated before prefetching.")
        try:
            self._next_batch = next(self._iterator)
        except StopIteration:
            self._next_batch = _MISSING_BATCH
            self.done = True
