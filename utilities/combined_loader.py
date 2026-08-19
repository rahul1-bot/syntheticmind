import math
from collections.abc import Iterable, Iterator, Sequence, Sized
from typing import Literal, Protocol, cast, override, runtime_checkable

from torch.utils import _pytree
from torch.utils.data.dataloader import (
    _BaseDataLoaderIter,
    _MultiProcessingDataLoaderIter
)

__all__: list[str] = ["CombinedLoader", "CombinedLoaderMode"]

type CombinedLoaderMode = Literal[
    "min_size",
    "max_size_cycle",
    "max_size",
    "sequential"
]
type _IteratorOutput = tuple[object, int, int]
type _Limit = int | float | None
type _LoaderState = dict[str, object]

_primitive_types: tuple[type[object], ...] = (
    bool,
    int,
    float,
    str,
    bytes,
    type(None)
)


@runtime_checkable
class _StatefulIterable(Protocol):
    def state_dict(self) -> _LoaderState: ...

    def load_state_dict(self, state_dict: _LoaderState) -> None: ...


def _is_primitive_collection(value: object) -> bool:
    children: Iterable[object]
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (list, tuple)):
        children = value
    else:
        return False
    child: object
    for child in children:
        if not isinstance(child, _primitive_types):
            return False
    return True


def _safe_len(iterable: object) -> int | None:
    if not isinstance(iterable, Sized):
        return None
    try:
        return len(iterable)
    except TypeError:
        return None


def _normalize_limit(limit: _Limit) -> _Limit:
    if limit is None:
        return None
    if type(limit) is bool or not isinstance(limit, (int, float)):
        raise TypeError("CombinedLoader limits must be non-negative integer counts or infinity.")
    if isinstance(limit, int):
        if limit < 0:
            raise ValueError("CombinedLoader limits must be non-negative integer counts or infinity.")
        return limit
    if math.isinf(limit) and limit > 0:
        return limit
    if not math.isfinite(limit) or limit < 0 or int(limit) != limit:
        raise ValueError("CombinedLoader limits must be non-negative integer counts or infinity.")
    return limit


class _ModeIterator(Iterator[_IteratorOutput]):
    def __init__(
        self,
        iterables: list[Iterable[object]],
        limits: list[int | None],
        length: int | None
    ) -> None:
        self.iterables: list[Iterable[object]] = iterables
        self.limits: list[int | None] = limits
        self.length: int | None = length
        self.iterators: list[Iterator[object]] = []
        self.counts: list[int] = []
        self.yielded: int = 0

    @override
    def __iter__(self) -> _ModeIterator:
        self.iterators: list[Iterator[object]] = []
        iterable: Iterable[object]
        for iterable in self.iterables:
            self.iterators.append(iter(iterable))
        self.counts: list[int] = [0] * len(self.iterables)
        self.yielded: int = 0
        return self

    def _next(self, index: int) -> object:
        value: object = next(self.iterators[index])
        self.counts[index] += 1
        return value

    def _is_complete(self) -> bool:
        return self.length is not None and self.yielded >= self.length

    def reset(self) -> None:
        self.iterators: list[Iterator[object]] = []
        self.counts: list[int] = []
        self.yielded: int = 0

    def __getstate__(self) -> dict[str, object]:
        state: dict[str, object] = dict(self.__dict__)
        serializable_iterators: list[Iterator[object] | None] = []
        iterator: Iterator[object]
        for iterator in self.iterators:
            serializable_iterators.append(
                None
                if isinstance(iterator, _BaseDataLoaderIter)
                else iterator
            )
        state["iterators"] = serializable_iterators
        return state


class _MinSize(_ModeIterator):
    @override
    def __next__(self) -> _IteratorOutput:
        if self._is_complete():
            raise StopIteration
        values: list[object] = []
        index: int
        for index in range(len(self.iterators)):
            values.append(self._next(index))
        batch_idx: int = self.yielded
        self.yielded += 1
        return values, batch_idx, 0


class _MaxSize(_ModeIterator):
    @override
    def __next__(self) -> _IteratorOutput:
        if self._is_complete():
            raise StopIteration
        values: list[object | None] = [None] * len(self.iterators)
        yielded_value: bool = False
        index: int
        for index in range(len(self.iterators)):
            try:
                values[index] = self._next(index)
                yielded_value = True
            except StopIteration:
                continue
        if not yielded_value:
            raise StopIteration
        batch_idx: int = self.yielded
        self.yielded += 1
        return values, batch_idx, 0


class _MaxSizeCycle(_ModeIterator):
    def __init__(
        self,
        iterables: list[Iterable[object]],
        limits: list[int | None],
        length: int | None
    ) -> None:
        super().__init__(iterables, limits, length)
        self.completed: list[bool] = []

    @override
    def __iter__(self) -> _MaxSizeCycle:
        super().__iter__()
        self.completed: list[bool] = [False] * len(self.iterables)
        return self

    @override
    def __next__(self) -> _IteratorOutput:
        if self._is_complete():
            raise StopIteration
        values: list[object] = []
        index: int
        iterable: Iterable[object]
        for index, iterable in enumerate(self.iterables):
            try:
                value: object = self._next(index)
            except StopIteration:
                self.completed[index] = True
                if all(self.completed):
                    raise
                self.iterators[index] = iter(iterable)
                self.counts[index] = 0
                value = self._next(index)
            values.append(value)
        batch_idx: int = self.yielded
        self.yielded += 1
        return values, batch_idx, 0

    @override
    def reset(self) -> None:
        super().reset()
        self.completed: list[bool] = []


class _Sequential(_ModeIterator):
    def __init__(
        self,
        iterables: list[Iterable[object]],
        limits: list[int | None],
        length: int | None
    ) -> None:
        super().__init__(iterables, limits, length)
        self.iterator_idx: int = 0
        self.batch_idx: int = 0

    @override
    def __iter__(self) -> _Sequential:
        self.iterator_idx: int = 0
        self.batch_idx: int = 0
        self.yielded: int = 0
        self.counts: list[int] = [0] * len(self.iterables)
        self._load_iterator()
        return self

    @override
    def __next__(self) -> _IteratorOutput:
        while self.iterator_idx < len(self.iterables):
            try:
                value: object = self._next(0)
            except StopIteration:
                self.iterator_idx += 1
                self.batch_idx: int = 0
                self._load_iterator()
                continue
            batch_idx: int = self.batch_idx
            dataloader_idx: int = self.iterator_idx
            self.batch_idx += 1
            self.yielded += 1
            return value, batch_idx, dataloader_idx
        raise StopIteration

    def _next(self, index: int) -> object:
        del index
        limit: int | None = self.limits[self.iterator_idx]
        if limit is not None and self.counts[self.iterator_idx] >= limit:
            raise StopIteration
        value: object = next(self.iterators[0])
        self.counts[self.iterator_idx] += 1
        return value

    def _load_iterator(self) -> None:
        self.iterators: list[Iterator[object]] = (
            [iter(self.iterables[self.iterator_idx])]
            if self.iterator_idx < len(self.iterables)
            else []
        )

    @override
    def reset(self) -> None:
        super().reset()
        self.iterator_idx: int = 0
        self.batch_idx: int = 0


_mode_iterators: dict[CombinedLoaderMode, type[_ModeIterator]] = {
    "min_size": _MinSize,
    "max_size_cycle": _MaxSizeCycle,
    "max_size": _MaxSize,
    "sequential": _Sequential
}


class CombinedLoader(Iterable[_IteratorOutput]):
    def __init__(
        self,
        iterables: object,
        mode: CombinedLoaderMode = "min_size"
    ) -> None:
        if mode not in _mode_iterators:
            raise ValueError(
                f"Unsupported mode {mode!r}; expected one of {tuple(_mode_iterators)}."
            )
        flattened: list[object]
        spec: _pytree.TreeSpec
        flattened, spec = _pytree.tree_flatten(
            iterables,
            is_leaf=_is_primitive_collection
        )
        self._iterables: object = iterables
        self._flattened: list[object] = flattened
        self._spec: _pytree.TreeSpec = spec
        self._mode: CombinedLoaderMode = mode
        self._limits: list[_Limit] | None = None
        self._iterator: _ModeIterator | None = None

    @property
    def iterables(self) -> object:
        return self._iterables

    @property
    def flattened(self) -> list[object]:
        return self._flattened

    @flattened.setter
    def flattened(self, flattened: list[object]) -> None:
        if len(flattened) != len(self._flattened):
            raise ValueError(
                f"Flattened length {len(flattened)} does not match {len(self._flattened)}."
            )
        self._flattened: list[object] = list(flattened)
        self._iterables: object = _pytree.tree_unflatten(
            self._flattened,
            self._spec
        )

    @property
    def sampler(self) -> object:
        values: list[object] = []
        iterable: object
        for iterable in self._flattened:
            values.append(getattr(iterable, "sampler", None))
        return _pytree.tree_unflatten(values, self._spec)

    @property
    def batch_sampler(self) -> object:
        values: list[object] = []
        iterable: object
        for iterable in self._flattened:
            values.append(getattr(iterable, "batch_sampler", None))
        return _pytree.tree_unflatten(values, self._spec)

    @property
    def limits(self) -> list[_Limit] | None:
        return self._limits

    @limits.setter
    def limits(self, limits: _Limit | Sequence[_Limit]) -> None:
        if limits is None:
            self._limits: list[_Limit] | None = None
            return
        if isinstance(limits, Sequence) and not isinstance(limits, (str, bytes)):
            limit_values: Sequence[_Limit] = cast(Sequence[_Limit], limits)
            if len(limit_values) != len(self._flattened):
                raise ValueError(
                    f"Limit count {len(limit_values)} does not match {len(self._flattened)} iterables."
                )
            normalized_limits: list[_Limit] = []
            limit: _Limit
            for limit in limit_values:
                normalized_limits.append(_normalize_limit(limit))
            self._limits: list[_Limit] | None = normalized_limits
            return
        normalized: _Limit = _normalize_limit(cast(_Limit, limits))
        self._limits: list[_Limit] | None = [normalized] * len(self._flattened)

    @property
    def mode(self) -> CombinedLoaderMode:
        return self._mode

    def _length(self) -> int | None:
        lengths: list[int | None] = []
        limits: list[int | None] = self._resolved_limits()
        iterable: object
        limit: int | None
        for iterable, limit in zip(self._flattened, limits, strict=True):
            length: int | None = _safe_len(iterable)
            if limit is not None:
                length: int | None = (
                    limit
                    if length is None
                    else min(length, limit)
                )
            lengths.append(length)
        if not lengths:
            return 0
        if self._mode == "min_size":
            finite: list[int] = []
            length: int | None
            for length in lengths:
                if length is not None:
                    finite.append(length)
            return min(finite) if finite else None
        if self._mode in {"max_size", "max_size_cycle"}:
            length: int | None
            for length in lengths:
                if length is None:
                    return None
            return max(cast(list[int], lengths))
        length: int | None
        for length in lengths:
            if length is None:
                return None
        return sum(cast(list[int], lengths))

    def _has_sized_iterables(self) -> bool:
        iterable: object
        for iterable in self._flattened:
            if _safe_len(iterable) is None:
                return False
        return True

    @override
    def __iter__(self) -> CombinedLoader:
        iterables: list[Iterable[object]] = []
        iterable: object
        for iterable in self._flattened:
            iterables.append(cast(Iterable[object], iterable))
        iterator_type: type[_ModeIterator] = _mode_iterators[self._mode]
        self._iterator: _ModeIterator | None = iterator_type(
            iterables,
            self._resolved_limits(),
            self._length()
        )
        try:
            iter(self._iterator)
        except TypeError as exception:
            raise TypeError(
                "CombinedLoader contains a non-iterable leaf."
            ) from exception
        return self

    def __next__(self) -> _IteratorOutput:
        if self._iterator is None:
            raise RuntimeError("CombinedLoader must be iterated before requesting a batch.")
        output: object
        batch_idx: int
        dataloader_idx: int
        output, batch_idx, dataloader_idx = next(self._iterator)
        if self._mode != "sequential":
            output: object = _pytree.tree_unflatten(
                cast(list[object], output),
                self._spec
            )
        return output, batch_idx, dataloader_idx

    def __len__(self) -> int:
        if self._iterator is None:
            raise RuntimeError("CombinedLoader must be iterated before requesting its length.")
        length: int | None = self._length()
        if length is None:
            raise TypeError("CombinedLoader length is unknown because at least one active iterable is unsized.")
        return length

    def _dataset_length(self) -> int:
        lengths: list[int] = []
        iterable: object
        for iterable in self._flattened:
            dataset: object | None = getattr(iterable, "dataset", None)
            length: int | None = _safe_len(dataset)
            if length is not None:
                lengths.append(length)
        if not lengths:
            raise NotImplementedError("All datasets are iterable-style datasets.")
        if self._mode == "min_size":
            return min(lengths)
        if self._mode in {"max_size", "max_size_cycle"}:
            return max(lengths)
        return sum(lengths)

    def _state_dicts(self) -> list[_LoaderState]:
        state_dicts: list[_LoaderState] = []
        iterable: object
        for iterable in self._flattened:
            if isinstance(iterable, _StatefulIterable):
                state_dicts.append(iterable.state_dict())
        return state_dicts

    def _load_state_dicts(self, state_dicts: list[_LoaderState]) -> None:
        if not state_dicts:
            return
        stateful_iterables: list[_StatefulIterable] = []
        candidate_iterable: object
        for candidate_iterable in self._flattened:
            if isinstance(candidate_iterable, _StatefulIterable):
                stateful_iterables.append(candidate_iterable)
        if len(stateful_iterables) != len(state_dicts):
            raise RuntimeError(
                f"CombinedLoader has {len(stateful_iterables)} stateful iterables, "
                f"but received {len(state_dicts)} states."
            )
        iterable: _StatefulIterable
        state_dict: _LoaderState
        for iterable, state_dict in zip(
            stateful_iterables,
            state_dicts,
            strict=True
        ):
            iterable.load_state_dict(state_dict)

    def reset(self) -> None:
        if self._iterator is not None:
            self._iterator.reset()
        self._iterator: _ModeIterator | None = None
        iterable: object
        for iterable in self._flattened:
            if hasattr(iterable, "_iterator"):
                if isinstance(
                    getattr(iterable, "_iterator"),
                    _MultiProcessingDataLoaderIter
                ):
                    delattr(iterable, "_iterator")
                setattr(iterable, "_iterator", None)

    def _resolved_limits(self) -> list[int | None]:
        if self._limits is None:
            return [None] * len(self._flattened)
        resolved_limits: list[int | None] = []
        limit: _Limit
        for limit in self._limits:
            resolved_limits.append(
                None
                if limit is None
                or (isinstance(limit, float) and math.isinf(limit))
                else int(limit)
            )
        return resolved_limits
