import math
from collections.abc import Iterable, Iterator, Sequence, Sized
from typing import Literal, Protocol, cast, override, runtime_checkable

from torch.utils import _pytree
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import (
    _BaseDataLoaderIter,
    _MultiProcessingDataLoaderIter
)

from syntheticmind.types import Batch, DataLoaderState

__all__: list[str] = ["CombinedLoader", "CombinedLoaderMode"]

type CombinedLoaderMode = Literal[
    "min_size",
    "max_size_cycle",
    "max_size",
    "sequential"
]
type _IterableLeaf = Iterable[Batch] | Dataset[Batch]
type _IterableCollection = (
    _IterableLeaf
    | dict[str, _IterableCollection]
    | list[_IterableCollection]
    | tuple[_IterableCollection, ...]
)
type _TreeNode = Batch | _IterableCollection
type _IteratorOutput = tuple[Batch, int, int]
type _Limit = int | float | None
type _SamplerLeaf = Sampler[int] | Sampler[list[int]] | None
type _SamplerCollection = (
    _SamplerLeaf
    | dict[str, _SamplerCollection]
    | list[_SamplerCollection]
    | tuple[_SamplerCollection, ...]
)
type _ModeStateValue = (
    list[Iterable[Batch]]
    | list[Iterator[Batch] | None]
    | list[int]
    | list[int | None]
    | list[bool]
    | int
    | None
)
type _ModeState = dict[str, _ModeStateValue]


@runtime_checkable
class _StatefulIterable(Protocol):
    def state_dict(self) -> DataLoaderState: ...

    def load_state_dict(self, state_dict: DataLoaderState) -> None: ...


class _ModeIterator(Iterator[_IteratorOutput]):
    def __init__(
        self,
        iterables: list[Iterable[Batch]],
        limits: list[int | None],
        length: int | None
    ) -> None:
        self.iterables: list[Iterable[Batch]] = iterables
        self.limits: list[int | None] = limits
        self.length: int | None = length
        self.iterators: list[Iterator[Batch]] = []
        self.counts: list[int] = []
        self.yielded: int = 0

    @override
    def __iter__(self) -> _ModeIterator:
        self.iterators: list[Iterator[Batch]] = []
        iterable: Iterable[Batch]
        for iterable in self.iterables:
            self.iterators.append(iter(iterable))
        self.counts: list[int] = [0] * len(self.iterables)
        self.yielded: int = 0
        return self

    def _next(self, index: int) -> Batch:
        value: Batch = next(self.iterators[index])
        self.counts[index] += 1
        return value

    def _is_complete(self) -> bool:
        return self.length is not None and self.yielded >= self.length

    def reset(self) -> None:
        self.iterators: list[Iterator[Batch]] = []
        self.counts: list[int] = []
        self.yielded: int = 0

    @override
    def __getstate__(self) -> _ModeState:
        state: _ModeState = cast(_ModeState, dict(self.__dict__))
        serializable_iterators: list[Iterator[Batch] | None] = []
        iterator: Iterator[Batch]
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
        values: list[Batch] = []
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
        values: list[Batch] = [None] * len(self.iterators)
        yielded_value: bool = False
        index: int
        for index in range(len(self.iterators)):
            try:
                values[index] = self._next(index)
                yielded_value: bool = True
            except StopIteration:
                continue
        if not yielded_value:
            raise StopIteration
        batch_idx: int = self.yielded
        self.yielded += 1
        return values, batch_idx, 0


class _MaxSizeCycle(_ModeIterator):
    @override
    def __init__(
        self,
        iterables: list[Iterable[Batch]],
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
        values: list[Batch] = []
        index: int
        iterable: Iterable[Batch]
        for index, iterable in enumerate(self.iterables):
            try:
                value: Batch = self._next(index)
            except StopIteration:
                self.completed[index] = True
                if all(self.completed):
                    raise
                self.iterators[index] = iter(iterable)
                self.counts[index] = 0
                value: Batch = self._next(index)
            values.append(value)
        batch_idx: int = self.yielded
        self.yielded += 1
        return values, batch_idx, 0

    @override
    def reset(self) -> None:
        super().reset()
        self.completed: list[bool] = []


class _Sequential(_ModeIterator):
    @override
    def __init__(
        self,
        iterables: list[Iterable[Batch]],
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
                value: Batch = self._next(0)
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

    @override
    def _next(self, index: int) -> Batch:
        del index
        limit: int | None = self.limits[self.iterator_idx]
        if limit is not None and self.counts[self.iterator_idx] >= limit:
            raise StopIteration
        value: Batch = next(self.iterators[0])
        self.counts[self.iterator_idx] += 1
        return value

    def _load_iterator(self) -> None:
        self.iterators: list[Iterator[Batch]] = (
            [iter(self.iterables[self.iterator_idx])]
            if self.iterator_idx < len(self.iterables)
            else []
        )

    @override
    def reset(self) -> None:
        super().reset()
        self.iterator_idx: int = 0
        self.batch_idx: int = 0


class CombinedLoader(Iterator[_IteratorOutput]):
    def __init__(
        self,
        iterables: _IterableCollection,
        mode: CombinedLoaderMode = "min_size"
    ) -> None:
        if mode not in ("min_size", "max_size_cycle", "max_size", "sequential"):
            raise ValueError(
                f"Unsupported mode {mode!r}; expected min_size, max_size_cycle, "
                "max_size, or sequential."
            )
        flattened_tree: tuple[list[_IterableLeaf], _pytree.TreeSpec] = cast(
            tuple[list[_IterableLeaf], _pytree.TreeSpec],
            _pytree.tree_flatten(
                iterables,
                is_leaf=self._is_primitive_collection
            )
        )
        flattened: list[_IterableLeaf] = flattened_tree[0]
        spec: _pytree.TreeSpec = flattened_tree[1]
        self._iterables: _IterableCollection = iterables
        self._flattened: list[_IterableLeaf] = flattened
        self._spec: _pytree.TreeSpec = spec
        self._mode: CombinedLoaderMode = mode
        self._limits: list[_Limit] | None = None
        self._iterator: _ModeIterator | None = None

    @property
    def iterables(self) -> _IterableCollection:
        return self._iterables

    @property
    def flattened(self) -> list[_IterableLeaf]:
        return self._flattened

    @flattened.setter
    def flattened(self, flattened: list[_IterableLeaf]) -> None:
        if len(flattened) != len(self._flattened):
            raise ValueError(
                f"Flattened length {len(flattened)} does not match {len(self._flattened)}."
            )
        self._flattened: list[_IterableLeaf] = list(flattened)
        self._iterables: _IterableCollection = cast(
            _IterableCollection,
            _pytree.tree_unflatten(self._flattened, self._spec)
        )

    @property
    def sampler(self) -> _SamplerCollection:
        values: list[_SamplerLeaf] = []
        iterable: _IterableLeaf
        for iterable in self._flattened:
            values.append(
                cast(_SamplerLeaf, getattr(iterable, "sampler", None))
            )
        return cast(
            _SamplerCollection,
            _pytree.tree_unflatten(values, self._spec)
        )

    @property
    def batch_sampler(self) -> _SamplerCollection:
        values: list[_SamplerLeaf] = []
        iterable: _IterableLeaf
        for iterable in self._flattened:
            values.append(
                cast(_SamplerLeaf, getattr(iterable, "batch_sampler", None))
            )
        return cast(
            _SamplerCollection,
            _pytree.tree_unflatten(values, self._spec)
        )

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
                    f"Limit count {len(limit_values)} does not match "
                    f"{len(self._flattened)} iterables."
                )
            normalized_limits: list[_Limit] = []
            limit: _Limit
            for limit in limit_values:
                normalized_limits.append(self._normalize_limit(limit))
            self._limits: list[_Limit] | None = normalized_limits
            return
        normalized: _Limit = self._normalize_limit(cast(_Limit, limits))
        self._limits: list[_Limit] | None = [normalized] * len(self._flattened)

    @property
    def mode(self) -> CombinedLoaderMode:
        return self._mode

    def _length(self) -> int | None:
        lengths: list[int | None] = []
        limits: list[int | None] = self._resolved_limits()
        iterable: _IterableLeaf
        limit: int | None
        for iterable, limit in zip(self._flattened, limits, strict=True):
            length: int | None = self._safe_length(iterable)
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
        iterable: _IterableLeaf
        for iterable in self._flattened:
            if self._safe_length(iterable) is None:
                return False
        return True

    @override
    def __iter__(self) -> CombinedLoader:
        iterables: list[Iterable[Batch]] = []
        iterable: _IterableLeaf
        for iterable in self._flattened:
            iterables.append(cast(Iterable[Batch], iterable))
        iterator_type: type[_ModeIterator] = self._iterator_type()
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

    @override
    def __next__(self) -> _IteratorOutput:
        if self._iterator is None:
            raise RuntimeError("CombinedLoader must be iterated before requesting a batch.")
        iterator_output: _IteratorOutput = next(self._iterator)
        output: Batch = iterator_output[0]
        batch_idx: int = iterator_output[1]
        dataloader_idx: int = iterator_output[2]
        if self._mode != "sequential":
            output: Batch = cast(
                Batch,
                _pytree.tree_unflatten(
                    cast(list[Batch], output),
                    self._spec
                )
            )
        return output, batch_idx, dataloader_idx

    def __len__(self) -> int:
        if self._iterator is None:
            raise RuntimeError("CombinedLoader must be iterated before requesting its length.")
        length: int | None = self._length()
        if length is None:
            raise TypeError(
                "CombinedLoader length is unknown because at least one active "
                "iterable is unsized."
            )
        return length

    def _dataset_length(self) -> int:
        lengths: list[int] = []
        iterable: _IterableLeaf
        for iterable in self._flattened:
            dataset: Dataset[Batch] | None = cast(
                Dataset[Batch] | None,
                getattr(iterable, "dataset", None)
            )
            length: int | None = self._safe_length(dataset)
            if length is not None:
                lengths.append(length)
        if not lengths:
            raise NotImplementedError("All datasets are iterable-style datasets.")
        if self._mode == "min_size":
            return min(lengths)
        if self._mode in {"max_size", "max_size_cycle"}:
            return max(lengths)
        return sum(lengths)

    def _state_dicts(self) -> list[DataLoaderState]:
        state_dicts: list[DataLoaderState] = []
        iterable: _IterableLeaf
        for iterable in self._flattened:
            if isinstance(iterable, _StatefulIterable):
                state_dict: DataLoaderState = iterable.state_dict()
                if not isinstance(state_dict, dict):
                    raise TypeError(
                        f"{type(iterable).__qualname__}.state_dict() must return a "
                        f"mapping, got {type(state_dict).__qualname__}."
                    )
                state_dicts.append(state_dict)
        return state_dicts

    def _has_stateful_iterables(self) -> bool:
        return any(
            isinstance(iterable, _StatefulIterable)
            for iterable in self._flattened
        )

    def _load_state_dicts(self, state_dicts: list[DataLoaderState]) -> None:
        if not state_dicts:
            return
        stateful_iterables: list[_StatefulIterable] = []
        candidate_iterable: _IterableLeaf
        for candidate_iterable in self._flattened:
            if isinstance(candidate_iterable, _StatefulIterable):
                stateful_iterables.append(candidate_iterable)
        if len(stateful_iterables) != len(state_dicts):
            raise RuntimeError(
                f"CombinedLoader has {len(stateful_iterables)} stateful iterables, "
                f"but received {len(state_dicts)} states. Ensure the resumed loader "
                "collection matches the checkpointed loader collection."
            )
        iterable: _StatefulIterable
        state_dict: DataLoaderState
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
        iterable: _IterableLeaf
        for iterable in self._flattened:
            if not isinstance(iterable, DataLoader):
                continue
            data_iterator: _BaseDataLoaderIter | None = cast(
                _BaseDataLoaderIter | None,
                getattr(iterable, "_iterator", None)
            )
            if isinstance(data_iterator, _MultiProcessingDataLoaderIter):
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

    def _iterator_type(self) -> type[_ModeIterator]:
        match self._mode:
            case "min_size":
                return _MinSize
            case "max_size_cycle":
                return _MaxSizeCycle
            case "max_size":
                return _MaxSize
            case "sequential":
                return _Sequential

    def _is_primitive_collection(self, value: _TreeNode) -> bool:
        if isinstance(value, dict):
            children: Iterable[_TreeNode] = cast(
                dict[str, _TreeNode],
                value
            ).values()
        elif isinstance(value, list | tuple):
            children: Iterable[_TreeNode] = cast(
                list[_TreeNode] | tuple[_TreeNode, ...],
                value
            )
        else:
            return False
        child: _TreeNode
        for child in children:
            if not isinstance(child, (bool, int, float, str, bytes, type(None))):
                return False
        return True

    def _safe_length(self, value: _TreeNode | None) -> int | None:
        if not isinstance(value, Sized):
            return None
        try:
            return len(value)
        except TypeError:
            return None

    def _normalize_limit(self, limit: _Limit) -> _Limit:
        if limit is None:
            return None
        if type(limit) is bool or not isinstance(limit, (int, float)):
            raise TypeError(
                "CombinedLoader limits must be non-negative integer counts or infinity."
            )
        if isinstance(limit, int):
            if limit < 0:
                raise ValueError(
                    "CombinedLoader limits must be non-negative integer counts or infinity."
                )
            return limit
        if math.isinf(limit) and limit > 0:
            return limit
        if not math.isfinite(limit) or limit < 0 or int(limit) != limit:
            raise ValueError(
                "CombinedLoader limits must be non-negative integer counts or infinity."
            )
        return limit
