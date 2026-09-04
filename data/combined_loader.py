# This module:
# 1. Defines CombinedLoader, the training-data composition primitive that turns
#    one loader or a nested loader tree into the uniform iterator consumed by
#    fit, evaluation, prediction, and progress reporting
# 2. Implements the minimum-size, maximum-size, cycling, and sequential
#    traversal algorithms used to coordinate loaders with different lengths
# 3. Preserves sampler topology, explicit batch limits, stateful-loader
#    continuation, pickling, and multiprocessing worker cleanup
#
# Design decisions:
# - PyTorch pytree flattening preserves the client's dict, list, and tuple shape
#   while the traversal algorithms operate on one ordered leaf sequence; every
#   non-sequential batch is reconstructed into the original tree before return
# - Primitive scalar collections are treated as dataset or batch leaves rather
#   than mistaken for loader structure, allowing ordinary list- and tuple-based
#   samples to pass through unchanged
# - Each traversal policy is a private iterator with its own counters and reset
#   state because minimum, maximum, cycling, and sequential exhaustion have
#   different algorithms but no independent public lifecycle
# - Sequential mode returns one loader leaf at a time with its dataloader index;
#   the other modes return the reconstructed loader tree with index zero
# - Stateful child loaders are discovered structurally and restored in flattened
#   order, so checkpoint continuation preserves client loader implementations
#   without wrapping them in a SyntheticMind-specific adapter
# - reset removes persistent multiprocessing iterator references after shutting
#   down the active traversal, preventing completed stages from retaining worker
#   processes or stale prefetched batches
#
# Author: Rahul Sawhney

import math
from collections.abc import Iterable, Iterator, Sequence, Sized
from typing import Literal, Protocol, cast, override, runtime_checkable

from torch.utils import _pytree
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import (
    _BaseDataLoaderIter,
    _MultiProcessingDataLoaderIter
)

from syntheticmind.primitives.types import Batch, DataLoaderState

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
    # Structural checkpoint contract implemented by stateful dataloaders. A
    # client loader participates by exposing this pair; inheritance from a
    # SyntheticMind class is not required.
    def state_dict(self) -> DataLoaderState: ...

    def load_state_dict(self, state_dict: DataLoaderState) -> None: ...


class _ModeIterator(Iterator[_IteratorOutput]):
    # Shared traversal state for loader-combination algorithms: concrete leaves,
    # resolved limits, known combined length, live iterators, per-leaf counts,
    # and the number of combined outputs already yielded.
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
        # Recreates every child iterator and resets all counters, making each
        # CombinedLoader iteration a fresh traversal over the current leaves.
        self.iterators: list[Iterator[Batch]] = []
        iterable: Iterable[Batch]
        for iterable in self.iterables:
            self.iterators.append(iter(iterable))
        self.counts: list[int] = [0] * len(self.iterables)
        self.yielded: int = 0
        return self

    def _next(self, index: int) -> Batch:
        # Pulls one value from the selected leaf and advances its consumed count.
        value: Batch = next(self.iterators[index])
        self.counts[index] += 1
        return value

    def _is_complete(self) -> bool:
        # Applies an explicit or inferred combined-length ceiling before a mode
        # requests another child value.
        return self.length is not None and self.yielded >= self.length

    def reset(self) -> None:
        # Releases iterator references and returns the traversal to its initial
        # counter state without modifying the underlying loader objects.
        self.iterators: list[Iterator[Batch]] = []
        self.counts: list[int] = []
        self.yielded: int = 0

    @override
    def __getstate__(self) -> _ModeState:
        # Excludes PyTorch DataLoader iterators from pickled state because their
        # multiprocessing machinery cannot cross a launcher boundary safely;
        # ordinary Python iterators remain serializable when their type permits.
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
    # Stops the combined traversal as soon as any leaf is exhausted. Every
    # returned batch contains exactly one value from every loader.
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
    # Continues until every leaf is exhausted and substitutes None for leaves
    # that ended earlier, preserving the complete longest-loader horizon.
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
    # Rewinds shorter leaves when they exhaust and stops only after every leaf
    # has completed at least once or the resolved maximum length is reached.
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
        # Pulls every leaf, restarting exhausted leaves unless all leaves have
        # completed; per-leaf counters restart with each new cycle.
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
    # Traverses one loader completely before opening the next and returns the
    # active loader index with each batch, matching multi-loader evaluation and
    # prediction step signatures.
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
        # Advances through loaders in flattened order, resetting the local batch
        # index at each boundary while preserving the global yielded count.
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
        # Reads only the active sequential iterator and treats its per-loader
        # limit as an exhaustion boundary even when the iterable has more data.
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
    # Uniform iterator over one loader or a nested loader collection. Clients
    # normally return plain loaders from Module or DataModule factories and let
    # Trainer select the standard mode; construct CombinedLoader explicitly when
    # a research method requires a particular multi-loader traversal policy.
    #
    # Integration: every iteration yields ``(batch, batch_idx, dataloader_idx)``.
    # Non-sequential modes reconstruct batch into the supplied loader tree;
    # sequential mode yields the active leaf and reports its flattened index.
    #
    # Example::
    #
    #     from torch.utils.data import DataLoader
    #
    #     from syntheticmind import CombinedLoader
    #
    #     loaders: dict[str, DataLoader] = {
    #         "labeled": labeled_loader,
    #         "unlabeled": unlabeled_loader
    #     }
    #     combined: CombinedLoader = CombinedLoader(
    #         loaders,
    #         mode="max_size_cycle"
    #     )
    #     for batch, batch_idx, dataloader_idx in combined:
    #         labeled_batch = batch["labeled"]
    #         unlabeled_batch = batch["unlabeled"]
    #
    # The explicit limits property may bound each leaf independently before
    # iteration. Length becomes available only after iter(combined) initializes
    # the traversal, matching the live iterator whose horizon is reported.
    def __init__(
        self,
        iterables: _IterableCollection,
        mode: CombinedLoaderMode = "min_size"
    ) -> None:
        # Validates the requested algorithm, flattens the client collection with
        # primitive-batch awareness, and stores both representations so later
        # sampler replacement can update leaves without losing tree structure.
        #
        # Args:
        #     iterables: One iterable or a nested dict, list, or tuple of
        #         iterable or Dataset leaves.
        #     mode: Combination algorithm: ``"min_size"``,
        #         ``"max_size_cycle"``, ``"max_size"``, or ``"sequential"``.
        #         Default: ``"min_size"``.
        #
        # Raises:
        #     ValueError: If mode is not one of the supported algorithms.
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
        # Original loader collection with any later flattened-leaf replacements
        # reconstructed into the same client-visible tree shape.
        return self._iterables

    @property
    def flattened(self) -> list[_IterableLeaf]:
        # Ordered loader leaves used by connectors, loops, and sampler policy.
        return self._flattened

    @flattened.setter
    def flattened(self, flattened: list[_IterableLeaf]) -> None:
        # Replaces leaves without changing the pytree specification. DataConnector
        # uses this after distributed sampler reconstruction; a cardinality
        # change is rejected because it cannot satisfy the original tree.
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
        # Sampler attributes from every leaf reconstructed into loader topology;
        # leaves without a sampler contribute None.
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
        # Batch-sampler attributes from every leaf reconstructed into loader
        # topology; this remains separate from sampler for client inspection.
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
        # Per-leaf limits before normalization; None means no explicit ceiling.
        return self._limits

    @limits.setter
    def limits(self, limits: _Limit | Sequence[_Limit]) -> None:
        # Applies one limit to every leaf or validates a leaf-aligned sequence.
        # Positive infinity represents an unbounded leaf and finite floats must
        # be exact non-negative integers.
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
        # Immutable traversal algorithm selected at construction.
        return self._mode

    def _length(self) -> int | None:
        # Computes the combined horizon from effective per-leaf lengths: minimum
        # for min_size, maximum for maximum modes, sum for sequential, and None
        # whenever the selected algorithm depends on an unsized leaf.
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
        # Whether every loader leaf exposes a usable finite or bounded length.
        iterable: _IterableLeaf
        for iterable in self._flattened:
            if self._safe_length(iterable) is None:
                return False
        return True

    @override
    def __iter__(self) -> CombinedLoader:
        # Constructs the selected private iterator over current leaves and
        # resolved limits. Non-iterable leaves fail at this boundary before a
        # loop begins consuming partial training state.
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
        # Advances the initialized traversal and restores non-sequential outputs
        # to client topology; requesting a batch before iter() is a usage error.
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
        # Returns the active traversal horizon after initialization. Unknown
        # horizons remain explicit TypeError rather than fabricated estimates.
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
        # Computes sample cardinality from child dataset attributes for tuning
        # and overfit decisions, following the selected combination algorithm.
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
        # Captures state from structurally stateful leaves in flattened order and
        # rejects implementations that violate their mapping return contract.
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
        # Whether any child loader participates in exact cursor continuation.
        return any(
            isinstance(iterable, _StatefulIterable)
            for iterable in self._flattened
        )

    def _load_state_dicts(self, state_dicts: list[DataLoaderState]) -> None:
        # Restores states against the same flattened stateful-loader order. A
        # count mismatch rejects a changed data topology before any cursor moves.
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
        # Releases the mode iterator and clears persistent DataLoader worker
        # iterator references so the next stage or epoch starts from a clean
        # loader lifecycle.
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
        # Converts stored integral floats and positive infinity into the integer
        # or unbounded representation consumed by traversal algorithms.
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
        # Resolves the validated public mode into its private traversal class.
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
        # Tells pytree when a dict, list, or tuple is a scalar batch leaf rather
        # than loader structure by requiring every direct child to be primitive.
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
        # Reads Sized length while treating an implementation-level TypeError as
        # an intentionally unknown horizon.
        if not isinstance(value, Sized):
            return None
        try:
            return len(value)
        except TypeError:
            return None

    def _normalize_limit(self, limit: _Limit) -> _Limit:
        # Accepts non-negative integer counts and positive infinity only; bools,
        # fractional values, negative values, NaN, and negative infinity fail.
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
