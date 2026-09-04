import inspect
import warnings
from collections.abc import Callable, Iterable, Iterator, Sized
from multiprocessing.context import BaseContext
from typing import Literal, cast, override

import torch
from loguru import logger as log
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    DistributedSampler,
    IterableDataset,
    Sampler,
    SequentialSampler
)

from syntheticmind.core.datamodule import DataModule
from syntheticmind.core.module import Module
from syntheticmind.loops.batch_limits import resolve_batch_limit
from syntheticmind.profilers.profiler import Profiler
from syntheticmind.strategies.strategy import Strategy
from syntheticmind.utilities.combined_loader import CombinedLoader
from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.utilities.types import Batch, DataLoaderSource, TrainerStage

__all__: list[str] = []

type _DataStage = Literal["train", "val", "test", "predict"]
type _DataSource = Module | DataModule | DataLoaderSource | CombinedLoader
type _DataloaderResult = DataLoaderSource | CombinedLoader
type _DataloaderHook = Callable[[], _DataloaderResult]
type _ModuleHookCaller = Callable[[Module, str, _DataloaderHook], _DataloaderResult]
type _DataModuleHookCaller = Callable[[DataModule, str, _DataloaderHook], _DataloaderResult]
type _DataloaderLeaf = DataLoader | Iterable[Batch]
type _DataLoaderConstructorValue = (
    Batch
    | DataLoaderSource
    | Dataset[Batch]
    | Sampler[int]
    | Sampler[list[int]]
    | BatchSampler
    | Callable[..., Batch | None]
    | BaseContext
    | torch.Generator
    | bool
    | int
    | float
    | str
    | None
)


class _DataConnector:
    def __init__(
        self,
        strategy: Strategy,
        profiler: Profiler,
        limit_train_batches: int | float,
        limit_val_batches: int | float,
        limit_test_batches: int | float,
        limit_predict_batches: int | float,
        overfit_batches: int | float,
        use_distributed_sampler: bool,
        reload_dataloaders_every_n_epochs: int
    ) -> None:
        self._strategy: Strategy = strategy
        self._profiler: Profiler = profiler
        self._limits: dict[_DataStage, int | float] = {
            "train": limit_train_batches,
            "val": limit_val_batches,
            "test": limit_test_batches,
            "predict": limit_predict_batches
        }
        self._overfit_batches: int | float = overfit_batches
        self._use_distributed_sampler: bool = use_distributed_sampler
        self._reload_interval: int = reload_dataloaders_every_n_epochs
        self.train_combined_loader: CombinedLoader | None = None
        self.val_combined_loader: CombinedLoader | None = None
        self.test_combined_loader: CombinedLoader | None = None
        self.predict_combined_loader: CombinedLoader | None = None
        self.train_source: _DataSource | None = None
        self.val_source: _DataSource | None = None
        self.test_source: _DataSource | None = None
        self.predict_source: _DataSource | None = None
        self.total_train_batches: int | None = None
        self.total_val_batches: list[int | None] | None = None
        self.total_test_batches: list[int | None] | None = None
        self.total_predict_batches: list[int | None] | None = None
        self.last_train_reload_epoch: int | None = None
        self.last_val_reload_epoch: int | None = None

    def attach_source(
        self,
        stage: _DataStage,
        module: Module,
        datamodule: DataModule | None,
        dataloaders: DataLoaderSource | CombinedLoader | None
    ) -> None:
        source: _DataSource = (
            dataloaders
            if dataloaders is not None
            else datamodule
            if datamodule is not None
            else module
        )
        match stage:
            case "train":
                self.train_source: _DataSource | None = source
            case "val":
                self.val_source: _DataSource | None = source
            case "test":
                self.test_source: _DataSource | None = source
            case "predict":
                self.predict_source: _DataSource | None = source

    def request_dataloaders(
        self,
        stage: _DataStage,
        trainer_stage: TrainerStage,
        current_epoch: int,
        module_hook_caller: _ModuleHookCaller,
        datamodule_hook_caller: _DataModuleHookCaller
    ) -> CombinedLoader:
        hook_name: str = f"{stage}_dataloader"
        source: _DataSource | None = self._source(stage)
        if source is None:
            raise RuntimeError(f"No {stage} dataloader source is attached.")
        with self._profiler.profile(f"setup_{stage}_dataloader"):
            if isinstance(source, DataModule):
                datamodule_hook: _DataloaderHook = cast(
                    _DataloaderHook,
                    getattr(source, hook_name)
                )
                dataloader_result: _DataloaderResult = datamodule_hook_caller(
                    source,
                    hook_name,
                    datamodule_hook
                )
            elif isinstance(source, Module):
                module_hook: _DataloaderHook = cast(
                    _DataloaderHook,
                    getattr(source, hook_name)
                )
                dataloader_result: _DataloaderResult = module_hook_caller(
                    source,
                    hook_name,
                    module_hook
                )
            else:
                dataloader_result: _DataloaderResult = source
        combined_loader: CombinedLoader = (
            dataloader_result
            if isinstance(dataloader_result, CombinedLoader)
            else CombinedLoader(
                dataloader_result,
                "max_size_cycle" if stage == "train" else "sequential"
            )
        )
        if stage == "predict" and combined_loader.mode != "sequential":
            raise ValueError(
                "Trainer.predict() supports CombinedLoader only in sequential mode."
            )
        flattened: list[_DataloaderLeaf] = cast(
            list[_DataloaderLeaf],
            combined_loader.flattened
        )
        processed_dataloaders: list[_DataloaderLeaf] = []
        totals: list[int | None] = []
        dataloader: _DataloaderLeaf
        for dataloader in flattened:
            if not isinstance(dataloader, DataLoader):
                try:
                    iter(dataloader)
                except TypeError as exception:
                    argument_name: str = (
                        f"{stage}_dataloaders"
                        if trainer_stage == "fit"
                        else "dataloaders"
                    )
                    source_name: str = (
                        f"{type(source).__qualname__}.{hook_name}()"
                        if isinstance(source, Module | DataModule)
                        else f"Trainer.{trainer_stage}({argument_name}=...)"
                    )
                    raise MisconfigurationError(
                        f"{source_name} produced a non-iterable leaf of type "
                        f"{type(dataloader).__qualname__}."
                    ) from exception
            if isinstance(dataloader, DataLoader):
                dataloader: _DataloaderLeaf = self._prepare_dataloader(
                    dataloader,
                    stage,
                    trainer_stage
                )
            processed_dataloaders.append(dataloader)
            totals.append(self._safe_len(dataloader))
        combined_loader.flattened: list[_DataloaderLeaf] = processed_dataloaders
        combined_loader.limits: list[int | None] = [
            resolve_batch_limit(total, self._limits[stage], stage)
            for total in totals
        ]
        self._strategy.barrier()
        self._record_loader(stage, combined_loader, totals, current_epoch)
        return combined_loader

    def reset_fit(self) -> None:
        self.total_train_batches: int | None = None
        self.total_val_batches: list[int | None] | None = None
        self.train_combined_loader: CombinedLoader | None = None
        self.val_combined_loader: CombinedLoader | None = None
        self.last_train_reload_epoch: int | None = None
        self.last_val_reload_epoch: int | None = None

    def reset_stage(self, stage: Literal["val", "test", "predict"]) -> None:
        match stage:
            case "val":
                self.total_val_batches: list[int | None] | None = None
                self.val_combined_loader: CombinedLoader | None = None
            case "test":
                self.total_test_batches: list[int | None] | None = None
                self.test_combined_loader: CombinedLoader | None = None
            case "predict":
                self.total_predict_batches: list[int | None] | None = None
                self.predict_combined_loader: CombinedLoader | None = None

    def train_loader_is_current(self, current_epoch: int) -> bool:
        return self._loader_is_current(
            self.train_combined_loader,
            self.last_train_reload_epoch,
            current_epoch
        )

    def val_loader_is_current(self, current_epoch: int) -> bool:
        return self._loader_is_current(
            self.val_combined_loader,
            self.last_val_reload_epoch,
            current_epoch
        )

    @property
    def reload_interval(self) -> int:
        return self._reload_interval

    def _source(self, stage: _DataStage) -> _DataSource | None:
        match stage:
            case "train":
                return self.train_source
            case "val":
                return self.val_source
            case "test":
                return self.test_source
            case "predict":
                return self.predict_source

    def _prepare_dataloader(
        self,
        dataloader: DataLoader,
        stage: _DataStage,
        trainer_stage: TrainerStage
    ) -> DataLoader:
        prepared_dataloader: DataLoader = dataloader
        if self._overfit_batches and (
            stage == "train" or (stage == "val" and trainer_stage == "fit")
        ):
            prepared_dataloader: DataLoader = self._prepare_overfit_dataloader(
                prepared_dataloader,
                stage
            )
        if self._use_distributed_sampler and self._strategy.is_distributed:
            if isinstance(prepared_dataloader.dataset, IterableDataset):
                if self._strategy.is_global_zero:
                    log.warning(
                        f"Skipping DistributedSampler for {stage} dataloader: "
                        "IterableDataset detected."
                    )
            elif stage == "predict":
                prepared_dataloader: DataLoader = self._wrap_with_unrepeated_sampler(
                    prepared_dataloader
                )
            else:
                prepared_dataloader: DataLoader = self._wrap_with_distributed_sampler(
                    prepared_dataloader,
                    shuffle=(stage == "train" and not bool(self._overfit_batches))
                )
        if (
            self._reload_interval > 0
            and prepared_dataloader.persistent_workers
            and prepared_dataloader.pin_memory
        ):
            warnings.warn(
                "The combination of persistent workers, pinned memory, and dataloader "
                "reloading can be unstable in PyTorch.",
                UserWarning,
                stacklevel=3
            )
        return prepared_dataloader

    def _wrap_with_distributed_sampler(
        self,
        dataloader: DataLoader,
        shuffle: bool
    ) -> DataLoader:
        sampler_kwargs: dict[str, int] | None = (
            self._strategy.distributed_sampler_kwargs
        )
        num_replicas: int | None = (
            sampler_kwargs["num_replicas"] if sampler_kwargs is not None else None
        )
        rank: int | None = (
            sampler_kwargs["rank"] if sampler_kwargs is not None else None
        )
        sampler: DistributedSampler = DistributedSampler(
            dataloader.dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle
        )
        return self._replace_dataloader_sampler(dataloader, sampler)

    def _wrap_with_unrepeated_sampler(self, dataloader: DataLoader) -> DataLoader:
        sampler_kwargs: dict[str, int] | None = (
            self._strategy.distributed_sampler_kwargs
        )
        num_replicas: int | None = (
            sampler_kwargs["num_replicas"] if sampler_kwargs is not None else None
        )
        rank: int | None = (
            sampler_kwargs["rank"] if sampler_kwargs is not None else None
        )
        sampler: _UnrepeatedDistributedSampler = _UnrepeatedDistributedSampler(
            dataloader.dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False
        )
        return self._replace_dataloader_sampler(dataloader, sampler)

    def _prepare_overfit_dataloader(
        self,
        dataloader: DataLoader,
        stage: _DataStage
    ) -> DataLoader:
        if isinstance(dataloader.dataset, IterableDataset):
            return dataloader
        active_sampler: Sampler[int] | Iterable[int] = cast(
            Sampler[int] | Iterable[int],
            dataloader.sampler
        )
        if dataloader.batch_size is None and dataloader.batch_sampler is not None:
            active_sampler: Sampler[int] | Iterable[int] = cast(
                Sampler[int] | Iterable[int],
                getattr(
                    dataloader.batch_sampler,
                    "sampler",
                    dataloader.batch_sampler
                )
            )
        if isinstance(active_sampler, SequentialSampler):
            return dataloader
        warnings.warn(
            f"overfit_batches requires deterministic {stage} batches; "
            f"disabling {stage} dataloader shuffling.",
            UserWarning,
            stacklevel=3
        )
        sampler: SequentialSampler = SequentialSampler(
            cast(Sized, dataloader.dataset)
        )
        return self._replace_dataloader_sampler(dataloader, sampler)

    def _replace_dataloader_sampler(
        self,
        dataloader: DataLoader,
        sampler: Sampler[int]
    ) -> DataLoader:
        if dataloader.batch_size is None and dataloader.batch_sampler is not None:
            batch_sampler: BatchSampler = cast(
                BatchSampler,
                dataloader.batch_sampler
            )
            if not isinstance(batch_sampler, BatchSampler):
                raise MisconfigurationError(
                    "SyntheticMind cannot inject a sampler into a custom batch sampler "
                    "that does not inherit torch.utils.data.BatchSampler."
                )
            try:
                replaced_batch_sampler: BatchSampler = type(batch_sampler)(
                    sampler,
                    batch_sampler.batch_size,
                    batch_sampler.drop_last
                )
            except TypeError as exception:
                raise MisconfigurationError(
                    f"SyntheticMind cannot reconstruct custom batch sampler "
                    f"{type(batch_sampler).__qualname__}. Its constructor must accept "
                    "sampler, batch_size, and drop_last."
                ) from exception
            return self._reinstantiate_dataloader(
                dataloader,
                self._dataloader_arguments(dataloader, replaced_batch_sampler)
            )
        arguments: dict[str, _DataLoaderConstructorValue] = self._dataloader_arguments(
            dataloader,
            sampler
        )
        arguments["batch_size"]: _DataLoaderConstructorValue = dataloader.batch_size
        arguments["drop_last"]: _DataLoaderConstructorValue = dataloader.drop_last
        return self._reinstantiate_dataloader(dataloader, arguments)

    def _dataloader_arguments(
        self,
        dataloader: DataLoader,
        sampler: Sampler[int] | BatchSampler
    ) -> dict[str, _DataLoaderConstructorValue]:
        arguments: dict[str, _DataLoaderConstructorValue] = {
            "dataset": cast(_DataLoaderConstructorValue, dataloader.dataset),
            "num_workers": dataloader.num_workers,
            "collate_fn": cast(_DataLoaderConstructorValue, dataloader.collate_fn),
            "pin_memory": dataloader.pin_memory,
            "timeout": dataloader.timeout,
            "worker_init_fn": cast(
                _DataLoaderConstructorValue,
                dataloader.worker_init_fn
            ),
            "multiprocessing_context": cast(
                _DataLoaderConstructorValue,
                getattr(dataloader, "multiprocessing_context", None)
            ),
            "generator": cast(
                _DataLoaderConstructorValue,
                getattr(dataloader, "generator", None)
            ),
            "prefetch_factor": dataloader.prefetch_factor,
            "persistent_workers": dataloader.persistent_workers,
            "pin_memory_device": cast(
                str,
                getattr(dataloader, "pin_memory_device", "")
            ),
            "in_order": cast(bool, getattr(dataloader, "in_order", True))
        }
        if isinstance(sampler, BatchSampler):
            arguments["batch_sampler"]: _DataLoaderConstructorValue = sampler
        else:
            arguments["sampler"]: _DataLoaderConstructorValue = sampler
        return arguments

    def _reinstantiate_dataloader(
        self,
        dataloader: DataLoader,
        constructor_kwargs: dict[str, _DataLoaderConstructorValue]
    ) -> DataLoader:
        constructor: type[DataLoader] = type(dataloader)
        parameters: dict[str, inspect.Parameter] = dict(
            inspect.signature(constructor).parameters
        )
        base_constructor: type
        for base_constructor in constructor.__mro__[1:]:
            if not issubclass(base_constructor, DataLoader):
                continue
            base_name: str
            base_parameter: inspect.Parameter
            for base_name, base_parameter in inspect.signature(
                base_constructor
            ).parameters.items():
                parameters.setdefault(base_name, base_parameter)
        dataloader_parameters: set[str] = set(
            inspect.signature(DataLoader).parameters
        )
        accepts_kwargs: bool = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        name: str
        parameter: inspect.Parameter
        for name, parameter in parameters.items():
            if (
                name not in constructor_kwargs
                and parameter.kind
                in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY
                }
                and hasattr(dataloader, name)
                and (
                    parameter.default is inspect.Parameter.empty
                    or name not in dataloader_parameters
                )
            ):
                constructor_kwargs[name]: _DataLoaderConstructorValue = cast(
                    _DataLoaderConstructorValue,
                    getattr(dataloader, name)
                )
        missing: list[str] = [
            name
            for name, parameter in parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY
            }
            and name not in constructor_kwargs
        ]
        if missing:
            raise MisconfigurationError(
                f"SyntheticMind cannot reconstruct {constructor.__qualname__} with a "
                f"replacement sampler because required constructor values {missing!r} "
                "are not available as dataloader attributes."
            )
        positional_args: list[_DataLoaderConstructorValue] = []
        for name, parameter in parameters.items():
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                positional_args.append(constructor_kwargs.pop(name))
        if not accepts_kwargs:
            unexpected: list[str] = [
                name
                for name in constructor_kwargs
                if name not in parameters
            ]
            if unexpected:
                raise MisconfigurationError(
                    f"SyntheticMind cannot reconstruct {constructor.__qualname__} with a "
                    f"replacement sampler because its constructor does not accept "
                    f"DataLoader parameters {unexpected!r}."
                )
        try:
            constructor_callable: Callable[..., DataLoader] = cast(
                Callable[..., DataLoader],
                constructor
            )
            rebuilt: DataLoader = constructor_callable(
                *positional_args,
                **constructor_kwargs
            )
        except TypeError as exception:
            raise MisconfigurationError(
                f"SyntheticMind could not reconstruct {constructor.__qualname__} with "
                "the replacement sampler. Its constructor must preserve the standard "
                "DataLoader parameter contract."
            ) from exception
        if not isinstance(rebuilt, DataLoader):
            raise TypeError(
                f"Reconstructing {constructor.__qualname__} returned "
                f"{type(rebuilt).__qualname__}, not a DataLoader."
            )
        return rebuilt

    def _safe_len(self, dataloader: _DataloaderLeaf) -> int | None:
        if not isinstance(dataloader, Sized):
            return None
        try:
            return len(dataloader)
        except TypeError:
            return None

    def _record_loader(
        self,
        stage: _DataStage,
        combined_loader: CombinedLoader,
        totals: list[int | None],
        current_epoch: int
    ) -> None:
        match stage:
            case "train":
                self.train_combined_loader: CombinedLoader | None = combined_loader
                self.total_train_batches: int | None = combined_loader._length()
                self.last_train_reload_epoch: int | None = current_epoch
            case "val":
                self.val_combined_loader: CombinedLoader | None = combined_loader
                self.total_val_batches: list[int | None] | None = totals
                self.last_val_reload_epoch: int | None = current_epoch
            case "test":
                self.test_combined_loader: CombinedLoader | None = combined_loader
                self.total_test_batches: list[int | None] | None = totals
            case "predict":
                self.predict_combined_loader: CombinedLoader | None = combined_loader
                self.total_predict_batches: list[int | None] | None = totals

    def _loader_is_current(
        self,
        combined_loader: CombinedLoader | None,
        last_reload_epoch: int | None,
        current_epoch: int
    ) -> bool:
        return combined_loader is not None and (
            self._reload_interval == 0
            or (
                last_reload_epoch is not None
                and current_epoch - last_reload_epoch < self._reload_interval
            )
        )


class _UnrepeatedDistributedSampler(DistributedSampler):
    @override
    def __iter__(self) -> Iterator[int]:
        dataset: Sized = cast(Sized, self.dataset)
        if self.shuffle:
            generator: torch.Generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices: list[int] = torch.randperm(
                len(dataset),
                generator=generator
            ).tolist()
        else:
            indices: list[int] = list(range(len(dataset)))
        rank_indices: list[int] = indices[self.rank::self.num_replicas]
        return iter(rank_indices)

    @override
    def __len__(self) -> int:
        dataset: Sized = cast(Sized, self.dataset)
        return max(0, (len(dataset) - self.rank + self.num_replicas - 1) // self.num_replicas)
