import inspect
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import IO, ClassVar, Self, cast, override

import syntheticmind
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from syntheticmind.core.hooks import DataHooks
from syntheticmind.core.hyperparameters import HyperparametersMixin
from syntheticmind.core.saving import _DataModuleCheckpointLoader
from syntheticmind.combined_loader import CombinedLoader
from syntheticmind.exceptions import MisconfigurationError
from syntheticmind.types import (
    Batch,
    DataLoaderSource,
    DatasetSource,
    HyperparameterValue,
    MapLocation,
    StateDict
)

__all__: list[str] = ["DataModule", "DatasetDataModule"]

type _DataLoaderFactory = Callable[[], DataLoaderSource]


class DataModule(DataHooks, HyperparametersMixin):
    _checkpoint_hyperparameters_key: ClassVar[str] = "datamodule_hyper_parameters"
    _checkpoint_hyperparameters_name: ClassVar[str] = "datamodule_hparams_name"
    _checkpoint_hyperparameters_type: ClassVar[str] = "datamodule_hparams_type"

    @override
    def __init__(self) -> None:
        super().__init__()
        self.trainer: syntheticmind.Trainer | None = None

    @classmethod
    def from_datasets(
        cls,
        train_dataset: DatasetSource | None = None,
        val_dataset: DatasetSource | None = None,
        test_dataset: DatasetSource | None = None,
        predict_dataset: DatasetSource | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        **datamodule_kwargs: HyperparameterValue
    ) -> Self:
        parameters: dict[str, inspect.Parameter] = dict(
            inspect.signature(cls.__init__).parameters
        )
        accepts_kwargs: bool = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        constructor_kwargs: dict[str, HyperparameterValue] = dict(datamodule_kwargs)
        if accepts_kwargs or "batch_size" in parameters:
            constructor_kwargs["batch_size"] = batch_size
        if accepts_kwargs or "num_workers" in parameters:
            constructor_kwargs["num_workers"] = num_workers
        datamodule: Self = cls(**constructor_kwargs)
        provider: DatasetDataModule = DatasetDataModule(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            predict_dataset=predict_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        if train_dataset is not None:
            setattr(datamodule, "train_dataloader", provider.train_dataloader)
        if val_dataset is not None:
            setattr(datamodule, "val_dataloader", provider.val_dataloader)
        if test_dataset is not None:
            setattr(datamodule, "test_dataloader", provider.test_dataloader)
        if predict_dataset is not None:
            setattr(datamodule, "predict_dataloader", provider.predict_dataloader)
        return datamodule

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | IO[bytes],
        map_location: MapLocation = None,
        hparams_file: str | Path | None = None,
        weights_only: bool | None = None,
        **kwargs: HyperparameterValue
    ) -> Self:
        return _DataModuleCheckpointLoader.load(
            cls,
            checkpoint_path,
            map_location,
            hparams_file,
            weights_only,
            kwargs
        )

    def state_dict(self) -> StateDict:
        state: StateDict = {}
        return state

    def load_state_dict(self, state_dict: StateDict) -> None:
        del state_dict

    def on_exception(self, exception: BaseException) -> None:
        del exception

    def _configured_loader_stages(self) -> list[str]:
        configured_loaders: list[str] = []
        stages: tuple[str, ...] = ("train", "val", "test", "predict")
        stage: str
        for stage in stages:
            method_name: str = f"{stage}_dataloader"
            if method_name in self.__dict__:
                configured_loaders.append(stage)
                continue
            class_method: _DataLoaderFactory | None = cast(
                _DataLoaderFactory | None,
                getattr(type(self), method_name, None)
            )
            base_method: _DataLoaderFactory | None = cast(
                _DataLoaderFactory | None,
                getattr(DataHooks, method_name, None)
            )
            if class_method is not None and class_method is not base_method:
                configured_loaders.append(stage)
        return configured_loaders

    @override
    def __repr__(self) -> str:
        configured_loaders: list[str] = self._configured_loader_stages()
        class_name: str = type(self).__name__
        loaders_repr: str = ", ".join(
            f"{name}=True" for name in configured_loaders
        )
        return f"{class_name}({loaders_repr})"


class DatasetDataModule(DataModule):
    @override
    def __init__(
        self,
        train_dataset: DatasetSource | None = None,
        val_dataset: DatasetSource | None = None,
        test_dataset: DatasetSource | None = None,
        predict_dataset: DatasetSource | None = None,
        batch_size: int = 1,
        num_workers: int = 0
    ) -> None:
        super().__init__()
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least one, got {batch_size}.")
        if num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {num_workers}.")
        self._train_dataset: DatasetSource | None = train_dataset
        self._val_dataset: DatasetSource | None = val_dataset
        self._test_dataset: DatasetSource | None = test_dataset
        self._predict_dataset: DatasetSource | None = predict_dataset
        self._batch_size: int = batch_size
        self._num_workers: int = num_workers

    @override
    def train_dataloader(self) -> DataLoaderSource:
        if self._train_dataset is None:
            raise MisconfigurationError(
                "train_dataloader() is unavailable because no train_dataset was supplied."
            )
        return self._build_dataloaders(self._train_dataset, shuffle=True)

    @override
    def val_dataloader(self) -> DataLoaderSource:
        if self._val_dataset is None:
            raise MisconfigurationError(
                "val_dataloader() is unavailable because no val_dataset was supplied."
            )
        return self._build_dataloaders(self._val_dataset, shuffle=False)

    @override
    def test_dataloader(self) -> DataLoaderSource:
        if self._test_dataset is None:
            raise MisconfigurationError(
                "test_dataloader() is unavailable because no test_dataset was supplied."
            )
        return self._build_dataloaders(self._test_dataset, shuffle=False)

    @override
    def predict_dataloader(self) -> DataLoaderSource:
        if self._predict_dataset is None:
            raise MisconfigurationError(
                "predict_dataloader() is unavailable because no predict_dataset was supplied."
            )
        return self._build_dataloaders(self._predict_dataset, shuffle=False)

    @override
    def _configured_loader_stages(self) -> list[str]:
        configured_loaders: list[str] = []
        if self._train_dataset is not None:
            configured_loaders.append("train")
        if self._val_dataset is not None:
            configured_loaders.append("val")
        if self._test_dataset is not None:
            configured_loaders.append("test")
        if self._predict_dataset is not None:
            configured_loaders.append("predict")
        return configured_loaders

    def _build_dataloaders(
        self,
        datasets: DatasetSource,
        shuffle: bool
    ) -> DataLoaderSource:
        resolved_datasets: DatasetSource = datasets
        if (
            not isinstance(datasets, Dataset)
            and isinstance(datasets, Iterable)
            and not isinstance(datasets, (dict, list, tuple, str, bytes))
        ):
            resolved_datasets: DatasetSource = cast(
                DatasetSource,
                list(datasets)
            )
        combined_loader: CombinedLoader = CombinedLoader(resolved_datasets)
        flattened_datasets: list[DatasetSource] = cast(
            list[DatasetSource],
            combined_loader.flattened
        )
        dataloaders: list[DataLoader[Batch]] = []
        dataset_candidate: DatasetSource
        for dataset_candidate in flattened_datasets:
            if not isinstance(dataset_candidate, Dataset):
                raise TypeError(
                    "DataModule.from_datasets() requires torch Dataset leaves, got "
                    f"{type(dataset_candidate).__qualname__}."
                )
            dataset: Dataset[Batch] = cast(Dataset[Batch], dataset_candidate)
            dataloader: DataLoader[Batch] = DataLoader(
                dataset,
                batch_size=self._batch_size,
                shuffle=shuffle and not isinstance(dataset, IterableDataset),
                num_workers=self._num_workers,
                pin_memory=True
            )
            dataloaders.append(dataloader)
        combined_loader.flattened: list[DataLoader[Batch]] = dataloaders
        return cast(DataLoaderSource, combined_loader.iterables)
