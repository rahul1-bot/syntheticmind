# This module:
# 1. Defines DataModule, the data lifecycle clients may subclass to prepare,
#    construct, transfer, checkpoint, and release stage-specific data
# 2. Defines DatasetDataModule, the concrete adapter that builds standard
#    dataloaders from supplied map-style, iterable, or nested dataset sources
# 3. Reconstructs DataModule subclasses from SyntheticMind checkpoints with
#    constructor hyperparameters and component state
#
# Design decisions:
# - DataModule composes DataHooks and HyperparametersMixin so one client class
#   owns preparation, setup, loader factories, transfer policy, state, and
#   reconstructable configuration without a wrapper object
# - prepare_data remains distinct from setup: Trainer controls whether
#   preparation executes once globally or once per node, while setup executes in
#   every participating process and may allocate rank-local dataset state
# - DataModule.from_datasets constructs the requested subclass and binds only
#   supplied stage factories from a DatasetDataModule provider, preserving the
#   subclass identity and its additional lifecycle behavior
# - DatasetDataModule uses CombinedLoader only to preserve nested dataset shape;
#   the returned value remains the corresponding nested dataloader collection
# - Map-style training datasets shuffle, iterable datasets retain their own
#   ordering, and validation, test, and prediction never shuffle by default
# - DataModule state crosses the same schema-v2 checkpoint boundary as model
#   state, but model strictness, parameter loading, configure_model, and device
#   placement remain excluded because they are not data lifecycle concepts
#
# Author: Rahul Sawhney

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
from syntheticmind.data.combined_loader import CombinedLoader
from syntheticmind.primitives.exceptions import MisconfigurationError
from syntheticmind.primitives.types import (
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
    # Base data contract consumed by Trainer. A client overrides the lifecycle
    # and dataloader methods required by its project; inherited defaults keep
    # unused stages inert and route nested batches through standard device
    # transfer. The Trainer attaches itself before setup and clears the reference
    # after stage teardown.
    #
    # Integration: subclass DataModule when dataset construction, external
    # resources, distributed process setup, custom batch transfer, or resumable
    # data state belongs to the project rather than to the model.
    #
    # Example::
    #
    #     from pathlib import Path
    #
    #     import torch
    #     from torch.utils.data import DataLoader
    #
    #     from syntheticmind import DataModule
    #     from syntheticmind.primitives.types import TrainerStage
    #
    #     class ImageDataModule(DataModule):
    #         def __init__(self, dataset_root: Path, batch_size: int) -> None:
    #             super().__init__()
    #             self.save_hyperparameters()
    #             self._dataset_root: Path = dataset_root
    #             self._batch_size: int = batch_size
    #             self._train_dataset: torch.utils.data.Dataset | None = None
    #
    #         def prepare_data(self) -> None:
    #             ...
    #
    #         def setup(self, stage: TrainerStage) -> None:
    #             if stage == "fit":
    #                 self._train_dataset = build_training_dataset(
    #                     self._dataset_root
    #                 )
    #
    #         def train_dataloader(self) -> DataLoader:
    #             return DataLoader(
    #                 self._train_dataset,
    #                 batch_size=self._batch_size,
    #                 shuffle=True
    #             )
    _checkpoint_hyperparameters_key: ClassVar[str] = "datamodule_hyper_parameters"
    _checkpoint_hyperparameters_name: ClassVar[str] = "datamodule_hparams_name"
    _checkpoint_hyperparameters_type: ClassVar[str] = "datamodule_hparams_type"

    @override
    def __init__(self) -> None:
        # Initializes cooperative hook and hyperparameter state and starts
        # detached from a Trainer. Subclasses must call super().__init__().
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
        # Constructs the requested DataModule subclass and attaches standard
        # loader factories for exactly the supplied dataset stages. Constructor
        # batch_size and num_workers values are forwarded only when the subclass
        # explicitly accepts them or accepts keyword arguments.
        #
        # Args:
        #     train_dataset: Optional training dataset or nested dataset tree.
        #     val_dataset: Optional validation dataset or nested dataset tree.
        #     test_dataset: Optional test dataset or nested dataset tree.
        #     predict_dataset: Optional prediction dataset or nested dataset tree.
        #     batch_size: Batch size shared by generated loaders. Default: ``1``.
        #     num_workers: Worker count shared by generated loaders. Default:
        #         ``0`` for main-process loading.
        #     **datamodule_kwargs: Additional arguments for the requested
        #         subclass constructor.
        #
        # Return:
        #     An instance of cls whose supplied stages delegate to a private
        #     DatasetDataModule provider while retaining cls lifecycle behavior.
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
        # Reconstructs the subclass from saved or externally supplied
        # hyperparameters, then restores its component state when present.
        # Explicit keyword arguments override checkpoint and hparams-file values.
        #
        # Args:
        #     checkpoint_path: Local, remote, or seekable schema-v2 checkpoint.
        #     map_location: torch.load device remapping policy. Default: ``None``.
        #     hparams_file: Optional CSV or YAML constructor configuration.
        #     weights_only: Optional torch.load safety policy.
        #     **kwargs: Constructor values overriding every persisted source.
        return _DataModuleCheckpointLoader.load(
            cls,
            checkpoint_path,
            map_location,
            hparams_file,
            weights_only,
            kwargs
        )

    def state_dict(self) -> StateDict:
        # Serializes DataModule-owned continuation state. The default contributes
        # no state; stateful clients override this together with load_state_dict.
        state: StateDict = {}
        return state

    def load_state_dict(self, state_dict: StateDict) -> None:
        # Restores state emitted by state_dict after checkpoint construction.
        del state_dict

    def on_exception(self, exception: BaseException) -> None:
        # Receives the causal stage exception before ordinary teardown so a
        # client can release or mark external data resources while context is
        # still available.
        del exception

    def _configured_loader_stages(self) -> list[str]:
        # Detects stage support from instance-bound factories and concrete class
        # overrides relative to DataHooks; used only for representation.
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
        # Reports the concrete class and supported loader stages without
        # constructing a dataloader or triggering client side effects.
        configured_loaders: list[str] = self._configured_loader_stages()
        class_name: str = type(self).__name__
        loaders_repr: str = ", ".join(
            f"{name}=True" for name in configured_loaders
        )
        return f"{class_name}({loaders_repr})"


class DatasetDataModule(DataModule):
    # Concrete DataModule for already constructed datasets. It is appropriate
    # when client code needs ordinary DataLoaders without custom preparation,
    # setup, transfer, or state behavior; nested datasets produce an equivalent
    # nested collection of loaders.
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
        # Validates shared loader settings and records optional stage datasets.
        # A missing dataset leaves that stage unsupported and its factory fails
        # with a stage-specific configuration message when requested.
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
        # Builds the training loader tree, shuffling map-style leaves while
        # allowing iterable datasets to control their own order.
        if self._train_dataset is None:
            raise MisconfigurationError(
                "train_dataloader() is unavailable because no train_dataset was supplied."
            )
        return self._build_dataloaders(self._train_dataset, shuffle=True)

    @override
    def val_dataloader(self) -> DataLoaderSource:
        # Builds the validation loader tree without shuffling so sample order is
        # stable across validation passes and resumed runs.
        if self._val_dataset is None:
            raise MisconfigurationError(
                "val_dataloader() is unavailable because no val_dataset was supplied."
            )
        return self._build_dataloaders(self._val_dataset, shuffle=False)

    @override
    def test_dataloader(self) -> DataLoaderSource:
        # Builds the test loader tree without shuffling.
        if self._test_dataset is None:
            raise MisconfigurationError(
                "test_dataloader() is unavailable because no test_dataset was supplied."
            )
        return self._build_dataloaders(self._test_dataset, shuffle=False)

    @override
    def predict_dataloader(self) -> DataLoaderSource:
        # Builds the prediction loader tree without shuffling so returned
        # predictions remain aligned with dataset order before distributed
        # partition reconstruction.
        if self._predict_dataset is None:
            raise MisconfigurationError(
                "predict_dataloader() is unavailable because no predict_dataset was supplied."
            )
        return self._build_dataloaders(self._predict_dataset, shuffle=False)

    @override
    def _configured_loader_stages(self) -> list[str]:
        # Reports supported stages from dataset presence because this concrete
        # class is configured through constructor values rather than overrides.
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
        # Flattens nested dataset topology, builds one DataLoader per leaf, and
        # reconstructs the original collection. pin_memory remains enabled for
        # efficient CUDA transfer while non-CUDA backends preserve correctness.
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
