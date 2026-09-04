import random
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import torch

from syntheticmind.callbacks.callback import Callback
from syntheticmind.callbacks.model_checkpoint import ModelCheckpoint
from syntheticmind.core.datamodule import DataModule
from syntheticmind.core.hyperparameters import HyperparametersMixin
from syntheticmind.core.optimizer import _SchedulerInstance
from syntheticmind.plugins.precision.precision import Precision
from syntheticmind.plugins.checkpoint_io import (
    CheckpointPath,
    CheckpointStorageOptions
)
from syntheticmind.strategies.strategy import Strategy
from syntheticmind.trainer.connectors.callback_connector import _CallbackConnector
from syntheticmind.utilities.exceptions import MisconfigurationError
from syntheticmind.storage.filesystem import (
    basename,
    list_directory,
    modified_time,
    normalize_path
)
from syntheticmind.utilities.types import (
    CheckpointDict,
    CheckpointValue,
    DataLoaderState,
    HyperparameterDict,
    StateDict,
    TrainerStage
)

__all__: list[str] = []


class _CheckpointConnector:
    def __init__(
        self,
        strategy: Strategy,
        callback_connector: _CallbackConnector,
        fast_dev_run: bool
    ) -> None:
        self._strategy: Strategy = strategy
        self._callback_connector: _CallbackConnector = callback_connector
        self._fast_dev_run: bool = fast_dev_run
        self._path: str | Path | None = None
        self._user_managed: bool = False
        self.train_dataloader_states: list[DataLoaderState] = []
        self.train_dataloader_state_active: bool = False
        self.train_dataloader_batches_completed: int = 0
        self.train_total_batch_idx: int = -1

    def select_path(
        self,
        checkpoint_path: str | Path | None,
        stage: TrainerStage,
        model_provided: bool,
        path_provided: bool
    ) -> str | Path | None:
        selected_path: str | Path | None = checkpoint_path
        if self._user_managed:
            if not path_provided:
                selected_path: str | Path | None = self._path
            elif selected_path != self._path:
                if self._strategy.is_global_zero:
                    warnings.warn(
                        f"trainer.ckpt_path={self._path!r} is overridden by "
                        f"Trainer.{stage}(ckpt_path={selected_path!r}).",
                        UserWarning,
                        stacklevel=3
                    )
                self._user_managed: bool = False
        if selected_path is None and not model_provided and stage != "fit":
            selected_path: str | Path | None = "best"
            if self._strategy.is_global_zero:
                warnings.warn(
                    f"Trainer.{stage}(ckpt_path=None) was called without a model; "
                    "the best checkpoint from the previous fit will be loaded.",
                    UserWarning,
                    stacklevel=3
                )
        if selected_path is None:
            self._path: str | Path | None = None
            return None
        if isinstance(selected_path, str) and selected_path == "best":
            return self._select_best_path(stage)
        if isinstance(selected_path, str) and selected_path == "last":
            return self._select_last_path(stage)
        if isinstance(selected_path, str) and selected_path in {"hpc", "registry"}:
            raise MisconfigurationError(
                f"ckpt_path={selected_path!r} is unavailable without its integration."
            )
        normalized_path: str = normalize_path(selected_path)
        if not self._strategy.checkpoint.is_checkpoint_path(normalized_path):
            raise FileNotFoundError(f"Checkpoint not found: {normalized_path}")
        self._path: str | Path | None = normalized_path
        return normalized_path

    def reset_training_continuation(self) -> None:
        self.train_dataloader_states: list[DataLoaderState] = []
        self.train_dataloader_state_active: bool = False
        self.train_dataloader_batches_completed: int = 0
        self.train_total_batch_idx: int = -1

    def build(
        self,
        model: torch.nn.Module,
        scheduler: _SchedulerInstance | None,
        schedulers: list[_SchedulerInstance] | None,
        epoch: int,
        global_step: int,
        datamodule: DataModule | None,
        completed_batches: int,
        total_batch_idx: int,
        epoch_rng_state: CheckpointDict | None,
        combined_loader_states: list[DataLoaderState],
        precision_plugin: Precision,
        weights_only: bool,
        model_state_dict: CheckpointValue,
        optimizer_states: list[CheckpointValue]
    ) -> CheckpointDict:
        checkpoint: CheckpointDict = {
            "schema_version": 1,
            "epoch": epoch,
            "global_step": global_step,
            "state_dict": model_state_dict
        }
        if isinstance(model, HyperparametersMixin):
            self._store_hyperparameters(checkpoint, model)
        if not weights_only:
            checkpoint["rng_state"]: CheckpointValue = self.collect_rng_state()
            checkpoint["optimizer_states"]: CheckpointValue = optimizer_states
        if completed_batches > 0:
            checkpoint["completed_batches"]: CheckpointValue = completed_batches
        if total_batch_idx >= 0:
            checkpoint["total_batch_idx"]: CheckpointValue = total_batch_idx
        if epoch_rng_state is not None:
            checkpoint["epoch_rng_state"]: CheckpointValue = epoch_rng_state
        if combined_loader_states:
            checkpoint["loops"]: CheckpointValue = {
                "fit_loop": {
                    "state_dict": {
                        "combined_loader": combined_loader_states
                    }
                }
            }
        if not weights_only:
            scheduler_list: list[_SchedulerInstance] = (
                list(schedulers)
                if schedulers is not None
                else [scheduler]
                if scheduler is not None
                else []
            )
            checkpoint["lr_schedulers"]: CheckpointValue = [
                current_scheduler.state_dict()
                for current_scheduler in scheduler_list
            ]
            callback_states: dict[str, StateDict] = {}
            callback: Callback
            for callback in self._callback_connector.callbacks:
                callback_state: StateDict = callback.state_dict()
                if callback_state:
                    callback_states[callback.state_key]: StateDict = callback_state
            checkpoint["callbacks"]: CheckpointValue = cast(
                CheckpointValue,
                callback_states
            )
        if datamodule is not None:
            self._store_hyperparameters(checkpoint, datamodule)
            datamodule_state: StateDict = datamodule.state_dict()
            if datamodule_state:
                checkpoint[type(datamodule).__qualname__]: CheckpointValue = cast(
                    CheckpointValue,
                    datamodule_state
                )
        if not weights_only:
            precision_state: StateDict = precision_plugin.state_dict()
            if precision_state:
                checkpoint[type(precision_plugin).__qualname__]: CheckpointValue = cast(
                    CheckpointValue,
                    precision_state
                )
        checkpoint["metadata"]: CheckpointValue = {
            "torch_version": str(torch.__version__),
            "strategy": type(self._strategy).__name__,
            "world_size": self._strategy.world_size
        }
        return checkpoint

    def load(
        self,
        checkpoint_path: CheckpointPath,
        weights_only: bool | None
    ) -> CheckpointDict:
        checkpoint: CheckpointDict = self._strategy.checkpoint.load(
            checkpoint_path,
            weights_only
        )
        self._validate_schema(checkpoint)
        return checkpoint

    def load_metadata(
        self,
        checkpoint_path: CheckpointPath,
        weights_only: bool | None
    ) -> CheckpointDict:
        checkpoint: CheckpointDict = self._strategy.checkpoint.load_metadata(
            checkpoint_path,
            weights_only
        )
        self._validate_schema(checkpoint)
        return checkpoint

    def restore_strategy_state(
        self,
        checkpoint: CheckpointDict,
        strict: bool,
        restore_optimizers: bool
    ) -> None:
        self._strategy.checkpoint.restore(
            checkpoint,
            strict,
            restore_optimizers
        )

    def save(
        self,
        checkpoint: CheckpointDict,
        filepath: CheckpointPath,
        storage_options: CheckpointStorageOptions,
        synchronize: bool
    ) -> None:
        self._strategy.checkpoint.save(
            checkpoint,
            filepath,
            storage_options,
            self._strategy.global_rank
        )
        if synchronize:
            self._strategy.barrier()

    def remove(self, filepath: CheckpointPath) -> None:
        self._strategy.checkpoint.remove(filepath, self._strategy.global_rank)

    def is_checkpoint_path(self, checkpoint_path: CheckpointPath) -> bool:
        return self._strategy.checkpoint.is_checkpoint_path(checkpoint_path)

    def model_state_dict(self) -> CheckpointValue:
        return cast(CheckpointValue, self._strategy.checkpoint.model_state_dict())

    def optimizer_state(
        self,
        optimizer: torch.optim.Optimizer
    ) -> CheckpointValue:
        return cast(
            CheckpointValue,
            self._strategy.checkpoint.optimizer_state(optimizer)
        )

    def connect_optimizers(
        self,
        optimizers: list[torch.optim.Optimizer]
    ) -> None:
        self._strategy.checkpoint.connect_optimizers(optimizers)

    def teardown(self) -> None:
        self._strategy.checkpoint.teardown()

    def collect_rng_state(self) -> CheckpointDict:
        numpy_rng_state: tuple[str, np.ndarray, int, int, float] = cast(
            tuple[str, np.ndarray, int, int, float],
            np.random.get_state()
        )
        states: CheckpointDict = {
            "python": random.getstate(),
            "numpy": {
                "bit_generator": numpy_rng_state[0],
                "state": torch.from_numpy(numpy_rng_state[1].copy()),
                "position": numpy_rng_state[2],
                "has_gauss": numpy_rng_state[3],
                "cached_gaussian": numpy_rng_state[4]
            },
            "torch": torch.random.get_rng_state()
        }
        if torch.cuda.is_available():
            states["cuda"]: CheckpointValue = torch.cuda.get_rng_state_all()
        return states

    def restore_training_state(
        self,
        checkpoint: CheckpointDict,
        scheduler: _SchedulerInstance | None,
        schedulers: list[_SchedulerInstance] | None,
        datamodule: DataModule | None,
        precision_plugin: Precision,
        restore_datamodule_state: bool
    ) -> tuple[int, int, int, int, CheckpointDict | None]:
        scheduler_states: CheckpointValue | None = checkpoint.get("lr_schedulers")
        scheduler_list: list[_SchedulerInstance] = (
            list(schedulers)
            if schedulers is not None
            else [scheduler]
            if scheduler is not None
            else []
        )
        if isinstance(scheduler_states, list):
            scheduler_pair: tuple[_SchedulerInstance, CheckpointValue]
            for scheduler_pair in zip(
                scheduler_list,
                scheduler_states,
                strict=False
            ):
                current_scheduler: _SchedulerInstance = scheduler_pair[0]
                scheduler_state: CheckpointValue = scheduler_pair[1]
                if not isinstance(scheduler_state, dict):
                    raise TypeError("Scheduler checkpoint state must be a mapping.")
                current_scheduler.load_state_dict(cast(StateDict, scheduler_state))
        rng_state: CheckpointValue | None = checkpoint.get("rng_state")
        if isinstance(rng_state, dict):
            self.restore_rng_state(cast(CheckpointDict, rng_state))
        callback_states_value: CheckpointValue | None = checkpoint.get("callbacks")
        callback_states: dict[str, StateDict] = (
            cast(dict[str, StateDict], callback_states_value)
            if isinstance(callback_states_value, dict)
            else {}
        )
        callback: Callback
        for callback in self._callback_connector.callbacks:
            callback_state: StateDict | None = callback_states.get(
                callback.state_key
            )
            if callback_state is not None:
                callback.load_state_dict(callback_state)
        if datamodule is not None and restore_datamodule_state:
            self.load_datamodule_state(checkpoint, datamodule)
        precision_state: CheckpointValue | None = checkpoint.get(
            type(precision_plugin).__qualname__
        )
        if isinstance(precision_state, dict):
            precision_plugin.load_state_dict(cast(StateDict, precision_state))
        epoch: int = self.integer_value(checkpoint.get("epoch"), 0)
        global_step: int = self.integer_value(checkpoint.get("global_step"), 0)
        completed_batches: int = self.integer_value(
            checkpoint.get("completed_batches"),
            0
        )
        total_batch_idx: int = self.integer_value(
            checkpoint.get("total_batch_idx"),
            -1
        )
        epoch_rng_state_value: CheckpointValue | None = checkpoint.get(
            "epoch_rng_state"
        )
        epoch_rng_state: CheckpointDict | None = (
            cast(CheckpointDict, epoch_rng_state_value)
            if isinstance(epoch_rng_state_value, dict)
            else None
        )
        return (
            epoch,
            global_step,
            completed_batches,
            total_batch_idx,
            epoch_rng_state
        )

    def restore_rng_state(self, rng_state: CheckpointDict) -> None:
        if "python" in rng_state:
            random.setstate(cast(tuple, rng_state["python"]))
        numpy_state: CheckpointValue | None = rng_state.get("numpy")
        if isinstance(numpy_state, dict):
            bit_generator: CheckpointValue | None = numpy_state.get("bit_generator")
            state_tensor: CheckpointValue | None = numpy_state.get("state")
            position: CheckpointValue | None = numpy_state.get("position")
            has_gauss: CheckpointValue | None = numpy_state.get("has_gauss")
            cached_gaussian: CheckpointValue | None = numpy_state.get(
                "cached_gaussian"
            )
            if (
                isinstance(bit_generator, str)
                and isinstance(state_tensor, torch.Tensor)
                and isinstance(position, int)
                and isinstance(has_gauss, int)
                and isinstance(cached_gaussian, float | int)
            ):
                np.random.set_state(
                    (
                        bit_generator,
                        state_tensor.cpu().numpy().astype(np.uint32, copy=False),
                        position,
                        has_gauss,
                        float(cached_gaussian)
                    )
                )
        torch_state: CheckpointValue | None = rng_state.get("torch")
        if isinstance(torch_state, torch.Tensor):
            torch.random.set_rng_state(torch_state.cpu())
        cuda_state: CheckpointValue | None = rng_state.get("cuda")
        if torch.cuda.is_available() and isinstance(cuda_state, list):
            cuda_states: list[torch.Tensor] = []
            state: CheckpointValue
            for state in cuda_state:
                if not isinstance(state, torch.Tensor):
                    raise TypeError("CUDA RNG checkpoint state must contain tensors.")
                cuda_states.append(state.cpu())
            torch.cuda.set_rng_state_all(cuda_states)

    def combined_loader_states(
        self,
        checkpoint: CheckpointDict
    ) -> list[DataLoaderState]:
        loops_value: CheckpointValue | None = checkpoint.get("loops")
        if loops_value is None:
            return []
        if not isinstance(loops_value, dict):
            raise TypeError("Checkpoint loops state must be a mapping.")
        fit_loop_value: CheckpointValue | None = loops_value.get("fit_loop")
        if fit_loop_value is None:
            return []
        if not isinstance(fit_loop_value, dict):
            raise TypeError("Checkpoint fit-loop state must be a mapping.")
        state_dict_value: CheckpointValue | None = fit_loop_value.get("state_dict")
        if state_dict_value is None:
            return []
        if not isinstance(state_dict_value, dict):
            raise TypeError("Checkpoint fit-loop state_dict must be a mapping.")
        combined_loader_value: CheckpointValue | None = state_dict_value.get(
            "combined_loader"
        )
        if combined_loader_value is None:
            return []
        if not isinstance(combined_loader_value, list):
            raise TypeError("Checkpoint combined-loader state must be a list.")
        states: list[DataLoaderState] = []
        state: CheckpointValue
        for state in combined_loader_value:
            if not isinstance(state, dict):
                raise TypeError("Checkpoint loader state must be a mapping.")
            states.append(cast(DataLoaderState, state))
        return states

    def load_datamodule_state(
        self,
        checkpoint: CheckpointDict,
        datamodule: DataModule
    ) -> None:
        datamodule_state: CheckpointValue | None = checkpoint.get(
            type(datamodule).__qualname__
        )
        if isinstance(datamodule_state, dict):
            datamodule.load_state_dict(cast(StateDict, datamodule_state))

    def integer_value(self, value: CheckpointValue | None, default: int) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float | str):
            try:
                return int(value)
            except (OverflowError, ValueError):
                return default
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return int(value.item())
        return default

    @property
    def path(self) -> str | Path | None:
        return self._path

    @path.setter
    def path(self, checkpoint_path: str | Path | None) -> None:
        if checkpoint_path is not None and not isinstance(checkpoint_path, str | Path):
            raise TypeError("ckpt_path must be a string, Path, or None.")
        self._path: str | Path | None = checkpoint_path
        self._user_managed: bool = checkpoint_path is not None

    @property
    def user_managed(self) -> bool:
        return self._user_managed

    def _select_best_path(self, stage: TrainerStage) -> str:
        checkpoint_callbacks: list[ModelCheckpoint] = [
            callback
            for callback in self._callback_connector.callbacks
            if isinstance(callback, ModelCheckpoint)
        ]
        if len(checkpoint_callbacks) > 1 and self._strategy.is_global_zero:
            warnings.warn(
                f"Trainer.{stage}(ckpt_path='best') received multiple ModelCheckpoint "
                "callbacks; the first callback owns best-checkpoint selection.",
                UserWarning,
                stacklevel=3
            )
        if checkpoint_callbacks:
            best_model_path: str = checkpoint_callbacks[0].best_model_path
            if best_model_path:
                self._path: str | Path | None = best_model_path
                return best_model_path
        if self._fast_dev_run:
            raise MisconfigurationError(
                f"Trainer.{stage}(ckpt_path='best') is unavailable with fast_dev_run."
            )
        raise MisconfigurationError(
            f"Trainer.{stage}(ckpt_path='best') requires a saved best_model_path."
        )

    def _select_last_path(self, stage: TrainerStage) -> str | None:
        candidates: list[str] = []
        callback: Callback
        for callback in self._callback_connector.callbacks:
            if not isinstance(callback, ModelCheckpoint):
                continue
            if callback.last_model_path:
                candidates.append(callback.last_model_path)
            if callback.dirpath:
                candidate: str
                for candidate in list_directory(callback.dirpath):
                    candidate_name: str = basename(candidate)
                    if (
                        candidate_name.startswith("last")
                        and candidate_name.endswith(".ckpt")
                        and self._strategy.checkpoint.is_checkpoint_path(candidate)
                    ):
                        candidates.append(candidate)
        unique_candidates: set[str] = set(candidates)
        if not unique_candidates:
            self._path: str | Path | None = None
            if self._strategy.is_global_zero:
                warnings.warn(
                    f"Trainer.{stage}(ckpt_path='last') found no last checkpoint.",
                    UserWarning,
                    stacklevel=3
                )
            return None
        selected_path: str = max(unique_candidates, key=modified_time)
        self._path: str | Path | None = selected_path
        return selected_path

    def _store_hyperparameters(
        self,
        checkpoint: CheckpointDict,
        source: HyperparametersMixin
    ) -> None:
        source._clean_hparams()
        if not source.hparams:
            return
        checkpoint[source._checkpoint_hyperparameters_name]: CheckpointValue = (
            source._hparams_name
        )
        stored_hyperparameters: HyperparameterDict = {
            key: value for key, value in source.hparams.items()
        }
        checkpoint[source._checkpoint_hyperparameters_key]: CheckpointValue = cast(
            CheckpointValue,
            stored_hyperparameters
        )

    def _validate_schema(self, checkpoint: CheckpointDict) -> None:
        schema_version: CheckpointValue | None = checkpoint.get("schema_version")
        if schema_version != 1:
            raise ValueError(
                f"Unsupported SyntheticMind checkpoint schema {schema_version!r}."
            )
