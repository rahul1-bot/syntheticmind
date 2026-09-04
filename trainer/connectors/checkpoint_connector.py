import warnings
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import cast

import torch
from pydantic import ValidationError

from syntheticmind.callbacks.callback import Callback
from syntheticmind.callbacks.checkpointing import ModelCheckpoint
from syntheticmind.core.datamodule import DataModule
from syntheticmind.core.hyperparameters import HyperparametersMixin
from syntheticmind.core.metric import LoggedMetric, _LoggedMetricCheckpoint
from syntheticmind.core.optimizer import _SchedulerInstance
from syntheticmind.plugins.precision.precision import Precision
from syntheticmind.randomness import _Randomness, _RandomStateSnapshot
from syntheticmind.plugins.checkpoint_io import (
    CheckpointPath,
    CheckpointStorageOptions
)
from syntheticmind.strategies.strategy import Strategy
from syntheticmind.trainer.connectors.callback_connector import _CallbackConnector
from syntheticmind.trainer.connectors.logger_connector.metric_accumulator import (
    MetricAccumulator,
    _MetricAccumulatorCheckpoint
)
from syntheticmind.trainer.configuration import _TrainerConfigurationSnapshot
from syntheticmind.exceptions import MisconfigurationError
from syntheticmind.storage.filesystem import StoragePath
from syntheticmind.types import (
    CheckpointDict,
    CheckpointValue,
    DataLoaderState,
    HyperparameterDict,
    OptimizerState,
    StateDict,
    TrainerStage
)

__all__: list[str] = []


class _CheckpointConnector:
    def __init__(
        self,
        strategy: Strategy,
        callback_connector: _CallbackConnector,
        fast_dev_run: bool,
        randomness: _Randomness
    ) -> None:
        self._strategy: Strategy = strategy
        self._callback_connector: _CallbackConnector = callback_connector
        self._fast_dev_run: bool = fast_dev_run
        self._randomness: _Randomness = randomness
        self._path: str | Path | None = None
        self._user_managed: bool = False
        self.train_dataloader_states: list[DataLoaderState] = []
        self.train_dataloader_state_active: bool = False
        self.train_dataloader_batches_completed: int = 0
        self.train_total_batch_idx: int = -1
        self.train_epoch_rng_state: CheckpointDict | None = None
        self.train_epoch_metric_states_by_rank: list[CheckpointDict] = []
        self.train_optimizer_states: list[OptimizerState] = []
        self.train_scheduler_states: list[CheckpointValue] = []

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
        normalized_path: str = StoragePath(selected_path).normalized
        if not self._strategy.checkpoint.is_checkpoint_path(normalized_path):
            raise FileNotFoundError(f"Checkpoint not found: {normalized_path}")
        self._path: str | Path | None = normalized_path
        return normalized_path

    def reset_training_continuation(self) -> None:
        self.train_dataloader_states: list[DataLoaderState] = []
        self.train_dataloader_state_active: bool = False
        self.train_dataloader_batches_completed: int = 0
        self.train_total_batch_idx: int = -1
        self.train_epoch_rng_state: CheckpointDict | None = None
        self.train_epoch_metric_states_by_rank: list[CheckpointDict] = []
        self.train_optimizer_states: list[OptimizerState] = []
        self.train_scheduler_states: list[CheckpointValue] = []

    def build(
        self,
        model: torch.nn.Module,
        epoch: int,
        global_step: int,
        datamodule: DataModule | None,
        completed_batches: int,
        total_batch_idx: int,
        epoch_rng_state: CheckpointDict | None,
        combined_loader_states: list[DataLoaderState],
        epoch_metric_states: list[CheckpointDict],
        precision_plugin: Precision,
        weights_only: bool,
        model_state_dict: CheckpointValue,
        optimizer_states: list[OptimizerState],
        scheduler_states: list[CheckpointValue],
        trainer_configuration: CheckpointValue
    ) -> CheckpointDict:
        checkpoint: CheckpointDict = {
            "schema_version": 2,
            "epoch": epoch,
            "global_step": global_step,
            "state_dict": model_state_dict,
            "trainer_configuration": trainer_configuration
        }
        if isinstance(model, HyperparametersMixin):
            self._store_hyperparameters(checkpoint, model)
        if not weights_only:
            checkpoint["rng_state"]: CheckpointValue = self.capture_random_state()
            checkpoint["optimizer_states"]: CheckpointValue = cast(
                CheckpointValue,
                optimizer_states
            )
        if not weights_only and completed_batches > 0:
            checkpoint["completed_batches"]: CheckpointValue = completed_batches
        if not weights_only and total_batch_idx >= 0:
            checkpoint["total_batch_idx"]: CheckpointValue = total_batch_idx
        if not weights_only and epoch_rng_state is not None:
            checkpoint["epoch_rng_state"]: CheckpointValue = epoch_rng_state
        fit_loop_state: CheckpointDict = {}
        if not weights_only and combined_loader_states:
            fit_loop_state["combined_loader"]: CheckpointValue = (
                combined_loader_states
            )
        if not weights_only and epoch_metric_states:
            fit_loop_state["epoch_metrics_by_rank"]: CheckpointValue = (
                epoch_metric_states
            )
        if fit_loop_state:
            checkpoint["loops"]: CheckpointValue = {
                "fit_loop": {
                    "state_dict": fit_loop_state
                }
            }
        if not weights_only:
            checkpoint["lr_schedulers"]: CheckpointValue = scheduler_states
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
        if not weights_only and datamodule is not None:
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
            "checkpoint_type": "weights" if weights_only else "training",
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

    def validate_fit_configuration(
        self,
        checkpoint: CheckpointDict,
        current: _TrainerConfigurationSnapshot
    ) -> None:
        checkpoint_type: str = self._checkpoint_type(checkpoint)
        if checkpoint_type == "weights":
            raise MisconfigurationError(
                "A weights-only model artifact cannot continue a fit trajectory. "
                "Load it as model weights or provide a training checkpoint."
            )
        saved: _TrainerConfigurationSnapshot = self.configuration_snapshot(
            checkpoint
        )
        mismatches: tuple[str, ...] = saved.resume_mismatches(current)
        if mismatches:
            raise MisconfigurationError(
                "Checkpoint training configuration is incompatible with the "
                f"current Trainer for fields {list(mismatches)}."
            )

    def configuration_snapshot(
        self,
        checkpoint: CheckpointDict
    ) -> _TrainerConfigurationSnapshot:
        configuration: CheckpointValue | None = checkpoint.get(
            "trainer_configuration"
        )
        if not isinstance(configuration, dict):
            raise TypeError(
                "Checkpoint `trainer_configuration` must be a mapping."
            )
        try:
            return _TrainerConfigurationSnapshot.model_validate(
                configuration,
                strict=True
            )
        except ValidationError as exception:
            raise ValueError(
                "Checkpoint Trainer configuration is invalid."
            ) from exception

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
    ) -> OptimizerState:
        return self._strategy.checkpoint.optimizer_state(optimizer)

    def connect_optimizers(
        self,
        optimizers: list[torch.optim.Optimizer]
    ) -> None:
        self._strategy.checkpoint.connect_optimizers(optimizers)

    def teardown(self) -> None:
        self._strategy.checkpoint.teardown()

    def capture_random_state(self) -> CheckpointDict:
        return cast(
            CheckpointDict,
            self._randomness.capture().model_dump(mode="python")
        )

    def restore_random_state(self, state: CheckpointDict) -> None:
        snapshot: _RandomStateSnapshot = self._parse_random_state(state)
        self._randomness.restore(snapshot)

    def restore_training_state(
        self,
        checkpoint: CheckpointDict,
        scheduler: _SchedulerInstance | None,
        schedulers: list[_SchedulerInstance] | None,
        datamodule: DataModule | None,
        precision_plugin: Precision,
        restore_datamodule_state: bool,
        restore_schedulers: bool
    ) -> tuple[int, int, int, int, CheckpointDict | None]:
        scheduler_list: list[_SchedulerInstance] = (
            list(schedulers)
            if schedulers is not None
            else [scheduler]
            if scheduler is not None
            else []
        )
        if restore_schedulers:
            self._restore_scheduler_states(checkpoint, scheduler_list)
        rng_state: CheckpointValue | None = checkpoint.get("rng_state")
        if rng_state is not None:
            if not isinstance(rng_state, dict):
                raise TypeError("Checkpoint `rng_state` must be a mapping.")
            self.restore_random_state(cast(CheckpointDict, rng_state))
        self._restore_callback_states(checkpoint)
        if datamodule is not None and restore_datamodule_state:
            self.load_datamodule_state(checkpoint, datamodule)
        self._restore_precision_state(checkpoint, precision_plugin)
        return self._training_progress(checkpoint)

    def _restore_scheduler_states(
        self,
        checkpoint: CheckpointDict,
        schedulers: list[_SchedulerInstance]
    ) -> None:
        scheduler_states: CheckpointValue | None = checkpoint.get("lr_schedulers")
        if scheduler_states is None:
            if schedulers:
                raise KeyError(
                    "Checkpoint is missing learning-rate scheduler state required "
                    "for exact fit continuation."
                )
            return
        if not isinstance(scheduler_states, list):
            raise TypeError("Checkpoint `lr_schedulers` must be a list.")
        if len(scheduler_states) != len(schedulers):
            raise MisconfigurationError(
                "Checkpoint scheduler count does not match the configured "
                f"scheduler count: checkpoint={len(scheduler_states)}, "
                f"configured={len(schedulers)}."
            )
        scheduler_pair: tuple[_SchedulerInstance, CheckpointValue]
        for scheduler_pair in zip(schedulers, scheduler_states, strict=True):
            current_scheduler: _SchedulerInstance = scheduler_pair[0]
            scheduler_state: CheckpointValue = scheduler_pair[1]
            if not isinstance(scheduler_state, dict):
                raise TypeError("Each scheduler checkpoint state must be a mapping.")
            current_scheduler.load_state_dict(cast(StateDict, scheduler_state))

    def _restore_callback_states(self, checkpoint: CheckpointDict) -> None:
        callback_states_value: CheckpointValue | None = checkpoint.get("callbacks")
        if callback_states_value is None:
            return
        if not isinstance(callback_states_value, dict):
            raise TypeError("Checkpoint `callbacks` state must be a mapping.")
        callback_states: dict[str, CheckpointValue] = callback_states_value
        current_state_keys: set[str] = {
            callback.state_key for callback in self._callback_connector.callbacks
        }
        missing_callbacks: set[str] = set(callback_states) - current_state_keys
        if missing_callbacks and self._strategy.is_global_zero:
            warnings.warn(
                "Checkpoint state exists for callbacks absent from the current "
                f"Trainer: {sorted(missing_callbacks)}. Add those callbacks to "
                "restore their training state.",
                UserWarning,
                stacklevel=3
            )
        callback: Callback
        for callback in self._callback_connector.callbacks:
            callback_state: CheckpointValue | None = callback_states.get(
                callback.state_key
            )
            if callback_state is None:
                continue
            if not isinstance(callback_state, dict):
                raise TypeError(
                    f"Checkpoint state for callback {callback.state_key!r} must "
                    "be a mapping."
                )
            isolated_state: StateDict = deepcopy(cast(StateDict, callback_state))
            callback.load_state_dict(isolated_state)

    def _restore_precision_state(
        self,
        checkpoint: CheckpointDict,
        precision_plugin: Precision
    ) -> None:
        precision_state: CheckpointValue | None = checkpoint.get(
            type(precision_plugin).__qualname__
        )
        if precision_state is None:
            return
        if not isinstance(precision_state, dict):
            raise TypeError(
                f"Checkpoint state for precision plugin "
                f"{type(precision_plugin).__qualname__!r} must be a mapping."
            )
        precision_plugin.load_state_dict(cast(StateDict, precision_state))

    def _training_progress(
        self,
        checkpoint: CheckpointDict
    ) -> tuple[int, int, int, int, CheckpointDict | None]:
        epoch: int = self.integer_value(checkpoint.get("epoch"), "epoch", 0)
        global_step: int = self.integer_value(
            checkpoint.get("global_step"),
            "global_step",
            0
        )
        completed_batches: int = self.integer_value(
            checkpoint.get("completed_batches"),
            "completed_batches",
            0
        )
        total_batch_idx: int = self.integer_value(
            checkpoint.get("total_batch_idx"),
            "total_batch_idx",
            -1
        )
        epoch_rng_state_value: CheckpointValue | None = checkpoint.get(
            "epoch_rng_state"
        )
        if epoch_rng_state_value is not None and not isinstance(
            epoch_rng_state_value,
            dict
        ):
            raise TypeError("Checkpoint `epoch_rng_state` must be a mapping.")
        if isinstance(epoch_rng_state_value, dict):
            self._parse_random_state(cast(CheckpointDict, epoch_rng_state_value))
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

    def _parse_random_state(
        self,
        state: CheckpointDict
    ) -> _RandomStateSnapshot:
        try:
            return _RandomStateSnapshot.model_validate(state, strict=True)
        except ValidationError as exception:
            raise ValueError("Checkpoint random state is invalid.") from exception

    def combined_loader_states(
        self,
        checkpoint: CheckpointDict
    ) -> list[DataLoaderState]:
        state_dict: CheckpointDict = self._fit_loop_state(checkpoint)
        combined_loader_value: CheckpointValue | None = state_dict.get(
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

    def epoch_metric_state(self, checkpoint: CheckpointDict) -> CheckpointDict:
        states: list[CheckpointDict] = self._epoch_metric_states(checkpoint)
        if not states:
            return {}
        if len(states) != self._strategy.world_size:
            raise MisconfigurationError(
                "Partial-epoch metric state requires the checkpoint world size."
            )
        return states[self._strategy.global_rank]

    def _epoch_metric_states(
        self,
        checkpoint: CheckpointDict
    ) -> list[CheckpointDict]:
        state_dict: CheckpointDict = self._fit_loop_state(checkpoint)
        states_value: CheckpointValue | None = state_dict.get(
            "epoch_metrics_by_rank"
        )
        if states_value is None:
            return []
        if not isinstance(states_value, list):
            raise TypeError("Checkpoint epoch metric states must be a list.")
        states: list[CheckpointDict] = []
        state_value: CheckpointValue
        for state_value in states_value:
            if not isinstance(state_value, dict):
                raise TypeError("Checkpoint rank metric state must be a mapping.")
            states.append(cast(CheckpointDict, state_value))
        return states

    def _validate_epoch_metric_states(self, checkpoint: CheckpointDict) -> int:
        states: list[CheckpointDict] = self._epoch_metric_states(checkpoint)
        rank_state: CheckpointDict
        for rank_state in states:
            storage_key: str
            entry_value: CheckpointValue
            for storage_key, entry_value in rank_state.items():
                if not isinstance(storage_key, str) or not storage_key:
                    raise TypeError("Checkpoint epoch metric keys must be strings.")
                if not isinstance(entry_value, dict):
                    raise TypeError("Checkpoint epoch metric entry must be a mapping.")
                self._validate_epoch_metric_entry(storage_key, entry_value)
        return len(states)

    def _validate_epoch_metric_entry(
        self,
        storage_key: str,
        entry: CheckpointDict
    ) -> None:
        expected_fields: set[str] = {
            "identity",
            "accumulator",
            "metric_state",
            "metric_update_count"
        }
        if set(entry) != expected_fields:
            raise ValueError("Checkpoint epoch metric entry fields are invalid.")
        identity_value: CheckpointValue | None = entry.get("identity")
        accumulator_value: CheckpointValue | None = entry.get("accumulator")
        metric_state_value: CheckpointValue | None = entry.get("metric_state")
        update_count_value: CheckpointValue | None = entry.get("metric_update_count")
        if not isinstance(identity_value, dict):
            raise TypeError("Checkpoint epoch metric identity must be a mapping.")
        if not isinstance(accumulator_value, dict):
            raise TypeError("Checkpoint epoch metric accumulator must be a mapping.")
        if not isinstance(metric_state_value, dict):
            raise TypeError("Checkpoint epoch Metric state must be a mapping.")
        identity: _LoggedMetricCheckpoint = LoggedMetric._parse_checkpoint_state(
            cast(StateDict, identity_value)
        )
        if not identity.on_epoch:
            raise ValueError("Checkpoint epoch metric identity must enable on_epoch.")
        accumulator: _MetricAccumulatorCheckpoint = (
            MetricAccumulator._parse_checkpoint_state(
                cast(StateDict, accumulator_value)
            )
        )
        MetricAccumulator._validate_checkpoint_values(accumulator)
        if accumulator.reduction != identity.reduction:
            raise ValueError("Checkpoint epoch metric reductions are inconsistent.")
        if accumulator.is_metric != identity.is_metric:
            raise ValueError("Checkpoint epoch metric kinds are inconsistent.")
        dataloader_key: str = (
            f".{identity.dataloader_idx}"
            if identity.add_dataloader_idx and identity.dataloader_idx is not None
            else ""
        )
        expected_storage_key: str = (
            f"{identity.hook_name}.{identity.name}{dataloader_key}"
        )
        if storage_key != expected_storage_key:
            raise ValueError("Checkpoint epoch metric storage key is inconsistent.")
        if isinstance(update_count_value, bool) or not isinstance(
            update_count_value,
            int
        ):
            raise TypeError("Checkpoint Metric update count must be an integer.")
        if update_count_value < 0:
            raise ValueError("Checkpoint Metric update count must be non-negative.")
        LoggedMetric._validate_metric_checkpoint_state(
            cast(StateDict, metric_state_value)
        )
        if identity.is_metric:
            if not identity.metric_attribute:
                raise ValueError("Checkpoint TorchMetric identity requires an attribute.")
        elif metric_state_value or update_count_value != 0:
            raise ValueError("A scalar epoch metric cannot contain Metric state.")

    def _fit_loop_state(self, checkpoint: CheckpointDict) -> CheckpointDict:
        loops_value: CheckpointValue | None = checkpoint.get("loops")
        if loops_value is None:
            return {}
        if not isinstance(loops_value, dict):
            raise TypeError("Checkpoint loops state must be a mapping.")
        fit_loop_value: CheckpointValue | None = loops_value.get("fit_loop")
        if fit_loop_value is None:
            return {}
        if not isinstance(fit_loop_value, dict):
            raise TypeError("Checkpoint fit-loop state must be a mapping.")
        state_dict_value: CheckpointValue | None = fit_loop_value.get("state_dict")
        if state_dict_value is None:
            return {}
        if not isinstance(state_dict_value, dict):
            raise TypeError("Checkpoint fit-loop state_dict must be a mapping.")
        return cast(CheckpointDict, state_dict_value)

    def load_datamodule_state(
        self,
        checkpoint: CheckpointDict,
        datamodule: DataModule
    ) -> None:
        state_key: str = type(datamodule).__qualname__
        datamodule_state: CheckpointValue | None = checkpoint.get(state_key)
        if datamodule_state is None:
            return
        if not isinstance(datamodule_state, dict):
            raise TypeError(
                f"Checkpoint state for data module {state_key!r} must be a mapping."
            )
        datamodule.load_state_dict(cast(StateDict, datamodule_state))

    def integer_value(
        self,
        value: CheckpointValue | None,
        field_name: str,
        default: int
    ) -> int:
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"Checkpoint `{field_name}` must be an integer, got "
                f"{type(value).__qualname__}."
            )
        return value

    @property
    def path(self) -> str | Path | None:
        return self._path

    @path.setter
    def path(self, checkpoint_path: str | Path | None) -> None:
        if checkpoint_path is not None and not isinstance(checkpoint_path, str | Path):
            raise TypeError("ckpt_path must be a string, Path, or None.")
        self._path: str | Path | None = checkpoint_path
        self._user_managed: bool = checkpoint_path is not None

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
                for candidate in StoragePath(callback.dirpath).list_directory():
                    candidate_name: str = StoragePath(candidate).name
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
        candidate_iterator: Iterator[str] = iter(unique_candidates)
        selected_path: str = next(candidate_iterator)
        selected_modified_time: float = StoragePath(selected_path).modified_time
        for candidate in candidate_iterator:
            candidate_modified_time: float = StoragePath(candidate).modified_time
            if candidate_modified_time > selected_modified_time:
                selected_path: str = candidate
                selected_modified_time: float = candidate_modified_time
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
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise TypeError("Checkpoint `schema_version` must be an integer.")
        if schema_version != 2:
            raise ValueError(
                f"Unsupported SyntheticMind checkpoint schema {schema_version!r}."
            )
        state_dict: CheckpointValue | None = checkpoint.get("state_dict")
        if state_dict is not None and not isinstance(state_dict, dict):
            raise TypeError("Checkpoint `state_dict` must be a mapping when present.")
        for required_field in (
            "epoch",
            "global_step",
            "metadata",
            "trainer_configuration"
        ):
            if required_field not in checkpoint:
                raise KeyError(f"Checkpoint is missing required field {required_field!r}.")
        self.configuration_snapshot(checkpoint)
        self._checkpoint_type(checkpoint)
        self._training_progress(checkpoint)
        self.combined_loader_states(checkpoint)
        epoch_metric_rank_count: int = self._validate_epoch_metric_states(checkpoint)
        self._validate_state_list(checkpoint, "optimizer_states")
        self._validate_state_list(checkpoint, "lr_schedulers")
        self._validate_callback_states(checkpoint)
        rng_state: CheckpointValue | None = checkpoint.get("rng_state")
        if rng_state is not None:
            if not isinstance(rng_state, dict):
                raise TypeError("Checkpoint `rng_state` must be a mapping.")
            self._parse_random_state(cast(CheckpointDict, rng_state))
        metadata: CheckpointValue | None = checkpoint.get("metadata")
        if not isinstance(metadata, dict):
            raise TypeError("Checkpoint `metadata` must be a mapping.")
        torch_version: CheckpointValue | None = metadata.get("torch_version")
        strategy_name: CheckpointValue | None = metadata.get("strategy")
        world_size: CheckpointValue | None = metadata.get("world_size")
        if not isinstance(torch_version, str) or not torch_version:
            raise TypeError("Checkpoint metadata `torch_version` must be a string.")
        if not isinstance(strategy_name, str) or not strategy_name:
            raise TypeError("Checkpoint metadata `strategy` must be a string.")
        if isinstance(world_size, bool) or not isinstance(world_size, int):
            raise TypeError("Checkpoint metadata `world_size` must be an integer.")
        if world_size < 1:
            raise ValueError("Checkpoint metadata `world_size` must be positive.")
        if epoch_metric_rank_count not in {0, world_size}:
            raise ValueError(
                "Checkpoint epoch metric rank state does not match metadata world size."
            )
        if (
            epoch_metric_rank_count > 0
            and epoch_metric_rank_count != self._strategy.world_size
        ):
            raise MisconfigurationError(
                "Partial-epoch metric state cannot resume under a different world size."
            )

    def _checkpoint_type(self, checkpoint: CheckpointDict) -> str:
        metadata: CheckpointValue | None = checkpoint.get("metadata")
        if not isinstance(metadata, dict):
            raise TypeError("Checkpoint `metadata` must be a mapping.")
        checkpoint_type: CheckpointValue | None = metadata.get("checkpoint_type")
        if not isinstance(checkpoint_type, str) or checkpoint_type not in {
            "training",
            "weights"
        }:
            raise ValueError(
                "Checkpoint metadata `checkpoint_type` must be 'training' or "
                "'weights'."
            )
        return checkpoint_type

    def _validate_state_list(
        self,
        checkpoint: CheckpointDict,
        field_name: str
    ) -> None:
        states: CheckpointValue | None = checkpoint.get(field_name)
        if states is None:
            return
        if not isinstance(states, list):
            raise TypeError(f"Checkpoint `{field_name}` must be a list.")
        index: int
        state: CheckpointValue
        for index, state in enumerate(states):
            if not isinstance(state, dict):
                raise TypeError(
                    f"Checkpoint `{field_name}[{index}]` must be a mapping."
                )

    def _validate_callback_states(self, checkpoint: CheckpointDict) -> None:
        states: CheckpointValue | None = checkpoint.get("callbacks")
        if states is None:
            return
        if not isinstance(states, dict):
            raise TypeError("Checkpoint `callbacks` must be a mapping.")
        state_key: str
        state: CheckpointValue
        for state_key, state in states.items():
            if not isinstance(state_key, str):
                raise TypeError("Checkpoint callback state keys must be strings.")
            if not isinstance(state, dict):
                raise TypeError(
                    f"Checkpoint callback state {state_key!r} must be a mapping."
                )
