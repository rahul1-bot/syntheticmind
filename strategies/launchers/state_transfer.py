from typing import cast

import syntheticmind
import torch

from syntheticmind.callbacks.callback import Callback
from syntheticmind.data.datamodule import DataModule
from syntheticmind.core.module import Module
from syntheticmind.strategies.launchers.contracts import (
    StageResult,
    _GlobalState,
    _WorkerOutput
)
from syntheticmind.primitives.types import (
    CheckpointDict,
    CheckpointValue,
    DataLoaderState,
    OptimizerState,
    RunningStage,
    StateDict,
    TrainerStage
)

__all__: list[str] = []


class _WorkerStateTransfer:
    @classmethod
    def capture_global_state(cls, trainer: syntheticmind.Trainer) -> _GlobalState:
        del cls
        return _GlobalState(
            random_state=trainer._randomness.capture(),
            deterministic_algorithms=trainer._determinism.algorithms_enabled,
            deterministic_warn_only=trainer._determinism.warn_only_enabled,
            cudnn_benchmark=trainer._determinism.benchmark_enabled,
            cudnn_deterministic=trainer._determinism.cudnn_deterministic_enabled
        )

    @classmethod
    def restore_global_state(
        cls,
        trainer: syntheticmind.Trainer,
        global_state: _GlobalState
    ) -> None:
        del cls
        trainer._randomness.restore(global_state.random_state)
        trainer._determinism.restore(
            global_state.deterministic_algorithms,
            global_state.deterministic_warn_only,
            global_state.cudnn_benchmark,
            global_state.cudnn_deterministic
        )

    @classmethod
    def collect(
        cls,
        trainer: syntheticmind.Trainer,
        module: Module,
        datamodule: DataModule | None,
        result: StageResult
    ) -> _WorkerOutput:
        callback_metrics: dict[str, torch.Tensor] = {}
        metric_name: str
        metric_value: torch.Tensor
        for metric_name, metric_value in trainer.callback_metrics.items():
            callback_metrics[metric_name]: torch.Tensor = metric_value.detach().cpu()
        logged_metrics: dict[str, torch.Tensor] = {}
        for metric_name, metric_value in trainer.logged_metrics.items():
            logged_metrics[metric_name]: torch.Tensor = metric_value.detach().cpu()
        progress_metrics: dict[RunningStage, dict[str, float]] = {}
        running_stage: RunningStage
        running_metrics: dict[str, float]
        for running_stage, running_metrics in (
            trainer._logger_connector.progress_bar_metrics_by_stage.items()
        ):
            progress_metrics[running_stage]: dict[str, float] = dict(running_metrics)
        callback_states: list[StateDict] = []
        callback: Callback
        for callback in trainer.callbacks:
            callback_states.append(callback.state_dict())
        return _WorkerOutput(
            result=result,
            module_state_dict=cast(StateDict, module.state_dict()),
            trainer_state_dict=trainer.state.state_dict(),
            should_stop=trainer.should_stop,
            callback_metrics=callback_metrics,
            logged_metrics=logged_metrics,
            progress_bar_metrics_by_stage=progress_metrics,
            last_progress_bar_stage=trainer._logger_connector.last_progress_bar_stage,
            hyperparameters_logged=(
                trainer._logger_connector.hyperparameters_logged
            ),
            callback_states=callback_states,
            datamodule_state=(
                datamodule.state_dict() if datamodule is not None else None
            ),
            precision_plugin_state=trainer.precision_plugin.state_dict(),
            validation_metrics=dict(trainer.validate_loop.epoch_metrics),
            test_metrics=dict(trainer.test_loop.epoch_metrics),
            total_train_batches=trainer._data_connector.total_train_batches,
            total_val_batches=trainer._data_connector.total_val_batches,
            total_test_batches=trainer._data_connector.total_test_batches,
            total_predict_batches=trainer._data_connector.total_predict_batches,
            train_dataloader_states=trainer._checkpoint_connector.train_dataloader_states,
            train_dataloader_state_active=(
                trainer._checkpoint_connector.train_dataloader_state_active
            ),
            train_dataloader_batches_completed=(
                trainer._checkpoint_connector.train_dataloader_batches_completed
            ),
            train_total_batch_idx=trainer._checkpoint_connector.train_total_batch_idx,
            train_epoch_rng_state=trainer._checkpoint_connector.train_epoch_rng_state,
            train_epoch_metric_states_by_rank=(
                trainer._checkpoint_connector.train_epoch_metric_states_by_rank
            ),
            train_optimizer_states=(
                trainer._checkpoint_connector.train_optimizer_states
            ),
            train_scheduler_states=(
                trainer._checkpoint_connector.train_scheduler_states
            ),
            received_sigterm=trainer.received_sigterm,
            is_last_batch=trainer.is_last_batch,
            global_state=cls.capture_global_state(trainer)
        )

    @classmethod
    def restore(
        cls,
        worker_output: _WorkerOutput,
        trainer: syntheticmind.Trainer,
        module: Module,
        datamodule: DataModule | None,
        stage: TrainerStage
    ) -> None:
        module.load_state_dict(worker_output.module_state_dict, strict=False)
        trainer.module: Module | None = module
        trainer.strategy.model: torch.nn.Module | None = module
        trainer.datamodule: DataModule | None = datamodule
        module.trainer: syntheticmind.Trainer = trainer
        if datamodule is not None:
            datamodule.trainer: syntheticmind.Trainer = trainer
        trainer.state.load_state_dict(worker_output.trainer_state_dict)
        if worker_output.trainer_state_dict.get("status") == "interrupted":
            trainer.state.interrupt()
        else:
            trainer.state.set_stage(stage)
            trainer.state.finish()
        trainer.state.should_stop: bool = worker_output.should_stop
        module._current_epoch: int = trainer.current_epoch
        module._global_step: int = trainer.global_step
        parameter: torch.nn.Parameter | None = next(module.parameters(), None)
        module._device: torch.device = (
            parameter.device if parameter is not None else torch.device("cpu")
        )
        trainer._logger_connector.callback_metrics.clear()
        trainer._logger_connector.callback_metrics.update(worker_output.callback_metrics)
        trainer._logger_connector.logged_metrics.clear()
        trainer._logger_connector.logged_metrics.update(worker_output.logged_metrics)
        restored_progress_metrics: dict[RunningStage, dict[str, float]] = {}
        running_stage: RunningStage
        running_metrics: dict[str, float]
        for running_stage, running_metrics in worker_output.progress_bar_metrics_by_stage.items():
            restored_progress_metrics[running_stage]: dict[str, float] = dict(
                running_metrics
            )
        trainer._logger_connector.progress_bar_metrics_by_stage: dict[
            RunningStage,
            dict[str, float]
        ] = restored_progress_metrics
        trainer._logger_connector.last_progress_bar_stage: RunningStage | None = (
            worker_output.last_progress_bar_stage
        )
        trainer._logger_connector.hyperparameters_logged: bool = (
            worker_output.hyperparameters_logged
        )
        callback: Callback
        callback_state: StateDict
        for callback, callback_state in zip(
            trainer.callbacks,
            worker_output.callback_states,
            strict=True
        ):
            callback.load_state_dict(callback_state)
        if datamodule is not None and worker_output.datamodule_state is not None:
            datamodule.load_state_dict(worker_output.datamodule_state)
        trainer.precision_plugin.load_state_dict(worker_output.precision_plugin_state)
        trainer.validate_loop.epoch_metrics: dict[str, float] = dict(
            worker_output.validation_metrics
        )
        trainer.test_loop.epoch_metrics: dict[str, float] = dict(
            worker_output.test_metrics
        )
        trainer._data_connector.total_train_batches: int | None = worker_output.total_train_batches
        trainer._data_connector.total_val_batches: list[int | None] | None = (
            worker_output.total_val_batches
        )
        trainer._data_connector.total_test_batches: list[int | None] | None = (
            worker_output.total_test_batches
        )
        trainer._data_connector.total_predict_batches: list[int | None] | None = (
            worker_output.total_predict_batches
        )
        cls._restore_checkpoint_state(worker_output, trainer)
        trainer._signal_connector.received_sigterm: bool = (
            trainer.received_sigterm or worker_output.received_sigterm
        )
        if trainer.fit_loop.epoch_loop is not None:
            trainer.fit_loop.epoch_loop.is_last_batch: bool = worker_output.is_last_batch
        cls.restore_global_state(trainer, worker_output.global_state)

    @classmethod
    def _restore_checkpoint_state(
        cls,
        worker_output: _WorkerOutput,
        trainer: syntheticmind.Trainer
    ) -> None:
        del cls
        trainer._checkpoint_connector.train_dataloader_states: list[DataLoaderState] = (
            worker_output.train_dataloader_states
        )
        trainer._checkpoint_connector.train_dataloader_state_active: bool = (
            worker_output.train_dataloader_state_active
        )
        trainer._checkpoint_connector.train_dataloader_batches_completed: int = (
            worker_output.train_dataloader_batches_completed
        )
        trainer._checkpoint_connector.train_total_batch_idx: int = (
            worker_output.train_total_batch_idx
        )
        trainer._checkpoint_connector.train_epoch_rng_state: CheckpointDict | None = (
            worker_output.train_epoch_rng_state
        )
        trainer._checkpoint_connector.train_epoch_metric_states_by_rank: list[
            CheckpointDict
        ] = worker_output.train_epoch_metric_states_by_rank
        trainer._checkpoint_connector.train_optimizer_states: list[OptimizerState] = (
            worker_output.train_optimizer_states
        )
        trainer._checkpoint_connector.train_scheduler_states: list[CheckpointValue] = (
            worker_output.train_scheduler_states
        )

    @classmethod
    @torch.no_grad()
    def unshare_module_storage(cls, module: Module) -> None:
        tensor: torch.Tensor
        for tensor in (*module.parameters(), *module.buffers()):
            tensor.data: torch.Tensor = tensor.data.clone()
