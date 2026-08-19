from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from syntheticmind.strategies.launchers.contracts import (
    StageResult,
    _GlobalState,
    _WorkerOutput
)
from syntheticmind.utilities.types import StateDict, TrainerStage

if TYPE_CHECKING:
    from syntheticmind.core.datamodule import DataModule
    from syntheticmind.core.module import Module
    from syntheticmind.core.trainer import Trainer

__all__: list[str] = []


class _StageInvoker:
    @classmethod
    def invoke(
        cls,
        stage: TrainerStage,
        trainer: Trainer,
        module: Module,
        datamodule: DataModule,
        stage_kwargs: dict[str, object]
    ) -> StageResult:
        stage_method: object = getattr(trainer, stage)
        if not callable(stage_method):
            raise TypeError(f"Trainer stage {stage!r} is not callable.")
        return cast(StageResult, stage_method(module, datamodule, **stage_kwargs))


class _WorkerStateTransfer:
    @classmethod
    def collect(
        cls,
        trainer: Trainer,
        module: Module,
        datamodule: DataModule,
        result: StageResult
    ) -> _WorkerOutput:
        return _WorkerOutput(
            result=result,
            module_state_dict=cast(StateDict, module.state_dict()),
            trainer_state_dict=trainer.state.state_dict(),
            should_stop=trainer.should_stop,
            callback_metrics=trainer.callback_metrics,
            progress_bar_metrics_by_stage={
                running_stage: dict(metrics)
                for running_stage, metrics in trainer._progress_bar_metrics_by_stage.items()
            },
            last_progress_bar_stage=trainer._last_progress_bar_stage,
            callback_states=[callback.state_dict() for callback in trainer.callbacks],
            datamodule_state=datamodule.state_dict(),
            precision_plugin_state=trainer.precision_plugin.state_dict(),
            validation_metrics=dict(trainer.validate_loop.epoch_metrics),
            test_metrics=dict(trainer.test_loop.epoch_metrics),
            total_train_batches=trainer._total_train_batches,
            total_val_batches=trainer._total_val_batches,
            total_test_batches=trainer._total_test_batches,
            total_predict_batches=trainer._total_predict_batches,
            global_state=_GlobalState.capture()
        )

    @classmethod
    def restore(
        cls,
        worker_output: _WorkerOutput,
        trainer: Trainer,
        module: Module,
        datamodule: DataModule,
        stage: TrainerStage
    ) -> None:
        module.load_state_dict(worker_output.module_state_dict, strict=False)
        trainer.module = module
        trainer.datamodule = datamodule
        module._trainer = trainer
        trainer.state.load_state_dict(worker_output.trainer_state_dict)
        if worker_output.trainer_state_dict.get("status") == "interrupted":
            trainer.state.interrupt()
        else:
            trainer.state.set_stage(stage)
            trainer.state.finish()
        trainer.state.should_stop = worker_output.should_stop
        module._current_epoch = trainer.current_epoch
        module._global_step = trainer.global_step
        parameter: torch.nn.Parameter | None = next(module.parameters(), None)
        module._device = parameter.device if parameter is not None else torch.device("cpu")
        trainer._callback_metrics.clear()
        trainer._callback_metrics.update(worker_output.callback_metrics)
        trainer._progress_bar_metrics_by_stage = {
            running_stage: dict(metrics)
            for running_stage, metrics in worker_output.progress_bar_metrics_by_stage.items()
        }
        trainer._last_progress_bar_stage = worker_output.last_progress_bar_stage
        for callback, callback_state in zip(
            trainer.callbacks,
            worker_output.callback_states,
            strict=True
        ):
            callback.load_state_dict(callback_state)
        datamodule.load_state_dict(worker_output.datamodule_state)
        trainer.precision_plugin.load_state_dict(worker_output.precision_plugin_state)
        trainer.validate_loop.epoch_metrics = dict(worker_output.validation_metrics)
        trainer.test_loop.epoch_metrics = dict(worker_output.test_metrics)
        trainer._total_train_batches = worker_output.total_train_batches
        trainer._total_val_batches = worker_output.total_val_batches
        trainer._total_test_batches = worker_output.total_test_batches
        trainer._total_predict_batches = worker_output.total_predict_batches
        worker_output.global_state.restore()

    @classmethod
    @torch.no_grad()
    def unshare_module_storage(cls, module: Module) -> None:
        tensor: torch.Tensor
        for tensor in (*module.parameters(), *module.buffers()):
            tensor.data = tensor.data.clone()
