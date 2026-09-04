from collections.abc import Callable
from typing import ClassVar, cast

import torch

from syntheticmind.exceptions import MisconfigurationError
from syntheticmind.types import (
    Batch,
    CheckpointDict,
    DataLoaderSource,
    DeviceTransferable,
    ModelOutput,
    StepOutput,
    TrainerStage
)

__all__: list[str] = ["CheckpointHooks", "DataHooks", "ModelHooks"]


class DataHooks:
    def __init__(self) -> None:
        super().__init__()
        self.prepare_data_per_node: bool = True
        self.allow_zero_length_dataloader_with_multiple_devices: bool = False

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: TrainerStage) -> None:
        pass

    def teardown(self, stage: TrainerStage) -> None:
        pass

    def train_dataloader(self) -> DataLoaderSource:
        raise MisconfigurationError(
            "train_dataloader() must be implemented before fitting a model."
        )

    def val_dataloader(self) -> DataLoaderSource:
        raise MisconfigurationError(
            "val_dataloader() must be implemented before validating a model."
        )

    def test_dataloader(self) -> DataLoaderSource:
        raise MisconfigurationError(
            "test_dataloader() must be implemented before testing a model."
        )

    def predict_dataloader(self) -> DataLoaderSource:
        raise MisconfigurationError(
            "predict_dataloader() must be implemented before predicting with a model."
        )

    def on_before_batch_transfer(
        self,
        batch: Batch,
        dataloader_idx: int = 0
    ) -> Batch:
        del dataloader_idx
        return batch

    def transfer_batch_to_device(
        self,
        batch: Batch,
        device: torch.device,
        dataloader_idx: int = 0
    ) -> Batch:
        del dataloader_idx
        return self._move_data_to_device(batch, device)

    def on_after_batch_transfer(
        self,
        batch: Batch,
        dataloader_idx: int = 0
    ) -> Batch:
        del dataloader_idx
        return batch

    def _move_data_to_device(
        self,
        data: Batch,
        device: torch.device
    ) -> Batch:
        non_blocking: bool = device.type == "cuda"
        if isinstance(data, torch.Tensor):
            return data.to(device, non_blocking=non_blocking)
        if isinstance(data, dict):
            moved_mapping: dict[str, Batch] = {}
            key: str
            value: Batch
            for key, value in data.items():
                moved_mapping[key]: Batch = self._move_data_to_device(value, device)
            return moved_mapping
        if isinstance(data, tuple):
            moved_items: list[Batch] = []
            item: Batch
            for item in data:
                moved_items.append(self._move_data_to_device(item, device))
            tuple_items: tuple[Batch, ...] = tuple(moved_items)
            if hasattr(data, "_fields"):
                maker: Callable[[tuple[Batch, ...]], Batch] = cast(
                    Callable[[tuple[Batch, ...]], Batch],
                    getattr(type(data), "_make")
                )
                return maker(tuple_items)
            return tuple_items
        if isinstance(data, list):
            moved_values: list[Batch] = []
            item: Batch
            for item in data:
                moved_values.append(self._move_data_to_device(item, device))
            return moved_values
        if isinstance(data, DeviceTransferable):
            return data.to(device, non_blocking=non_blocking)
        return data


class ModelHooks:
    _training_step_logging_hooks: ClassVar[frozenset[str]] = frozenset(
        {
            "training_step",
            "on_before_backward",
            "backward",
            "on_after_backward",
            "on_before_optimizer_step",
            "optimizer_step",
            "configure_gradient_clipping",
            "clip_gradients",
            "on_before_zero_grad",
            "optimizer_zero_grad",
            "on_train_batch_start",
            "on_train_batch_end"
        }
    )
    _evaluation_step_logging_hooks: ClassVar[frozenset[str]] = frozenset(
        {
            "validation_step",
            "test_step",
            "on_validation_batch_start",
            "on_validation_batch_end",
            "on_test_batch_start",
            "on_test_batch_end"
        }
    )
    _epoch_logging_hooks: ClassVar[frozenset[str]] = frozenset(
        {
            "on_train_start",
            "on_validation_start",
            "on_test_start",
            "on_train_epoch_start",
            "on_train_epoch_end",
            "on_validation_epoch_start",
            "on_validation_epoch_end",
            "on_test_epoch_start",
            "on_test_epoch_end"
        }
    )
    _blocked_logging_hooks: ClassVar[frozenset[str]] = frozenset(
        {
            "on_fit_start",
            "on_fit_end",
            "on_train_end",
            "on_validation_end",
            "on_test_end",
            "on_predict_start",
            "on_predict_end",
            "on_predict_epoch_start",
            "on_predict_epoch_end",
            "on_predict_batch_start",
            "on_predict_batch_end",
            "on_before_batch_transfer",
            "transfer_batch_to_device",
            "on_after_batch_transfer",
            "prepare_data",
            "setup",
            "teardown",
            "train_dataloader",
            "val_dataloader",
            "test_dataloader",
            "predict_dataloader",
            "on_validation_model_zero_grad",
            "on_validation_model_eval",
            "on_validation_model_train",
            "on_test_model_eval",
            "on_test_model_train",
            "on_predict_model_eval",
            "configure_model",
            "on_save_checkpoint",
            "on_load_checkpoint",
            "predict_step",
            "configure_optimizers",
            "lr_scheduler_step",
            "on_sanity_check_start",
            "on_sanity_check_end",
            "on_exception",
            "state_dict",
            "load_state_dict",
            "configure_callbacks"
        }
    )

    def on_fit_start(self) -> None:
        pass

    def on_fit_end(self) -> None:
        pass

    def on_train_start(self) -> None:
        pass

    def on_train_end(self) -> None:
        pass

    def on_validation_start(self) -> None:
        pass

    def on_validation_end(self) -> None:
        pass

    def on_test_start(self) -> None:
        pass

    def on_test_end(self) -> None:
        pass

    def on_predict_start(self) -> None:
        pass

    def on_predict_end(self) -> None:
        pass

    def on_train_epoch_start(self) -> None:
        pass

    def on_train_epoch_end(self) -> None:
        pass

    def on_validation_epoch_start(self) -> None:
        pass

    def on_validation_epoch_end(self) -> None:
        pass

    def on_test_epoch_start(self) -> None:
        pass

    def on_test_epoch_end(self) -> None:
        pass

    def on_predict_epoch_start(self) -> None:
        pass

    def on_predict_epoch_end(self) -> None:
        pass

    def on_validation_model_zero_grad(self) -> None:
        pass

    def on_validation_model_eval(self) -> None:
        pass

    def on_validation_model_train(self) -> None:
        pass

    def on_test_model_eval(self) -> None:
        pass

    def on_test_model_train(self) -> None:
        pass

    def on_predict_model_eval(self) -> None:
        pass

    def on_train_batch_start(self, batch: Batch, batch_idx: int) -> int | None:
        del batch, batch_idx
        return None

    def on_train_batch_end(
        self,
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int
    ) -> None:
        del outputs, batch, batch_idx

    def on_validation_batch_start(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        del batch, batch_idx, dataloader_idx

    def on_validation_batch_end(
        self,
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        del outputs, batch, batch_idx, dataloader_idx

    def on_test_batch_start(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        del batch, batch_idx, dataloader_idx

    def on_test_batch_end(
        self,
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        del outputs, batch, batch_idx, dataloader_idx

    def on_predict_batch_start(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        del batch, batch_idx, dataloader_idx

    def on_predict_batch_end(
        self,
        outputs: ModelOutput,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        del outputs, batch, batch_idx, dataloader_idx

    def on_before_backward(self, loss: torch.Tensor) -> None:
        del loss

    def on_after_backward(self) -> None:
        pass

    def on_before_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer
    ) -> None:
        del optimizer

    def on_before_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer

    def configure_model(self) -> None:
        pass

    @classmethod
    def _resolve_logging_schedule(
        cls,
        hook_name: str,
        on_step: bool | None,
        on_epoch: bool | None
    ) -> tuple[bool, bool]:
        if hook_name in cls._training_step_logging_hooks:
            resolved_on_step: bool = True if on_step is None else on_step
            resolved_on_epoch: bool = False if on_epoch is None else on_epoch
            return resolved_on_step, resolved_on_epoch
        if hook_name in cls._evaluation_step_logging_hooks:
            resolved_on_step: bool = False if on_step is None else on_step
            resolved_on_epoch: bool = True if on_epoch is None else on_epoch
            return resolved_on_step, resolved_on_epoch
        if hook_name in cls._epoch_logging_hooks:
            resolved_on_step: bool = False if on_step is None else on_step
            resolved_on_epoch: bool = True if on_epoch is None else on_epoch
            if resolved_on_step:
                raise MisconfigurationError(
                    f"self.log(on_step=True) is invalid inside {hook_name}."
                )
            if not resolved_on_epoch:
                raise MisconfigurationError(
                    f"self.log(on_epoch=False) is invalid inside {hook_name}."
                )
            return resolved_on_step, resolved_on_epoch
        if hook_name in cls._blocked_logging_hooks:
            raise MisconfigurationError(f"self.log() cannot be used inside {hook_name}.")
        raise MisconfigurationError(
            f"self.log() is not registered for trainer hook {hook_name}."
        )


class CheckpointHooks:
    def on_load_checkpoint(self, checkpoint: CheckpointDict) -> None:
        del checkpoint

    def on_save_checkpoint(self, checkpoint: CheckpointDict) -> None:
        del checkpoint
