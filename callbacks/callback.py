from typing import override

import syntheticmind
import torch

from syntheticmind.types import (
    Batch,
    CheckpointDict,
    ModelOutput,
    StateDict,
    StateScalar,
    StepOutput,
    TrainerStage
)

__all__: list[str] = ["Callback"]


class Callback:
    def setup(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module,
        stage: TrainerStage
    ) -> None:
        pass

    def teardown(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module,
        stage: TrainerStage
    ) -> None:
        pass

    def on_fit_start(
        self, trainer: syntheticmind.Trainer, module: syntheticmind.Module
    ) -> None:
        pass

    def on_fit_end(
        self, trainer: syntheticmind.Trainer, module: syntheticmind.Module
    ) -> None:
        pass

    def on_train_start(
        self, trainer: syntheticmind.Trainer, module: syntheticmind.Module
    ) -> None:
        pass

    def on_train_end(
        self, trainer: syntheticmind.Trainer, module: syntheticmind.Module
    ) -> None:
        pass

    def on_train_epoch_start(
        self, trainer: syntheticmind.Trainer, module: syntheticmind.Module
    ) -> None:
        pass

    def on_train_epoch_end(
        self, trainer: syntheticmind.Trainer, module: syntheticmind.Module
    ) -> None:
        pass

    def on_train_batch_start(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module,
        batch: Batch,
        batch_idx: int
    ) -> None:
        pass

    def on_train_batch_end(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module,
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int
    ) -> None:
        pass

    def on_before_backward(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module,
        loss: torch.Tensor
    ) -> None:
        pass

    def on_after_backward(
        self, trainer: syntheticmind.Trainer, module: syntheticmind.Module
    ) -> None:
        pass

    def on_before_optimizer_step(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module,
        optimizer: torch.optim.Optimizer
    ) -> None:
        pass

    def on_before_zero_grad(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module,
        optimizer: torch.optim.Optimizer
    ) -> None:
        pass

    def on_validation_start(
        self, trainer: syntheticmind.Trainer, module: syntheticmind.Module
    ) -> None:
        pass

    def on_validation_end(
        self, trainer: syntheticmind.Trainer, module: syntheticmind.Module
    ) -> None:
        pass

    def on_validation_epoch_start(
        self, trainer: syntheticmind.Trainer, module: syntheticmind.Module
    ) -> None:
        pass

    def on_validation_epoch_end(
        self, trainer: syntheticmind.Trainer, module: syntheticmind.Module
    ) -> None:
        pass

    def on_validation_batch_start(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        pass

    def on_validation_batch_end(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module,
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        pass

    def on_test_start(
        self, trainer: syntheticmind.Trainer, module: syntheticmind.Module
    ) -> None:
        pass

    def on_test_end(
        self, trainer: syntheticmind.Trainer, module: syntheticmind.Module
    ) -> None:
        pass

    def on_test_epoch_start(
        self, trainer: syntheticmind.Trainer, module: syntheticmind.Module
    ) -> None:
        pass

    def on_test_epoch_end(
        self, trainer: syntheticmind.Trainer, module: syntheticmind.Module
    ) -> None:
        pass

    def on_test_batch_start(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        pass

    def on_test_batch_end(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module,
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        pass

    def on_predict_start(
        self, trainer: syntheticmind.Trainer, module: syntheticmind.Module
    ) -> None:
        pass

    def on_predict_end(
        self, trainer: syntheticmind.Trainer, module: syntheticmind.Module
    ) -> None:
        pass

    def on_sanity_check_start(
        self, trainer: syntheticmind.Trainer, module: syntheticmind.Module
    ) -> None:
        pass

    def on_sanity_check_end(
        self, trainer: syntheticmind.Trainer, module: syntheticmind.Module
    ) -> None:
        pass

    def on_predict_epoch_start(
        self, trainer: syntheticmind.Trainer, module: syntheticmind.Module
    ) -> None:
        pass

    def on_predict_epoch_end(
        self, trainer: syntheticmind.Trainer, module: syntheticmind.Module
    ) -> None:
        pass

    def on_predict_batch_start(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        pass

    def on_predict_batch_end(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module,
        outputs: ModelOutput,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        pass

    def on_exception(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module,
        exception: BaseException
    ) -> None:
        pass

    def on_save_checkpoint(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module,
        checkpoint: CheckpointDict
    ) -> None:
        pass

    def on_load_checkpoint(
        self,
        trainer: syntheticmind.Trainer,
        module: syntheticmind.Module,
        checkpoint: CheckpointDict
    ) -> None:
        pass

    def _generate_state_key(self, **kwargs: StateScalar) -> str:
        return f"{type(self).__qualname__}{repr(kwargs)}"

    @property
    def state_key(self) -> str:
        return type(self).__qualname__

    def state_dict(self) -> StateDict:
        state_dict: StateDict = {}
        return state_dict

    def load_state_dict(self, state_dict: StateDict) -> None:
        del state_dict

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
