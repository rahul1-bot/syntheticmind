from typing import cast, override

import syntheticmind
import torch
from torch.distributed.tensor import DTensor

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.exceptions import MisconfigurationError
from syntheticmind.types import (
    Batch,
    CheckpointDict,
    CheckpointValue,
    StateDict,
    StateValue,
    StepOutput,
    TrainerStage
)

__all__: list[str] = ["EMAWeightAveraging"]

type _AveragedState = dict[str, torch.Tensor]


class EMAWeightAveraging(Callback):
    def __init__(
        self,
        decay: float = 0.999,
        update_every_n_steps: int = 1,
        update_starting_at_step: int | None = None,
        use_buffers: bool = True
    ) -> None:
        if type(decay) is not float or not 0.0 <= decay < 1.0:
            raise ValueError("`decay` must be a float in the interval [0.0, 1.0).")
        if type(update_every_n_steps) is not int or update_every_n_steps < 1:
            raise ValueError("`update_every_n_steps` must be a positive integer.")
        if (
            update_starting_at_step is not None
            and (
                type(update_starting_at_step) is not int
                or update_starting_at_step < 0
            )
        ):
            raise ValueError(
                "`update_starting_at_step` must be a non-negative integer or None."
            )
        if type(use_buffers) is not bool:
            raise TypeError("`use_buffers` must be a bool.")
        self.decay: float = decay
        self.update_every_n_steps: int = update_every_n_steps
        self.update_starting_at_step: int | None = update_starting_at_step
        self.use_buffers: bool = use_buffers
        self._average_state: _AveragedState | None = None
        self._original_state: _AveragedState | None = None
        self._finalized_training_state: _AveragedState | None = None
        self._last_processed_step: int = 0
        self._num_averaged: int = 0
        self._state_restored: bool = False

    @property
    @override
    def state_key(self) -> str:
        return self._generate_state_key(
            decay=self.decay,
            update_every_n_steps=self.update_every_n_steps,
            update_starting_at_step=self.update_starting_at_step,
            use_buffers=self.use_buffers
        )

    @override
    def on_fit_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        del trainer
        if self._finalized_training_state is not None:
            self._copy_state_to_module(module, self._finalized_training_state)
            self._finalized_training_state: _AveragedState | None = None
        if any(
            isinstance(tensor, DTensor)
            for tensor in self._current_state(module).values()
        ):
            raise MisconfigurationError(
                "EMAWeightAveraging does not support DTensor or sharded model state."
            )
        if self._state_restored and self._average_state is not None:
            self._align_average_state(module)
            self._state_restored: bool = False
            return
        self._average_state: _AveragedState | None = self._clone_current_state(module)
        self._original_state: _AveragedState | None = None
        self._finalized_training_state: _AveragedState | None = None
        self._last_processed_step: int = 0
        self._num_averaged: int = 0
        self._state_restored: bool = False

    @override
    def on_train_batch_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        outputs: StepOutput,
        batch: Batch,
        batch_idx: int
    ) -> None:
        del outputs, batch, batch_idx
        if trainer.global_step <= self._last_processed_step:
            return
        self._last_processed_step: int = trainer.global_step
        step_index: int = trainer.global_step - 1
        starting_step: int = self.update_starting_at_step or 0
        if step_index < starting_step:
            return
        if (step_index - starting_step) % self.update_every_n_steps != 0:
            return
        self._update_average(module)

    @override
    def on_train_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        del trainer
        if self._num_averaged > 0:
            self._finalized_training_state: _AveragedState | None = (
                self._clone_current_state(module)
            )
            self._copy_average_to_module(module)

    @override
    def on_validation_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        del trainer
        self._swap_to_average(module)

    @override
    def on_validation_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        del trainer
        self._restore_original_state(module)

    @override
    def on_test_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        del trainer
        self._swap_to_average(module)

    @override
    def on_test_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        del trainer
        self._restore_original_state(module)

    @override
    def on_predict_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        del trainer
        self._swap_to_average(module)

    @override
    def on_predict_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> None:
        del trainer
        self._restore_original_state(module)

    @override
    def on_exception(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        exception: BaseException
    ) -> None:
        del trainer, exception
        self._restore_original_state(module)

    @override
    def teardown(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        stage: TrainerStage
    ) -> None:
        del trainer, stage
        self._restore_original_state(module)

    @override
    def state_dict(self) -> StateDict:
        if self._average_state is None:
            return {}
        average_state: dict[str, StateValue] = {
            name: tensor.detach().clone()
            for name, tensor in self._average_state.items()
        }
        return {
            "average_state": average_state,
            "last_processed_step": self._last_processed_step,
            "num_averaged": self._num_averaged
        }

    @override
    def load_state_dict(self, state_dict: StateDict) -> None:
        average_value: StateValue | None = state_dict.get("average_state")
        last_step_value: StateValue | None = state_dict.get("last_processed_step")
        count_value: StateValue | None = state_dict.get("num_averaged")
        if not isinstance(average_value, dict):
            raise TypeError("EMA callback state requires an `average_state` mapping.")
        if type(last_step_value) is not int or last_step_value < 0:
            raise TypeError("EMA callback `last_processed_step` must be non-negative int.")
        if type(count_value) is not int or count_value < 0:
            raise TypeError("EMA callback `num_averaged` must be a non-negative int.")
        average_state: _AveragedState = {}
        name: str
        value: StateValue
        for name, value in average_value.items():
            if not isinstance(name, str) or not isinstance(value, torch.Tensor):
                raise TypeError("EMA callback average state must map names to tensors.")
            average_state[name] = value.detach().clone()
        self._average_state: _AveragedState | None = average_state
        self._original_state: _AveragedState | None = None
        self._finalized_training_state: _AveragedState | None = None
        self._last_processed_step: int = last_step_value
        self._num_averaged: int = count_value
        self._state_restored: bool = True

    @override
    def on_save_checkpoint(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        checkpoint: CheckpointDict
    ) -> None:
        del trainer, module
        live_training_state: _AveragedState | None = (
            self._original_state
            if self._original_state is not None
            else self._finalized_training_state
        )
        if live_training_state is None:
            return
        checkpoint_state_value: CheckpointValue | None = checkpoint.get("state_dict")
        if not isinstance(checkpoint_state_value, dict):
            raise TypeError("Checkpoint model state must be a mapping during EMA swap.")
        checkpoint_state: dict[str, CheckpointValue] = cast(
            dict[str, CheckpointValue],
            checkpoint_state_value
        )
        live_state: dict[str, CheckpointValue] = dict(checkpoint_state)
        name: str
        tensor: torch.Tensor
        for name, tensor in live_training_state.items():
            if name not in live_state:
                raise RuntimeError(
                    f"EMA live state contains {name!r}, absent from checkpoint model state."
                )
            live_state[name] = tensor.detach().clone()
        checkpoint["state_dict"] = live_state

    def _update_average(self, module: Module) -> None:
        current_state: _AveragedState = self._current_state(module)
        if self._average_state is None:
            self._average_state: _AveragedState | None = self._clone_state(current_state)
        self._validate_state(current_state)
        assert self._average_state is not None
        with torch.no_grad():
            name: str
            current_tensor: torch.Tensor
            for name, current_tensor in current_state.items():
                average_tensor: torch.Tensor = self._average_state[name]
                if average_tensor.device != current_tensor.device:
                    average_tensor: torch.Tensor = average_tensor.to(
                        current_tensor.device
                    )
                    self._average_state[name] = average_tensor
                if self._num_averaged == 0:
                    average_tensor.copy_(current_tensor.detach())
                elif average_tensor.is_floating_point() or average_tensor.is_complex():
                    average_tensor.mul_(self.decay).add_(
                        current_tensor.detach(),
                        alpha=1.0 - self.decay
                    )
                else:
                    average_tensor.copy_(current_tensor.detach())
        self._num_averaged += 1

    def _swap_to_average(self, module: Module) -> None:
        if self._average_state is None:
            return
        if self._original_state is not None:
            raise RuntimeError("EMA weights are already active for another evaluation stage.")
        self._align_average_state(module)
        self._original_state: _AveragedState | None = self._clone_current_state(module)
        self._copy_average_to_module(module)

    def _restore_original_state(self, module: Module) -> None:
        if self._original_state is None:
            return
        original_state: _AveragedState = self._original_state
        self._original_state: _AveragedState | None = None
        self._copy_state_to_module(module, original_state)

    def _copy_average_to_module(self, module: Module) -> None:
        if self._average_state is None:
            return
        self._align_average_state(module)
        self._copy_state_to_module(module, self._average_state)

    def _align_average_state(self, module: Module) -> None:
        if self._average_state is None:
            return
        current_state: _AveragedState = self._current_state(module)
        self._validate_state(current_state)
        aligned_state: _AveragedState = {}
        name: str
        current_tensor: torch.Tensor
        for name, current_tensor in current_state.items():
            average_tensor: torch.Tensor = self._average_state[name]
            aligned_state[name] = average_tensor.to(
                device=current_tensor.device,
                dtype=current_tensor.dtype
            ).detach().clone()
        self._average_state: _AveragedState | None = aligned_state

    def _validate_state(self, current_state: _AveragedState) -> None:
        assert self._average_state is not None
        current_names: set[str] = set(current_state)
        average_names: set[str] = set(self._average_state)
        if current_names != average_names:
            missing: list[str] = sorted(current_names - average_names)
            unexpected: list[str] = sorted(average_names - current_names)
            raise RuntimeError(
                "EMA state does not match the configured model. "
                f"Missing={missing}, unexpected={unexpected}."
            )
        name: str
        tensor: torch.Tensor
        for name, tensor in current_state.items():
            average_tensor: torch.Tensor = self._average_state[name]
            if average_tensor.shape != tensor.shape:
                raise RuntimeError(
                    f"EMA tensor shape for {name!r} is {tuple(average_tensor.shape)}, "
                    f"expected {tuple(tensor.shape)}."
                )

    def _current_state(self, module: Module) -> _AveragedState:
        state: _AveragedState = {
            name: parameter
            for name, parameter in module.named_parameters()
        }
        if self.use_buffers:
            state.update(
                {
                    name: buffer
                    for name, buffer in module.named_buffers()
                }
            )
        return state

    def _clone_current_state(self, module: Module) -> _AveragedState:
        return self._clone_state(self._current_state(module))

    def _clone_state(self, state: _AveragedState) -> _AveragedState:
        return {
            name: tensor.detach().clone()
            for name, tensor in state.items()
        }

    def _copy_state_to_module(
        self,
        module: Module,
        state: _AveragedState
    ) -> None:
        current_state: _AveragedState = self._current_state(module)
        current_names: set[str] = set(current_state)
        state_names: set[str] = set(state)
        if current_names != state_names:
            raise RuntimeError("EMA state keys changed while weights were being swapped.")
        with torch.no_grad():
            name: str
            current_tensor: torch.Tensor
            for name, current_tensor in current_state.items():
                source_tensor: torch.Tensor = state[name].to(
                    device=current_tensor.device,
                    dtype=current_tensor.dtype
                )
                current_tensor.copy_(source_tensor)

    @override
    def __repr__(self) -> str:
        return (
            f"EMAWeightAveraging(decay={self.decay}, "
            f"update_every_n_steps={self.update_every_n_steps}, "
            f"update_starting_at_step={self.update_starting_at_step}, "
            f"use_buffers={self.use_buffers})"
        )
