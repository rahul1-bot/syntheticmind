from contextlib import AbstractContextManager
from typing import override

import syntheticmind
import torch
import torch.distributed as dist

__all__: list[str] = ["Loop"]


class Loop:
    def __init__(self) -> None:
        self._trainer: syntheticmind.Trainer | None = None
        self._should_stop: bool = False

    @property
    def trainer(self) -> syntheticmind.Trainer:
        if self._trainer is None:
            raise RuntimeError("Loop is not connected to a Trainer.")
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: syntheticmind.Trainer) -> None:
        self._trainer: syntheticmind.Trainer | None = trainer

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    @should_stop.setter
    def should_stop(self, value: bool) -> None:
        if type(value) is not bool:
            raise TypeError("Loop should_stop must be a bool.")
        self._should_stop: bool = value

    def reset(self) -> None:
        self._should_stop: bool = False

    def _evaluation_context(
        self,
        inference_mode: bool
    ) -> AbstractContextManager[None]:
        if dist.is_available() and dist.is_initialized() and dist.get_backend() == "gloo":
            return torch.no_grad()
        if inference_mode:
            return torch.inference_mode()
        return torch.no_grad()

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
