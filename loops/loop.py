# This module:
# 1. Defines Loop, the shared Trainer attachment, cooperative stop state, and
#    evaluation autograd-context policy inherited by execution loops
#
# Design decisions:
# - A loop cannot execute detached from Trainer; failed access identifies the
#   missing lifecycle connection rather than returning partial state
# - Loop-local should_stop remains distinct from Trainer's cross-callback stop
#   state so an inner lifecycle can finish or reset independently
# - Gloo evaluation uses no_grad rather than inference_mode because distributed
#   and tensor implementations may require version-counter behavior unavailable
#   under inference tensors
#
# Author: Rahul Sawhney

from contextlib import AbstractContextManager
from typing import override

import syntheticmind
import torch
import torch.distributed as dist

__all__: list[str] = ["Loop"]


class Loop:
    # Shared stateful attachment for Trainer-owned iterative lifecycles.
    def __init__(self) -> None:
        self._trainer: syntheticmind.Trainer | None = None
        self._should_stop: bool = False

    @property
    def trainer(self) -> syntheticmind.Trainer:
        # Connected Trainer or a lifecycle error before composition.
        if self._trainer is None:
            raise RuntimeError("Loop is not connected to a Trainer.")
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: syntheticmind.Trainer) -> None:
        self._trainer: syntheticmind.Trainer | None = trainer

    @property
    def should_stop(self) -> bool:
        # Loop-local cooperative stop request.
        return self._should_stop

    @should_stop.setter
    def should_stop(self, value: bool) -> None:
        if type(value) is not bool:
            raise TypeError("Loop should_stop must be a bool.")
        self._should_stop: bool = value

    def reset(self) -> None:
        # Clears transient stop state before loop reuse.
        self._should_stop: bool = False

    def _evaluation_context(
        self,
        inference_mode: bool
    ) -> AbstractContextManager[None]:
        # Selects inference_mode only where the backend can safely consume
        # inference tensors; otherwise preserves no-gradient version semantics.
        if dist.is_available() and dist.is_initialized() and dist.get_backend() == "gloo":
            return torch.no_grad()
        if inference_mode:
            return torch.inference_mode()
        return torch.no_grad()

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
