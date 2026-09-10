# This module:
# 1. Applies explicit non-reentrant activation recomputation to a selected module
# 2. Supports selective preservation of client-selected PyTorch operators
#
# Design decisions:
# - The researcher selects the region; the wrapper never guesses model structure
# - Module calls preserve PyTorch hooks, and no-grad execution bypasses recompute
# - Checkpoint control keywords cannot silently consume a model's keyword inputs
# - RNG advancement is rejected during compilation because PyTorch preserves RNG
#
# Author: Rahul Sawhney

from typing import Literal, cast, override

import torch
from torch import nn
from torch._ops import OpOverload
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts

from syntheticmind.primitives.types import Batch, ModelOutput

__all__: list[str] = ["ActivationCheckpointing"]

type ActivationRngPolicy = Literal["preserve", "advance"]


class ActivationCheckpointing(nn.Module):
    # Construct within the client's architecture. The wrapped child is part of
    # model state, so checkpoint reconstruction must rebuild the same wrapper.
    def __init__(
        self,
        module: nn.Module,
        rng_policy: ActivationRngPolicy = "preserve",
        saved_operations: tuple[OpOverload, ...] = ()
    ) -> None:
        super().__init__()
        if not isinstance(module, nn.Module):
            raise TypeError("ActivationCheckpointing requires a torch.nn.Module.")
        if rng_policy not in {"preserve", "advance"}:
            raise ValueError(f"Unsupported activation RNG policy: {rng_policy!r}.")
        operation: OpOverload
        if not isinstance(saved_operations, tuple) or not all(
            isinstance(operation, OpOverload) for operation in saved_operations
        ):
            raise TypeError("saved_operations must be a tuple of torch operator overloads.")
        self._module: nn.Module = module
        self._rng_policy: ActivationRngPolicy = rng_policy
        self._saved_operations: tuple[OpOverload, ...] = saved_operations

    @override
    def forward(self, *args: Batch, **kwargs: Batch) -> ModelOutput:
        if not torch.is_grad_enabled():
            return self._module(*args, **kwargs)
        reserved: frozenset[str] = frozenset(
            {"use_reentrant", "preserve_rng_state", "context_fn", "determinism_check", "debug", "early_stop"}
        )
        if reserved.intersection(kwargs):
            raise ValueError("Model keyword inputs conflict with activation-checkpoint controls.")
        if self._rng_policy == "advance" and torch.compiler.is_compiling():
            raise RuntimeError("Compiled activation checkpointing requires rng_policy='preserve'.")
        if self._saved_operations:
            return cast(
                ModelOutput,
                checkpoint(
                    self._module, *args, use_reentrant=False,
                    preserve_rng_state=self._rng_policy == "preserve",
                    context_fn=self._selective_contexts, **kwargs
                )
            )
        return cast(
            ModelOutput,
            checkpoint(
                self._module, *args, use_reentrant=False,
                preserve_rng_state=self._rng_policy == "preserve", **kwargs
            )
        )

    @property
    def module(self) -> nn.Module:
        return self._module

    @property
    def rng_policy(self) -> ActivationRngPolicy:
        return self._rng_policy

    def _selective_contexts(self) -> tuple[TorchDispatchMode, TorchDispatchMode]:
        return create_selective_checkpoint_contexts(list(self._saved_operations))
