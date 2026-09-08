# This module:
# 1. Owns replicated-parameter training over a context-parallel sequence mesh
#
# Design decisions:
# - Module owns attention and sequence partitioning; Strategy synchronizes gradients
# - Manual backward remains on the normal Precision and Module hook path
# - Reduction occurs inside the optimizer closure before pre-step hooks
# - Rank-local checkpoints preserve RNG, mutable buffers, and continuation metadata
#
# Author: Rahul Sawhney

from datetime import timedelta
from functools import partial
from typing import cast, override

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.constants import default_pg_timeout
from torch.distributed.tensor import DTensor

from syntheticmind.core.module import Module
from syntheticmind.primitives.exceptions import MisconfigurationError
from syntheticmind.primitives.types import (
    BackwardInputs, OptimizerClosure, OptimizerClosureResult,
    OptimizerStepOutput, OptimizerStepValue
)
from syntheticmind.strategies.checkpoint import StrategyCheckpoint
from syntheticmind.strategies.model_parallel import ModelParallelStrategy
from syntheticmind.strategies.model_parallel_checkpointing.checkpoint_io import ModelParallelCheckpointWrite
from syntheticmind.strategies.model_parallel_checkpointing.rank_local import _RankLocalCheckpoint
from syntheticmind.strategies.parallel_configuration import ContextParallelConfiguration

__all__: list[str] = ["ContextParallelStrategy"]


class ContextParallelStrategy(ModelParallelStrategy):
    # Client steps slice the sequence, call manual_backward, then the managed
    # optimizer. Mean reduction corresponds to equal-size local mean losses.
    @override
    def __init__(
        self,
        configuration: ContextParallelConfiguration,
        checkpoint_write: ModelParallelCheckpointWrite = "synchronous",
        process_group_backend: str | None = None,
        timeout: timedelta | None = default_pg_timeout
    ) -> None:
        if not isinstance(configuration, ContextParallelConfiguration):
            raise TypeError("configuration must be ContextParallelConfiguration.")
        super().__init__(
            1, 1, True, process_group_backend, timeout,
            context_parallel_size=configuration.context_parallel_size,
            checkpoint_write=checkpoint_write
        )
        self._context_configuration: ContextParallelConfiguration = configuration
        self._pending_backward: bool = False
        self._failed: bool = False
        self.checkpoint: StrategyCheckpoint = _RankLocalCheckpoint(
            configuration, checkpoint_write, self.barrier, self.broadcast,
            self._validate_checkpoint_boundary
        )

    @override
    def setup(self, model: nn.Module, device: torch.device) -> nn.Module:
        execution_model: nn.Module = super().setup(model, device)
        if self.precision_plugin.precision not in {"32-true", "bf16-true"}:
            raise MisconfigurationError("ContextParallelStrategy supports float32 or true BF16 only.")
        tensor: torch.Tensor
        tensors: list[torch.Tensor] = [*model.parameters(), *model.buffers()]
        if any(isinstance(tensor, DTensor) for tensor in tensors):
            raise MisconfigurationError("ContextParallelStrategy requires ordinary replicated tensors.")
        signature: list[tuple[str, tuple[int, ...], str]] = []
        name: str
        for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
            signature.append((name, tuple(tensor.shape), str(tensor.dtype)))
        signatures: list[list[tuple[str, tuple[int, ...], str]] | None] = [None] * self.world_size
        dist.all_gather_object(signatures, signature)
        if any(entry != signature for entry in signatures):
            raise MisconfigurationError("Context-parallel ranks constructed incompatible model replicas.")
        for tensor in tensors:
            dist.broadcast(tensor.detach(), src=0)
        self._pending_backward: bool = False
        self._failed: bool = False
        return execution_model

    @override
    def backward(
        self,
        closure_loss: torch.Tensor,
        optimizer: torch.optim.Optimizer | None = None,
        gradient: torch.Tensor | None = None,
        retain_graph: bool | None = None,
        create_graph: bool = False,
        inputs: BackwardInputs = None
    ) -> torch.Tensor:
        self._pending_backward: bool = True
        return super().backward(
            closure_loss, optimizer, gradient, retain_graph, create_graph, inputs
        )

    @override
    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        closure: OptimizerClosure,
        model: nn.Module | None = None,
        **kwargs: OptimizerStepValue
    ) -> OptimizerStepOutput:
        if len(self.optimizers) != 1 or optimizer is not self.optimizers[0]:
            raise MisconfigurationError("ContextParallelStrategy requires one managed optimizer.")
        if isinstance(optimizer, torch.optim.LBFGS):
            raise MisconfigurationError("Repeated-closure LBFGS is not supported by this context policy.")
        result: OptimizerStepOutput = super().optimizer_step(
            optimizer, partial(self._synchronized_closure, closure, optimizer), model, **kwargs
        )
        self._pending_backward: bool = False
        return result

    @override
    def sync_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        # Every rank participates even when a Parameter has no local gradient.
        # Average equal local mean losses; sum is an explicit alternate objective.
        group: dist.ProcessGroup = self.device_mesh["context_parallel"].get_group()
        parameter: nn.Parameter
        for parameter in self.precision_plugin.main_params(optimizer):
            flags: torch.Tensor = torch.tensor(
                [int(parameter.grad is not None), int(parameter.grad is not None and parameter.grad.is_sparse)],
                device=parameter.device
            )
            dist.all_reduce(flags, group=group)
            if flags[1].item():
                raise MisconfigurationError("Sparse context-parallel gradients are unsupported.")
            if not flags[0].item():
                continue
            gradient: torch.Tensor = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
            dist.all_reduce(gradient, group=group)
            if self._context_configuration.gradient_reduction == "mean":
                gradient.div_(self.context_parallel_size)
            parameter.grad: torch.Tensor | None = gradient

    @override
    def on_exception(self, exception: BaseException) -> None:
        self._failed: bool = True
        super().on_exception(exception)

    @property
    def configuration(self) -> ContextParallelConfiguration:
        return self._context_configuration

    def _synchronized_closure(
        self, closure: OptimizerClosure, optimizer: torch.optim.Optimizer
    ) -> OptimizerClosureResult:
        result: OptimizerClosureResult = closure()
        if not self._pending_backward:
            raise MisconfigurationError("Call Module.manual_backward before a context-parallel optimizer step.")
        self.sync_gradients(optimizer)
        return result

    def _validate_checkpoint_boundary(self) -> None:
        if self._pending_backward or self._failed:
            raise MisconfigurationError("Context checkpoint requires a successful managed optimizer boundary.")
