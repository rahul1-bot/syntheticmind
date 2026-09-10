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
        policy: str = self.configuration.model_dump_json()
        policies: list[str | None] = [None] * self.world_size
        dist.all_gather_object(policies, policy)
        if any(entry != policy for entry in policies):
            raise MisconfigurationError("Context-parallel configuration must agree on every rank.")
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

    def normalize_loss(self, local_loss_sum: torch.Tensor, local_weight: int) -> torch.Tensor:
        # Converts local summed loss into a globally weighted mean after the
        # configured parameter-gradient reduction. A rank may own zero valid terms.
        error: str | None = None
        maximum_weight: int = (2 ** 53 - 1) // self.context_parallel_size
        if not isinstance(local_loss_sum, torch.Tensor) or local_loss_sum.numel() != 1:
            error: str | None = "local_loss_sum must be a scalar Tensor."
        elif not local_loss_sum.is_floating_point() or local_loss_sum.device != self.root_device:
            error: str | None = "local_loss_sum must be floating and on the Strategy device."
        if type(local_weight) is not int or not 0 <= local_weight <= maximum_weight:
            error: str | None = f"local_weight must be a non-negative integer no greater than {maximum_weight}."
        errors: list[str | None] = [None] * self.world_size
        dist.all_gather_object(errors, error)
        if any(message is not None for message in errors):
            raise MisconfigurationError(f"Context loss normalization rejected: {errors}.")
        total: torch.Tensor = torch.tensor(local_weight, dtype=torch.int64, device=self.root_device)
        dist.all_reduce(total, group=self.device_mesh["context_parallel"].get_group())
        total_weight: int = int(total.item())
        if total_weight <= 0:
            raise MisconfigurationError("Context loss requires a positive global normalization weight.")
        multiplier: int = self.context_parallel_size if self.configuration.gradient_reduction == "mean" else 1
        return local_loss_sum * (multiplier / total_weight)

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
        self._validate_optimizer(optimizer)
        try:
            result: OptimizerStepOutput = super().optimizer_step(
                optimizer, partial(self._synchronized_closure, closure, optimizer), model, **kwargs
            )
        except BaseException:
            self._failed: bool = True
            raise
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

    def _validate_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        # Parameter traversal order is a collective contract, not merely matching
        # model names. Validate it before any rank begins gradient all-reduces.
        error: str | None = None
        if self._failed or len(self.optimizers) != 1 or optimizer is not self.optimizers[0]:
            error: str | None = "Context training requires one managed optimizer and a healthy stage."
        elif isinstance(optimizer, torch.optim.LBFGS):
            error: str | None = "Repeated-closure LBFGS is unsupported."
        name: str
        parameter: nn.Parameter
        names: dict[int, str] = {
            id(parameter): name for name, parameter in self._require_module().named_parameters()
        }
        ordered: tuple[str, ...] = tuple(
            names.get(id(parameter), "<unowned>")
            for parameter in self.precision_plugin.main_params(optimizer)
        )
        if len(set(ordered)) != len(ordered) or "<unowned>" in ordered:
            error: str | None = "Optimizer parameters must be unique and owned by the connected model."
        records: list[tuple[str | None, tuple[str, ...]] | None] = [None] * self.world_size
        dist.all_gather_object(records, (error, ordered))
        if any(record != (None, ordered) for record in records):
            raise MisconfigurationError(f"Context optimizer collective contract differs: {records}.")

    def _synchronized_closure(
        self, closure: OptimizerClosure, optimizer: torch.optim.Optimizer
    ) -> OptimizerClosureResult:
        result: OptimizerClosureResult = closure()
        if not self.reduce_boolean_decision(self._pending_backward):
            raise MisconfigurationError("Call Module.manual_backward before a context-parallel optimizer step.")
        self.sync_gradients(optimizer)
        return result

    def _validate_checkpoint_boundary(self) -> None:
        if self._pending_backward or self._failed:
            raise MisconfigurationError("Context checkpoint requires a successful managed optimizer boundary.")
