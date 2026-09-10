# This module:
# 1. Executes native GPipe and 1F1B schedules under the existing Trainer lifecycle
#
# Design decisions:
# - One local Module is one sequential stage; no second training runner exists
# - Native schedules own microbatch forwards/backwards; managed optimizers own steps
# - Static tensor metadata avoids a state-mutating example forward during setup
# - Rank-local checkpoint representation prevents stage-name and RNG collisions
# - Checkpoints reject incomplete schedules and unapplied gradients
#
# Author: Rahul Sawhney

from collections.abc import Callable
from datetime import timedelta
from typing import cast, override

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.constants import default_pg_timeout
from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe
from torch.distributed.tensor import DTensor

from syntheticmind.core.module import Module
from syntheticmind.primitives.exceptions import MisconfigurationError
from syntheticmind.primitives.types import OptimizerClosure, OptimizerStepOutput, OptimizerStepValue
from syntheticmind.strategies.checkpoint import StrategyCheckpoint
from syntheticmind.strategies.model_parallel import ModelParallelStrategy
from syntheticmind.strategies.model_parallel_checkpointing.checkpoint_io import ModelParallelCheckpointWrite
from syntheticmind.strategies.model_parallel_checkpointing.rank_local import _RankLocalCheckpoint
from syntheticmind.strategies.parallel_configuration import PipelineParallelConfiguration, PipelineStageShape

__all__: list[str] = ["PipelineParallelStrategy"]

type _PipelineLoss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class PipelineParallelStrategy(ModelParallelStrategy):
    # Construct one stage Module per rank. In training_step: zero the managed
    # optimizer, call pipeline_step on every rank, then call optimizer.step once.
    # Native scheduling already differentiates the loss: do not manual_backward it.
    @override
    def __init__(
        self,
        configuration: PipelineParallelConfiguration,
        loss_fn: _PipelineLoss,
        checkpoint_write: ModelParallelCheckpointWrite = "synchronous",
        process_group_backend: str | None = None,
        timeout: timedelta | None = default_pg_timeout
    ) -> None:
        if not isinstance(configuration, PipelineParallelConfiguration):
            raise TypeError("configuration must be PipelineParallelConfiguration.")
        if not callable(loss_fn):
            raise TypeError("loss_fn must compute one scalar microbatch loss.")
        super().__init__(
            1, 1, True, process_group_backend, timeout,
            pipeline_parallel_size=configuration.world_size,
            checkpoint_write=checkpoint_write
        )
        self._pipeline_configuration: PipelineParallelConfiguration = configuration
        self._loss_fn: _PipelineLoss = loss_fn
        self._schedule: ScheduleGPipe | Schedule1F1B | None = None
        self._is_executing: bool = False
        self._pending_optimizer_step: bool = False
        self._failed: bool = False
        self.checkpoint: StrategyCheckpoint = _RankLocalCheckpoint(
            configuration, checkpoint_write, self.barrier, self.broadcast,
            self._validate_checkpoint_boundary
        )

    @override
    def setup(self, model: nn.Module, device: torch.device) -> nn.Module:
        execution_model: nn.Module = super().setup(model, device)
        module: Module = self._require_module()
        if module.trainer.state.stage != "fit" or module.trainer.enable_validation:
            raise MisconfigurationError("PipelineParallelStrategy currently supports fit without validation only.")
        if self.precision_plugin.precision not in {"32-true", "bf16-true"}:
            raise MisconfigurationError("PipelineParallelStrategy supports float32 or true BF16 only.")
        tensor: torch.Tensor
        if any(isinstance(tensor, DTensor) for tensor in [*model.parameters(), *model.buffers()]):
            raise MisconfigurationError("This pipeline policy excludes DTensor composition.")
        policy: str = self.configuration.model_dump_json()
        policies: list[str | None] = [None] * self.world_size
        dist.all_gather_object(policies, policy)
        if any(entry != policy for entry in policies):
            raise MisconfigurationError("Pipeline configuration must agree on every rank.")
        shape: PipelineStageShape = self.configuration.stages[self.global_rank]
        dtype: torch.dtype = torch.bfloat16 if self.precision_plugin.precision == "bf16-true" else torch.float32
        stage: PipelineStage = PipelineStage(
            execution_model, self.global_rank, self.world_size, device,
            input_args=torch.empty(
                shape.input_shape, device=device, dtype=dtype, requires_grad=self.global_rank > 0
            ),
            output_args=torch.empty(shape.output_shape, device=device, dtype=dtype, requires_grad=True),
            group=self.device_mesh["pipeline_parallel"].get_group()
        )
        schedule_type: type[ScheduleGPipe] | type[Schedule1F1B] = (
            ScheduleGPipe if self.configuration.schedule == "gpipe" else Schedule1F1B
        )
        self._schedule: ScheduleGPipe | Schedule1F1B | None = schedule_type(
            stage, self.configuration.n_microbatches, loss_fn=self._compute_loss,
            scale_grads=self.configuration.loss_reduction == "mean"
        )
        self._is_executing: bool = False
        self._pending_optimizer_step: bool = False
        self._failed: bool = False
        return execution_model

    def pipeline_step(
        self, inputs: torch.Tensor | None = None, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        # First rank supplies the full input batch, last rank its target. Other
        # ranks pass None. All ranks receive the same detached reporting loss.
        self._validate_step(inputs, target)
        schedule: ScheduleGPipe | Schedule1F1B | None = self._schedule
        if schedule is None:
            raise RuntimeError("Pipeline schedule is unavailable before setup.")
        self._is_executing: bool = True
        self._pending_optimizer_step: bool = True
        losses: list[torch.Tensor] = []
        try:
            if self.global_rank == 0:
                schedule.step(inputs)
            elif self.global_rank == self.world_size - 1:
                schedule.step(target=target, losses=losses)
            else:
                schedule.step()
            reporting_loss: torch.Tensor = torch.zeros((), device=self.root_device)
            if self.global_rank == self.world_size - 1:
                detached: torch.Tensor = torch.stack([loss.detach().float() for loss in losses])
                reporting_loss: torch.Tensor = detached.mean() if self.configuration.loss_reduction == "mean" else detached.sum()
            dist.broadcast(reporting_loss, src=self.world_size - 1)
            return reporting_loss
        except BaseException:
            self._failed: bool = True
            raise
        finally:
            self._is_executing: bool = False

    @override
    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        closure: OptimizerClosure,
        model: nn.Module | None = None,
        **kwargs: OptimizerStepValue
    ) -> OptimizerStepOutput:
        if self._is_executing or self._failed or not self._pending_optimizer_step:
            raise MisconfigurationError("Complete one pipeline_step before its managed optimizer step.")
        if len(self.optimizers) != 1 or optimizer is not self.optimizers[0]:
            raise MisconfigurationError("PipelineParallelStrategy requires one managed optimizer per stage.")
        if isinstance(optimizer, torch.optim.LBFGS):
            raise MisconfigurationError("Repeated-closure LBFGS is not supported by this pipeline policy.")
        try:
            result: OptimizerStepOutput = super().optimizer_step(optimizer, closure, model, **kwargs)
        except BaseException:
            self._failed: bool = True
            raise
        self._pending_optimizer_step: bool = False
        return result

    @override
    def teardown(self) -> None:
        self._schedule: ScheduleGPipe | Schedule1F1B | None = None
        super().teardown()

    @override
    def on_exception(self, exception: BaseException) -> None:
        self._failed: bool = True
        super().on_exception(exception)

    @property
    def configuration(self) -> PipelineParallelConfiguration:
        return self._pipeline_configuration

    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = self._loss_fn(output, target)
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise ValueError("Pipeline loss_fn must return a scalar Tensor.")
        return loss.reshape(())

    def _validate_step(self, inputs: torch.Tensor | None, target: torch.Tensor | None) -> None:
        error: str | None = None
        if self._schedule is None or self._is_executing or self._pending_optimizer_step or self._failed:
            error: str | None = "Pipeline schedule is unavailable, active, failed, or awaiting an optimizer step."
        elif len(self.optimizers) != 1:
            error: str | None = "Exactly one managed optimizer is required per stage."
        elif self.global_rank == 0:
            shape: tuple[int, ...] = self.configuration.stages[0].input_shape
            expected: tuple[int, ...] = (shape[0] * self.configuration.n_microbatches, *shape[1:])
            if not isinstance(inputs, torch.Tensor) or tuple(inputs.shape) != expected:
                error: str | None = f"First stage inputs must have shape {expected}."
            elif inputs.device != self.root_device:
                error: str | None = "First stage inputs are on the wrong device."
            elif inputs.dtype != (torch.bfloat16 if self.precision_plugin.precision == "bf16-true" else torch.float32):
                error: str | None = "First stage inputs must match the configured floating precision."
            elif inputs.requires_grad:
                error: str | None = "This static pipeline does not differentiate first-stage input batches."
            elif target is not None:
                error: str | None = "Only the final stage supplies a target."
        elif inputs is not None:
            error: str | None = "Only the first stage supplies inputs."
        if self.global_rank == self.world_size - 1:
            batch_size: int = self.configuration.stages[-1].output_shape[0] * self.configuration.n_microbatches
            if not isinstance(target, torch.Tensor) or target.ndim < 1 or target.shape[0] != batch_size:
                error: str | None = "Final stage target must contain the complete batch."
            elif target.device != self.root_device:
                error: str | None = "Final stage target is on the wrong device."
        elif target is not None:
            error: str | None = "Only the final stage supplies a target."
        errors: list[str | None] = [None] * self.world_size
        dist.all_gather_object(errors, error)
        if any(message is not None for message in errors):
            raise MisconfigurationError(f"Pipeline step input rejected before communication: {errors}.")

    def _validate_checkpoint_boundary(self) -> None:
        if self._is_executing or self._pending_optimizer_step or self._failed:
            raise MisconfigurationError("Pipeline checkpoint requires a completed schedule and optimizer step.")
