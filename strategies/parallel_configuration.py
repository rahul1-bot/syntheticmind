# This module:
# 1. Defines immutable execution contracts for context and pipeline parallelism
#
# Design decisions:
# - Checkpoints compare the same validated policy used to construct execution
# - Shapes describe one tensor microbatch, not a live tensor or inferred model
# - Distinct algorithms use explicit discriminators rather than optional fields
#
# Author: Rahul Sawhney

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

__all__: list[str] = [
    "ContextParallelConfiguration",
    "PipelineParallelConfiguration",
    "PipelineStageShape"
]

type _Shape = Annotated[tuple[Annotated[int, Field(gt=0)], ...], Field(min_length=1)]
type _Identity = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]


class PipelineStageShape(BaseModel):
    # Static input/output metadata for one sequential pipeline partition.
    input_shape: _Shape
    output_shape: _Shape
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", strict=True, validate_default=True
    )

    @model_validator(mode="after")
    def _validate_batch_axis(self) -> PipelineStageShape:
        if self.input_shape[0] != self.output_shape[0]:
            raise ValueError("Pipeline stages must preserve the microbatch axis.")
        return self


class PipelineParallelConfiguration(BaseModel):
    # One stage per rank, equal-size microbatches, and a declared scalar objective.
    # loss_id is client-owned and must change when the objective changes meaning.
    kind: Literal["pipeline"] = "pipeline"
    schema_version: Literal[1] = 1
    stages: Annotated[tuple[PipelineStageShape, ...], Field(min_length=2)]
    n_microbatches: Annotated[int, Field(ge=1)]
    schedule: Literal["gpipe", "1f1b"] = "gpipe"
    loss_reduction: Literal["mean", "sum"] = "mean"
    loss_id: _Identity
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", strict=True, validate_default=True
    )

    @property
    def world_size(self) -> int:
        return len(self.stages)

    @model_validator(mode="after")
    def _validate_connections(self) -> PipelineParallelConfiguration:
        previous: PipelineStageShape
        following: PipelineStageShape
        for previous, following in zip(self.stages[:-1], self.stages[1:], strict=True):
            if previous.output_shape != following.input_shape:
                raise ValueError("Adjacent pipeline tensor shapes must match exactly.")
        if self.schedule == "1f1b" and self.n_microbatches < self.world_size:
            raise ValueError("1F1B requires at least one microbatch per stage in this policy.")
        return self


class ContextParallelConfiguration(BaseModel):
    # Replicated parameters with sequence-sharded computation and explicit
    # gradient reduction; independent tensor/data/pipeline axes are excluded.
    kind: Literal["context"] = "context"
    schema_version: Literal[1] = 1
    context_parallel_size: Annotated[int, Field(ge=2)]
    gradient_reduction: Literal["mean", "sum"] = "mean"
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", strict=True, validate_default=True
    )

    @property
    def world_size(self) -> int:
        return self.context_parallel_size


type _ParallelConfiguration = Annotated[
    ContextParallelConfiguration | PipelineParallelConfiguration,
    Field(discriminator="kind")
]
