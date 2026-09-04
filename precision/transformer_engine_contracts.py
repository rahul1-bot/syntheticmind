from collections.abc import Callable
from enum import Enum
from typing import Annotated, ClassVar, Literal, Protocol, runtime_checkable

import torch
from pydantic import BaseModel, ConfigDict, Field

__all__: list[str] = [
    "DelayedScalingConfiguration",
    "DelayedScalingRecipe"
]

type AmaxHistoryLength = Annotated[int, Field(ge=1)]
type AmaxComputeAlgorithm = Literal["max", "most_recent"]
type FP8Format = Literal["E4M3", "HYBRID"]


class DelayedScalingConfiguration(BaseModel):
    schema_version: Literal[1] = 1
    margin: int = 0
    fp8_format: FP8Format = "HYBRID"
    amax_history_len: AmaxHistoryLength = 1024
    amax_compute_algo: AmaxComputeAlgorithm = "max"
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        validate_default=True
    )


@runtime_checkable
class DelayedScalingRecipe(Protocol):
    margin: int
    fp8_format: Enum
    amax_history_len: int
    amax_compute_algo: str | Callable[[torch.Tensor], torch.Tensor]
