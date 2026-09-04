# This module:
# 1. Defines the strict delayed-scaling configuration accepted by Transformer
#    Engine precision
# 2. Defines the structural runtime recipe contract returned by or supplied to
#    NVIDIA Transformer Engine
#
# Design decisions:
# - Configuration admits only verified format and amax algorithm names and a
#   positive history length, rejecting arbitrary recipe dictionaries
# - The live Protocol permits the NVIDIA enum and callable amax forms without
#   importing the optional package at ordinary SyntheticMind startup
#
# Author: Rahul Sawhney

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
    # Immutable client configuration translated by the optional adapter into a
    # live Transformer Engine DelayedScaling recipe.
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
    # Structural shape of an already constructed Transformer Engine recipe.
    margin: int
    fp8_format: Enum
    amax_history_len: int
    amax_compute_algo: str | Callable[[torch.Tensor], torch.Tensor]
