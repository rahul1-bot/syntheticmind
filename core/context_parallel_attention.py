# This module:
# 1. Computes exact sequence-sharded self-attention with differentiable KV gathering
#
# Design decisions:
# - Each rank owns its queries and receives global keys/values for attention
# - Causal positions include the rank's global sequence offset
# - The all-gather implementation is a correctness-oriented backend, not RingAttention
# - The caller supplies contiguous equal sequence shards and synchronizes parameters
#
# Author: Rahul Sawhney

import math
from typing import Literal, cast, override

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.nn.functional import all_gather

__all__: list[str] = ["ContextParallelAttention"]

type _AttentionDescriptor = tuple[
    str | None, tuple[int, ...], tuple[int, ...], tuple[int, ...], str, str
]


class ContextParallelAttention(nn.Module):
    # Construct in Module.configure_model with the Strategy's context mesh.
    # Inputs are local [batch, heads, sequence, features] self-attention shards.
    # The borrowed group is owned by Strategy; this layer owns no parameters.
    def __init__(
        self,
        mesh: DeviceMesh,
        attention_mask: Literal["causal", "bidirectional"] = "causal",
        dropout_p: float = 0.0,
        scale: float | None = None
    ) -> None:
        super().__init__()
        if not isinstance(mesh, DeviceMesh) or mesh.ndim != 1 or mesh.size() < 2:
            raise ValueError("Context attention requires a one-dimensional multi-rank mesh.")
        if attention_mask not in {"causal", "bidirectional"}:
            raise ValueError("attention_mask must be 'causal' or 'bidirectional'.")
        if isinstance(dropout_p, bool) or not isinstance(dropout_p, int | float) or not 0 <= dropout_p < 1:
            raise ValueError("dropout_p must be in [0, 1).")
        if scale is not None and (
            isinstance(scale, bool) or not isinstance(scale, int | float)
            or not math.isfinite(scale) or scale <= 0
        ):
            raise ValueError("Attention scale must be finite and positive.")
        self._group: dist.ProcessGroup = mesh.get_group()
        self._rank: int = mesh.get_local_rank()
        self._world_size: int = mesh.size()
        self._attention_mask: Literal["causal", "bidirectional"] = attention_mask
        self._dropout_p: float = float(dropout_p)
        self._scale: float | None = float(scale) if scale is not None else None

    @override
    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        # All ranks agree input metadata before entering shape-sensitive tensor
        # collectives. Backward reduces gathered KV contributions to their owner.
        self._validate_inputs(query, key, value)
        keys: tuple[torch.Tensor, ...] = cast(
            tuple[torch.Tensor, ...], all_gather(key.contiguous(), group=self._group)
        )
        values: tuple[torch.Tensor, ...] = cast(
            tuple[torch.Tensor, ...], all_gather(value.contiguous(), group=self._group)
        )
        global_key: torch.Tensor = torch.cat(keys, dim=2)
        global_value: torch.Tensor = torch.cat(values, dim=2)
        mask: torch.Tensor | None = None
        if self._attention_mask == "causal":
            positions: torch.Tensor = torch.arange(query.shape[2], device=query.device)
            positions: torch.Tensor = positions + self._rank * query.shape[2]
            key_positions: torch.Tensor = torch.arange(global_key.shape[2], device=query.device)
            mask: torch.Tensor | None = key_positions.unsqueeze(0) <= positions.unsqueeze(1)
        return torch.nn.functional.scaled_dot_product_attention(
            query, global_key, global_value, attn_mask=mask,
            dropout_p=self._dropout_p if self.training else 0.0,
            scale=self._scale
        )

    def _validate_inputs(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        error: str | None = None
        shapes: list[tuple[int, ...]] = []
        tensor: torch.Tensor
        for tensor in (query, key, value):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
                error: str | None = "Context attention expects three rank-four tensors."
                shapes.append(())
            else:
                shapes.append(tuple(tensor.shape))
        if error is None:
            if query.shape != key.shape or value.shape[:3] != key.shape[:3]:
                error: str | None = "Equal query/key self-attention shards and aligned values are required."
            elif not query.is_floating_point() or query.dtype != key.dtype or query.dtype != value.dtype:
                error: str | None = "Attention inputs must share a floating dtype."
            elif query.device != key.device or query.device != value.device:
                error: str | None = "Attention inputs must share their local device."
            elif min(*query.shape, *value.shape) < 1:
                error: str | None = "Attention dimensions must be positive."
        descriptor: _AttentionDescriptor = (
            error, shapes[0], shapes[1], shapes[2],
            str(query.dtype) if isinstance(query, torch.Tensor) else "invalid",
            query.device.type if isinstance(query, torch.Tensor) else "invalid"
        )
        descriptors: list[_AttentionDescriptor | None] = [None] * self._world_size
        dist.all_gather_object(descriptors, descriptor, group=self._group)
        entry: _AttentionDescriptor | None
        if any(entry is None or entry[0] is not None for entry in descriptors):
            raise ValueError(f"Invalid context attention input on a participating rank: {descriptors}.")
        if any(entry != descriptor for entry in descriptors):
            raise ValueError("Context attention requires equal tensor metadata on all ranks.")
