# This module:
# 1. Answers rank, local-rank, world-size, and is-distributed queries with safe
#    single-process fallbacks whenever no process group is initialized
# 2. Wraps the barrier and all-reduce collectives so call sites need no
#    initialization guards of their own
#
# Design decisions:
# - Every query degrades to single-process values (rank 0, local rank 0, world
#   size 1) so identical call sites work in local and distributed execution
#   without branching
# - The node-local rank is read from the LOCAL_RANK environment variable because
#   torch.distributed exposes only the global rank; the torchrun launcher is the
#   authority that sets it
# - is_distributed additionally requires world size above one, so a one-process
#   group behaves exactly like plain local execution
# - all_reduce follows the collective's in-place semantics and returns the same
#   tensor for chaining; the single-process path returns the input untouched
# - Module-level function aliases mirror the classmethod namespace so importing
#   code can bind bare functions; both routes reach the same implementations
#
# Author: Rahul Sawhney

import os

import torch
import torch.distributed as dist

__all__: list[str] = [
    "get_rank",
    "get_local_rank",
    "get_world_size",
    "is_distributed",
    "is_rank_zero",
    "barrier",
    "all_reduce"
]


class DistributedUtils:
    # Classmethod namespace over torch.distributed. Every member first checks
    # whether a process group is available and initialized, so the harness can
    # call these helpers unconditionally from any execution mode.

    @classmethod
    def get_rank(cls) -> int:
        # Returns the global rank of this process, or zero when no process
        # group is initialized so single-process runs behave as rank zero.
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    @classmethod
    def get_local_rank(cls) -> int:
        # Returns the node-local rank from the LOCAL_RANK environment variable
        # set by the launcher; falls back to zero when the variable is absent
        # or no process group is initialized.
        if dist.is_available() and dist.is_initialized():
            local_rank: str | None = os.environ.get("LOCAL_RANK")
            if local_rank is not None:
                return int(local_rank)
        return 0

    @classmethod
    def get_world_size(cls) -> int:
        # Returns the number of participating processes, or one when no process
        # group is initialized.
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    @classmethod
    def is_distributed(cls) -> bool:
        # True only for an initialized process group with more than one member,
        # so a degenerate one-process group is treated as local execution.
        return dist.is_available() and dist.is_initialized() and cls.get_world_size() > 1

    @classmethod
    def is_rank_zero(cls) -> bool:
        # True on the coordinating process; single-process runs are always rank zero.
        return cls.get_rank() == 0

    @classmethod
    def barrier(cls) -> None:
        # Blocks until every process reaches this point; a no-op in local execution
        # so synchronization points need no caller-side guards.
        if cls.is_distributed():
            dist.barrier()

    @classmethod
    def all_reduce(
        cls,
        tensor: torch.Tensor,
        op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM,
        group: dist.ProcessGroup | None = None
    ) -> torch.Tensor:
        # Applies the reduction collective in place across all ranks and returns
        # the same tensor for chaining; local execution returns it untouched.
        if not cls.is_distributed():
            return tensor
        dist.all_reduce(tensor, op=op, group=group)  # type: ignore[arg-type]
        return tensor

    @classmethod
    def all_reduce_mean(
        cls,
        tensor: torch.Tensor,
        group: dist.ProcessGroup | None = None
    ) -> torch.Tensor:
        # Computes the cross-rank mean by summing in place and dividing by the
        # world size; local execution returns the tensor untouched.
        if not cls.is_distributed():
            return tensor
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        reduced_mean: torch.Tensor = tensor / dist.get_world_size(group)
        return reduced_mean


def get_rank() -> int:
    # Function alias for DistributedUtils.get_rank.
    return DistributedUtils.get_rank()


def get_local_rank() -> int:
    # Function alias for DistributedUtils.get_local_rank.
    return DistributedUtils.get_local_rank()


def get_world_size() -> int:
    # Function alias for DistributedUtils.get_world_size.
    return DistributedUtils.get_world_size()


def is_distributed() -> bool:
    # Function alias for DistributedUtils.is_distributed.
    return DistributedUtils.is_distributed()


def is_rank_zero() -> bool:
    # Function alias for DistributedUtils.is_rank_zero.
    return DistributedUtils.is_rank_zero()


def barrier() -> None:
    # Function alias for DistributedUtils.barrier.
    DistributedUtils.barrier()


def all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM,
    group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    # Function alias for DistributedUtils.all_reduce.
    return DistributedUtils.all_reduce(tensor, op=op, group=group)
