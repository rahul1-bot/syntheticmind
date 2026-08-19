# This module:
# 1. Implements the strategy contract for multi-process data-parallel execution
#    on top of torch.distributed and DistributedDataParallel
# 2. Validates the launch environment, initializes and destroys the process
#    group, and wraps the model for gradient-synchronized training
# 3. Overrides the collective operations (barrier, broadcast, boolean decision
#    reduction, all-gather) with real torch.distributed implementations
#
# Design decisions:
# - setup_environment validates its preconditions before touching the process
#   group: the NCCL backend is refused when CUDA is unavailable, and the four
#   launcher-provided environment variables (RANK, WORLD_SIZE, MASTER_ADDR,
#   MASTER_PORT) are required so a missing torchrun launch fails with a precise
#   configuration error instead of a hang inside init_process_group
# - The training step executes through the DistributedDataParallel wrapper
#   rather than the unwrapped module, because gradient bucketing hooks fire only
#   when the wrapper's own forward path runs. The strategy temporarily redirects
#   the wrapped module's forward to its training_step, invokes the wrapper, and
#   restores the original forward in a finally block, guaranteeing restoration
#   even when the step raises
# - Evaluation, test, and prediction steps deliberately bypass the wrapper and
#   call the unwrapped module directly, because no gradients exist in those
#   stages and wrapper synchronization would add cost without effect
# - reduce_boolean_decision reduces with a logical-OR convention, implemented as
#   an integer sum compared against zero, so any single rank requesting an
#   action (such as early stopping) commits every rank to it and no rank
#   continues alone
# - teardown destroys the process group so consecutive runs inside one process
#   can re-initialize cleanly
#
# Author: Rahul Sawhney

from __future__ import annotations

import atexit
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, cast, override

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from syntheticmind.core.module import Module
from syntheticmind.plugins.environments import ClusterEnvironment
from syntheticmind.plugins.io import CheckpointIO
from syntheticmind.plugins.layer_sync import LayerSync
from syntheticmind.precision.precision import Precision
from syntheticmind.strategies.launcher import _ProcessLauncher, ProcessStartMethod
from syntheticmind.strategies.strategy import Strategy, TBroadcast
from syntheticmind.utilities.types import Batch, ModelOutput, StepOutput

if TYPE_CHECKING:
    from syntheticmind.accelerators.accelerator import Accelerator

__all__: list[str] = ["DDPStrategy"]


class DDPStrategy(Strategy):
    # Distributed data-parallel strategy. Each participating process owns one
    # device and one replica; this class manages the process-group lifecycle,
    # the DistributedDataParallel wrapping, and the collective operations the
    # trainer and callbacks invoke through the Strategy interface.
    def __init__(
        self,
        accelerator: Accelerator | None = None,
        parallel_devices: list[torch.device] | None = None,
        cluster_environment: ClusterEnvironment | None = None,
        checkpoint_io: CheckpointIO | None = None,
        precision_plugin: Precision | None = None,
        process_group_backend: str | None = None,
        start_method: ProcessStartMethod = "popen",
        layer_sync: LayerSync | None = None
    ) -> None:
        # Records the requested communication backend. The local rank is
        # resolved later, during environment setup, from the launcher-provided
        # environment variables.
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
            layer_sync=layer_sync
        )
        if process_group_backend is not None and not isinstance(
            process_group_backend,
            str
        ):
            raise TypeError(
                f"`process_group_backend` must be a string or None, got {process_group_backend!r}."
            )
        self._process_group_backend: str | None = (
            process_group_backend.lower()
            if process_group_backend is not None
            else None
        )
        self._launcher: _ProcessLauncher = _ProcessLauncher(self, start_method)
        self._local_rank: int = 0
        self._process_group_cleanup_registered: bool = False

    @override
    def setup_environment(self) -> None:
        # Initializes the process group exactly once per process. The method
        # first validates that the backend is usable and that the launcher
        # environment is complete, so misconfiguration surfaces as an immediate
        # MisconfigurationError with corrective guidance rather than a
        # rendezvous hang inside torch.distributed.
        from syntheticmind.utilities.exceptions import MisconfigurationError

        self.cluster_environment.validate_settings(
            self.num_processes,
            self.num_nodes
        )
        self._local_rank = self.cluster_environment.local_rank()
        if self._local_rank < 0 or self._local_rank >= self.num_processes:
            raise MisconfigurationError(
                f"LOCAL_RANK={self._local_rank} is outside the configured per-node device "
                f"range [0, {self.num_processes - 1}]."
            )
        backend: str = self._process_group_backend or (
            "nccl" if self.root_device.type == "cuda" else "gloo"
        )
        configured_world_size: int = self.num_nodes * self.num_processes
        launched_world_size: int = self.cluster_environment.world_size()
        global_rank: int = self.cluster_environment.global_rank()
        node_rank: int = self.cluster_environment.node_rank()
        if global_rank < 0 or global_rank >= launched_world_size:
            raise MisconfigurationError(
                f"global_rank={global_rank} is outside the launched world range "
                f"[0, {launched_world_size - 1}]."
            )
        if launched_world_size != configured_world_size:
            raise MisconfigurationError(
                f"world_size={launched_world_size} does not match the configured topology "
                f"of num_nodes={self.num_nodes} and num_devices={self.num_processes} "
                f"({configured_world_size} processes)."
            )
        if node_rank < 0 or node_rank >= self.num_nodes:
            raise MisconfigurationError(
                f"node_rank={node_rank} is outside the configured node range "
                f"[0, {self.num_nodes - 1}]."
            )
        expected_global_rank: int = node_rank * self.num_processes + self._local_rank
        if expected_global_rank != global_rank:
            raise MisconfigurationError(
                f"global_rank={global_rank} does not match node_rank={node_rank} and "
                f"local_rank={self._local_rank}; expected global_rank={expected_global_rank}."
            )
        if dist.is_initialized():
            active_world_size: int = dist.get_world_size()
            active_backend: str = str(dist.get_backend()).lower()
            if active_world_size != configured_world_size:
                raise MisconfigurationError(
                    f"The initialized process group has world_size={active_world_size}, but the "
                    f"configured topology requires {configured_world_size} processes."
                )
            if active_backend != backend:
                raise MisconfigurationError(
                    f"The initialized process group uses backend={active_backend!r}, but "
                    f"DDPStrategy requires backend={backend!r}."
                )
            self._register_process_group_cleanup()
            return
        if backend == "nccl" and not torch.cuda.is_available():
            raise MisconfigurationError(
                "DDPStrategy with process_group_backend='nccl' requires CUDA, but CUDA is not available. "
                "Use DDPStrategy(process_group_backend='gloo') for CPU distributed training, "
                "or ensure CUDA is properly installed."
            )

        os.environ["MASTER_ADDR"] = self.cluster_environment.main_address
        os.environ["MASTER_PORT"] = str(self.cluster_environment.main_port)
        os.environ["WORLD_SIZE"] = str(launched_world_size)
        os.environ["RANK"] = str(global_rank)
        os.environ["LOCAL_RANK"] = str(self._local_rank)
        dist.init_process_group(
            backend=backend,
            rank=global_rank,
            world_size=launched_world_size
        )
        self._register_process_group_cleanup()

    @override
    def setup(self, model: nn.Module, device: torch.device) -> nn.Module:
        # Places the replica on this process's device and wraps it in
        # DistributedDataParallel. The device index is passed to the wrapper
        # only for CUDA devices, because the CPU-based gloo path expects no
        # device_ids argument.
        self._device: torch.device = device
        self._module: nn.Module | None = model
        placed_model: nn.Module = model.to(device)
        self._model: nn.Module | None = DistributedDataParallel(
            placed_model,
            device_ids=[device.index] if device.type == "cuda" else None
        )
        return self._model

    @override
    def training_step(
        self, model: nn.Module, batch: Batch, batch_idx: int
    ) -> StepOutput:
        # Executes the training step through the DistributedDataParallel
        # forward path so gradient-bucketing hooks participate in the step.
        # The wrapped module's forward attribute is temporarily redirected to a
        # closure that restores the original forward and invokes training_step;
        # calling the wrapper then routes through that closure while the hooks
        # remain armed. The finally block guarantees the original forward is
        # restored even when the step raises.
        if isinstance(model, DistributedDataParallel):
            unwrapped: Module = model.module  # type: ignore[assignment]
            original_forward: Callable[..., object] = unwrapped.forward

            def redirected_forward(batch: Batch, batch_idx: int) -> StepOutput:
                # Restores the module's original forward before delegating, so
                # any forward call made inside training_step reaches the true
                # model computation rather than this redirection closure.
                unwrapped.forward: Callable[..., object] = original_forward  # type: ignore[assignment]
                return unwrapped.training_step(batch, batch_idx)

            unwrapped.forward: Callable[..., object] = redirected_forward  # type: ignore[assignment]
            try:
                output: StepOutput = model(batch, batch_idx)
            finally:
                unwrapped.forward: Callable[..., object] = original_forward  # type: ignore[assignment]
            return output
        if not isinstance(model, Module):
            return cast(
                StepOutput,
                self._forward_redirection(
                    model,
                    "training_step",
                    batch,
                    batch_idx
                )
            )
        return model.training_step(batch, batch_idx)

    @override
    def validation_step(
        self, model: nn.Module, batch: Batch, batch_idx: int
    ) -> StepOutput:
        # Executes the validation step on the unwrapped module. Gradient
        # synchronization is irrelevant without a backward pass, so the
        # DistributedDataParallel forward path is intentionally bypassed.
        unwrapped: nn.Module = model.module if isinstance(model, DistributedDataParallel) else model
        if not isinstance(unwrapped, Module):
            return cast(
                StepOutput,
                self._forward_redirection(
                    model,
                    "validation_step",
                    batch,
                    batch_idx
                )
            )
        return unwrapped.validation_step(batch, batch_idx)

    @override
    def test_step(
        self, model: nn.Module, batch: Batch, batch_idx: int
    ) -> StepOutput:
        # Executes the test step on the unwrapped module, mirroring the
        # validation-step rationale for bypassing the wrapper.
        unwrapped: nn.Module = model.module if isinstance(model, DistributedDataParallel) else model
        if not isinstance(unwrapped, Module):
            return cast(
                StepOutput,
                self._forward_redirection(
                    model,
                    "test_step",
                    batch,
                    batch_idx
                )
            )
        return unwrapped.test_step(batch, batch_idx)

    @override
    def predict_step(
        self, model: nn.Module, batch: Batch, batch_idx: int
    ) -> ModelOutput:
        # Executes the prediction step on the unwrapped module, mirroring the
        # validation-step rationale for bypassing the wrapper.
        unwrapped: nn.Module = model.module if isinstance(model, DistributedDataParallel) else model
        if not isinstance(unwrapped, Module):
            return cast(
                ModelOutput,
                self._forward_redirection(
                    model,
                    "predict_step",
                    batch,
                    batch_idx
                )
            )
        return unwrapped.predict_step(batch, batch_idx)

    @override
    def backward(
        self,
        closure_loss: torch.Tensor,
        optimizer: torch.optim.Optimizer | None = None,
        *args: object,
        **kwargs: object
    ) -> torch.Tensor:
        # Executes the standard autograd backward pass. Gradient averaging
        # across ranks is performed by the DistributedDataParallel hooks armed
        # during the training-step forward, not by this method.
        return self._precision_backward(closure_loss, optimizer, *args, **kwargs)

    @override
    def barrier(self) -> None:
        # Blocks until every rank reaches this synchronization point. The
        # initialization guard keeps the method callable before setup.
        if dist.is_initialized():
            dist.barrier()

    @override
    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        # Broadcasts the tensor from the source rank to all ranks in place and
        # returns it, so call sites can use the result directly.
        if not dist.is_initialized():
            return obj
        objects: list[object] = [obj]
        dist.broadcast_object_list(
            objects,
            src=src,
            device=(self.root_device if self.root_device.type == "cuda" else None)
        )
        return cast(TBroadcast, objects[0])

    @property
    @override
    def is_distributed(self) -> bool:
        # This strategy always coordinates multiple processes.
        return True

    @property
    @override
    def launcher(self) -> _ProcessLauncher:
        return self._launcher

    @property
    @override
    def root_device(self) -> torch.device:
        if not self.parallel_devices:
            return self._device
        return self.parallel_devices[self.local_rank]

    @property
    @override
    def local_rank(self) -> int:
        # Node-local rank resolved from the launcher environment during
        # environment setup. Zero before setup and in single-node launches
        # without an explicit LOCAL_RANK.
        return self._local_rank

    @property
    @override
    def global_rank(self) -> int:
        if dist.is_initialized():
            return dist.get_rank()
        return self.cluster_environment.global_rank()

    @property
    @override
    def node_rank(self) -> int:
        return self.cluster_environment.node_rank()

    @property
    @override
    def world_size(self) -> int:
        # Number of participating processes, or one before the process group
        # is initialized.
        if dist.is_initialized():
            return dist.get_world_size()
        return self.cluster_environment.world_size()

    @property
    def process_group_backend(self) -> str | None:
        return self._process_group_backend

    @property
    def start_method(self) -> ProcessStartMethod:
        return self._launcher.start_method

    @override
    def reduce_boolean_decision(self, decision: bool) -> bool:
        # Agrees on a boolean control decision across ranks with logical-OR
        # semantics. The local decision is encoded as an integer, summed across
        # ranks, and compared against zero, so a request raised by any single
        # rank commits every rank to the same outcome.
        if not dist.is_initialized():
            return decision
        decision_tensor: torch.Tensor = torch.tensor(
            int(decision),
            device=self.root_device
        )
        dist.all_reduce(decision_tensor, op=dist.ReduceOp.SUM)
        return bool(decision_tensor.item() > 0)

    @override
    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        # Gathers the tensor from every rank into a new leading dimension. A
        # zero-initialized buffer is allocated per rank and filled by the
        # collective, and the buffers are stacked so the caller receives one
        # tensor of shape (world_size, *tensor.shape).
        if not dist.is_initialized():
            return tensor
        gathered: list[torch.Tensor] = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor)
        return torch.stack(gathered)

    @override
    def sync_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        if not dist.is_initialized():
            return
        synchronized_parameters: set[int] = set()
        for parameter_group in optimizer.param_groups:
            parameters: object = parameter_group["params"]
            if not isinstance(parameters, list):
                raise TypeError("Optimizer parameter groups must contain a parameter list.")
            for parameter in parameters:
                if not isinstance(parameter, torch.nn.Parameter):
                    raise TypeError("Optimizer parameter groups must contain Parameters.")
                parameter_identity: int = id(parameter)
                if parameter_identity in synchronized_parameters:
                    continue
                gradient_flags: torch.Tensor = torch.tensor(
                    [
                        int(parameter.grad is not None),
                        int(parameter.grad is not None and parameter.grad.is_sparse)
                    ],
                    device=parameter.device,
                    dtype=torch.int64
                )
                dist.all_reduce(gradient_flags, op=dist.ReduceOp.SUM)
                if int(gradient_flags[1].item()) > 0:
                    raise RuntimeError(
                        "Distributed accumulation tails with sparse gradients are not supported."
                    )
                if int(gradient_flags[0].item()) == 0:
                    synchronized_parameters.add(parameter_identity)
                    continue
                gradient: torch.Tensor = (
                    parameter.grad
                    if parameter.grad is not None
                    else torch.zeros_like(parameter)
                )
                dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
                gradient.div_(self.world_size)
                if parameter.grad is None:
                    parameter.grad = gradient
                synchronized_parameters.add(parameter_identity)

    @override
    def on_exception(self, exception: BaseException) -> None:
        self._destroy_process_group()

    @override
    def teardown(self) -> None:
        # Destroys the process group so a subsequent run inside the same
        # process can initialize a fresh group without conflict.
        if isinstance(self._model, DistributedDataParallel):
            self._model = self._model.module

    def _register_process_group_cleanup(self) -> None:
        if self._process_group_cleanup_registered:
            return
        atexit.register(self._destroy_process_group)
        self._process_group_cleanup_registered = True

    def _destroy_process_group(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
