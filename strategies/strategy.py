# This module:
# 1. Defines Strategy, the stateful execution policy connecting Accelerator,
#    model placement, precision, step dispatch, backward, optimizer mutation,
#    collectives, initialization, sampling coordinates, and checkpoint
#    representation
# 2. Defines the wrapped-forward redirection that preserves PyTorch wrapper hooks
#    while invoking SyntheticMind stage-step methods
#
# Design decisions:
# - Strategy is a regular class with complete single-process behavior, not an
#   ABC: every Trainer can call collectives, initialization, checkpoint, and
#   stepping unconditionally, while parallel subclasses override real deltas
# - The client Module and the live execution model remain distinct identities
#   because DDP or model-parallel policy may wrap or transform execution while
#   hooks and client methods still belong to the original Module
# - Wrapped steps enter through wrapper.__call__ so DistributedDataParallel,
#   compiled modules, and registered forward hooks participate; temporary
#   redirection restores the client's true forward before step code calls self
# - Precision owns numerical conversion and optimizer mechanics, while Strategy
#   sequences that policy with model and distributed execution
# - Single-process collectives are identity operations, removing distributed
#   branches from Trainer, callbacks, metrics, and client Module code
# - tensor_init_context composes optional empty initialization, root-device
#   construction, and precision dtype as one reversible model-building boundary
# - Checkpoint representation is delegated to StrategyCheckpoint because full
#   local state and sharded distributed state require different algorithms
#
# Author: Rahul Sawhney

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Literal, cast

import torch
from torch import nn

from syntheticmind.accelerators.accelerator import Accelerator
from syntheticmind.core.module import Module
from syntheticmind.plugins.checkpoint_io import CheckpointIO
from syntheticmind.plugins.precision.precision import Precision
from syntheticmind.strategies.checkpoint import (
    StrategyCheckpoint,
    _SingleFileCheckpoint
)
from syntheticmind.strategies.launchers.launcher import _ProcessLauncher
from syntheticmind.strategies.model_initialization import _ModuleInitializer
from syntheticmind.primitives.types import (
    BackwardInputs,
    Batch,
    CheckpointDict,
    ModelOutput,
    OptimizerClosure,
    OptimizerStepOutput,
    OptimizerStepValue,
    Reduction,
    StepArgument,
    StepOutput,
    TrainingStepOutput
)

__all__: list[str] = ["Strategy"]

type StepName = Literal[
    "predict_step",
    "test_step",
    "training_step",
    "validation_step"
]


class Strategy:
    # Base execution policy with working single-process semantics. Clients select
    # a built-in Strategy through Trainer or supply a concrete instance when a
    # research runtime genuinely changes model wrapping, topology, collectives,
    # or checkpoint representation.
    def __init__(
        self,
        accelerator: Accelerator | None = None,
        parallel_devices: list[torch.device] | None = None,
        checkpoint_io: CheckpointIO | None = None,
        precision_plugin: Precision | None = None
    ) -> None:
        # Records optional preconfigured collaborators, installs single-file
        # checkpointing and full precision defaults, and prepares separate client
        # Module and execution-model slots.
        self.accelerator: Accelerator | None = accelerator
        self.parallel_devices: list[torch.device] = list(parallel_devices or [])
        self.num_nodes: int = 1
        self.is_distributed: bool = False
        self.model: nn.Module | None = None
        self.optimizers: list[torch.optim.Optimizer] = []
        self.checkpoint: StrategyCheckpoint = _SingleFileCheckpoint(checkpoint_io)
        self.precision_plugin: Precision = precision_plugin or Precision()
        self.configured_precision_plugin: Precision | None = precision_plugin
        self._module: Module | None = None
        self._root_device: torch.device = (
            self.parallel_devices[0]
            if self.parallel_devices
            else torch.device("cpu")
        )
        self._forward_redirection: _ForwardRedirection = _ForwardRedirection()

    def configure_topology(
        self,
        accelerator: Accelerator,
        parallel_devices: list[torch.device],
        num_nodes: int
    ) -> None:
        # Binds exactly one device on one node for the base policy. Parallel
        # subclasses override this with multi-process topology validation.
        if num_nodes != 1 or len(parallel_devices) != 1:
            raise ValueError("Single-device execution requires one node and one device.")
        self.accelerator: Accelerator | None = accelerator
        self.parallel_devices: list[torch.device] = list(parallel_devices)
        self.num_nodes: int = num_nodes
        self._root_device: torch.device = parallel_devices[0]

    def connect(self, model: nn.Module) -> None:
        # Attaches the client Module to execution and checkpoint ownership before
        # setup may wrap or transform it.
        if not isinstance(model, Module):
            raise TypeError("Strategy requires a SyntheticMind Module.")
        self._module: Module | None = model
        self.model: nn.Module | None = model
        self.checkpoint.connect_model(model)

    def setup_environment(self) -> None:
        # Prepares process-level resources. Single-device execution owns none.
        pass

    def setup(self, model: nn.Module, device: torch.device) -> nn.Module:
        # Places the model on the selected device and returns the object loops
        # execute. The base policy introduces no wrapper.
        self._root_device: torch.device = device
        placed_model: nn.Module = model.to(device)
        self.model: nn.Module | None = placed_model
        return placed_model

    def training_step(
        self,
        model: nn.Module,
        batch: Batch,
        batch_idx: int
    ) -> TrainingStepOutput:
        # Executes the client's training_step through the live model identity.
        return cast(
            TrainingStepOutput,
            self._execute_step(model, "training_step", batch, batch_idx)
        )

    def validation_step(
        self,
        model: nn.Module,
        *args: StepArgument,
        **kwargs: StepArgument
    ) -> StepOutput:
        # Executes validation_step with optional dataloader coordinates through
        # the live model identity.
        return cast(
            StepOutput,
            self._execute_step(model, "validation_step", *args, **kwargs)
        )

    def test_step(
        self,
        model: nn.Module,
        *args: StepArgument,
        **kwargs: StepArgument
    ) -> StepOutput:
        # Executes test_step through the live model identity.
        return cast(
            StepOutput,
            self._execute_step(model, "test_step", *args, **kwargs)
        )

    def predict_step(
        self,
        model: nn.Module,
        *args: StepArgument,
        **kwargs: StepArgument
    ) -> ModelOutput:
        # Executes predict_step through the live model identity and returns the
        # client output for prediction collection.
        return self._execute_step(model, "predict_step", *args, **kwargs)

    def backward(
        self,
        closure_loss: torch.Tensor,
        optimizer: torch.optim.Optimizer | None = None,
        gradient: torch.Tensor | None = None,
        retain_graph: bool | None = None,
        create_graph: bool = False,
        inputs: BackwardInputs = None
    ) -> torch.Tensor:
        # Runs precision pre-hooks, the overridable Module.backward operation,
        # and precision post-hooks, returning the detached reporting loss selected
        # by the precision policy.
        module: Module = self._require_module()
        prepared_loss: torch.Tensor = self.precision_plugin.pre_backward(
            closure_loss,
            module
        )
        self.precision_plugin.backward(
            prepared_loss,
            module,
            optimizer,
            gradient,
            retain_graph,
            create_graph,
            inputs
        )
        return self.precision_plugin.post_backward(prepared_loss, module)

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        closure: OptimizerClosure,
        model: nn.Module | None = None,
        **kwargs: OptimizerStepValue
    ) -> OptimizerStepOutput:
        # Delegates one managed optimizer mutation to precision with the connected
        # Module and closure produced by the optimization loop.
        resolved_model: nn.Module | None = model if model is not None else self._module
        if not isinstance(resolved_model, Module):
            raise TypeError("Strategy optimizer_step requires a SyntheticMind Module.")
        return cast(
            OptimizerStepOutput,
            self.precision_plugin.optimizer_step(
                optimizer,
                model=resolved_model,
                closure=closure,
                **kwargs
            )
        )

    def teardown(self) -> None:
        # Releases Strategy-owned runtime resources; the base policy owns none.
        pass

    def on_exception(self, exception: BaseException) -> None:
        # Performs exception-time Strategy cleanup; parallel policies destroy
        # process resources here before outer teardown continues.
        del exception

    @contextmanager
    def tensor_init_context(
        self,
        empty_init: bool | None = None
    ) -> Iterator[None]:
        # Combines optional torch initialization suppression, root-device tensor
        # construction, and precision default dtype. Clients use Trainer.init_module
        # rather than constructing this context directly.
        if empty_init is not None and type(empty_init) is not bool:
            raise TypeError("`empty_init` must be a bool or None.")
        with (
            _ModuleInitializer(enabled=bool(empty_init)),
            self.root_device,
            self.precision_plugin.tensor_init_context()
        ):
            yield

    def barrier(self) -> None:
        # Single-process synchronization boundary requiring no operation.
        pass

    def broadcast[T](self, value: T, source_rank: int = 0) -> T:
        # Single-process broadcast identity preserving the original value type.
        del source_rank
        return value

    def reduce_boolean_decision(
        self,
        decision: bool,
        all: bool = True
    ) -> bool:
        # Single-process all-rank or any-rank decision identity; ``all`` remains
        # validated so local and parallel call contracts match.
        if type(all) is not bool:
            raise TypeError("`all` must be a bool.")
        return decision

    def reduce(
        self,
        tensor: torch.Tensor,
        group: torch.distributed.ProcessGroup | None = None,
        reduce_op: Reduction = "mean"
    ) -> torch.Tensor:
        # Single-process scalar/tensor reduction identity with closed operation
        # validation.
        if reduce_op not in ("mean", "sum", "min", "max"):
            raise ValueError(
                f"Unsupported reduction {reduce_op!r}; expected mean, sum, min, or max."
            )
        del group
        return tensor

    def all_gather(
        self,
        tensor: torch.Tensor,
        group: torch.distributed.ProcessGroup | None = None,
        sync_grads: bool = False
    ) -> torch.Tensor:
        # Single-process gather identity that adds no leading rank dimension.
        del group, sync_grads
        return tensor

    def _gather_checkpoint_states(
        self,
        state: CheckpointDict
    ) -> list[CheckpointDict]:
        # Wraps rank-local partial-epoch state in the same list shape produced by
        # parallel all-gather.
        return [state]

    def sync_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        # Synchronizes deferred tail gradients. Single-device execution has no
        # replicas and therefore no operation.
        del optimizer

    def defer_backward_sync(self) -> AbstractContextManager[None]:
        # Context suppressing distributed synchronization during accumulation;
        # local execution returns a no-op context.
        return nullcontext()

    def attach_precision_plugin(self, precision_plugin: Precision) -> None:
        # Installs the validated numerical policy selected by the Trainer
        # accelerator connector.
        if not isinstance(precision_plugin, Precision):
            raise TypeError("precision_plugin must be a Precision instance.")
        self.precision_plugin: Precision = precision_plugin

    @property
    def distributed_sampler_kwargs(self) -> dict[str, int] | None:
        # Replica count and rank for DataConnector sampler replacement; local
        # execution needs no distributed sampler.
        return None

    @property
    def launcher(self) -> _ProcessLauncher | None:
        # Local execution creates no worker-launch boundary.
        return None

    @property
    def num_processes(self) -> int:
        # Local execution owns exactly one process.
        return 1

    @property
    def global_rank(self) -> int:
        # Local execution is the sole global rank.
        return 0

    @property
    def local_rank(self) -> int:
        # Local execution is the sole rank on its host.
        return 0

    @property
    def node_rank(self) -> int:
        # Local execution is the sole node coordinate.
        return 0

    @property
    def world_size(self) -> int:
        # Local collective identity contains one participant.
        return 1

    @property
    def data_parallel_size(self) -> int:
        # One local process owns one sample-processing replica.
        return 1

    @property
    def checkpoint_requires_all_ranks(self) -> bool:
        # Whether every rank must participate in checkpoint representation.
        return False

    @property
    def is_global_zero(self) -> bool:
        # Whether this process owns singleton presentation and persistence effects.
        return self.global_rank == 0

    @property
    def root_device(self) -> torch.device:
        # Process-local model and batch placement selected during topology setup.
        return self._root_device

    def _execute_step(
        self,
        model: nn.Module,
        step_name: StepName,
        *args: StepArgument,
        **kwargs: StepArgument
    ) -> ModelOutput:
        # Routes the client step through wrapped-forward redirection when the live
        # execution model differs from the Module identity.
        return self._forward_redirection.execute(
            model,
            self._require_module(),
            step_name,
            *args,
            **kwargs
        )

    def _require_module(self) -> Module:
        # Resolves the connected client Module before backward or step dispatch.
        if self._module is None:
            raise RuntimeError("Strategy has no connected Module.")
        return self._module


class _ForwardRedirection:
    # Transactional bridge between a PyTorch wrapper's __call__ path and a client
    # Module stage method. It owns only one active step and restores every
    # temporary forward binding in finally.
    def __init__(self) -> None:
        self._active: bool = False
        self._module: Module | None = None
        self._original_forward: Callable[..., ModelOutput] | None = None
        self._step: Callable[..., ModelOutput] | None = None
        self._wrapper: nn.Module | None = None

    def execute(
        self,
        wrapper: nn.Module,
        module: Module,
        step_name: StepName,
        *args: StepArgument,
        **kwargs: StepArgument
    ) -> ModelOutput:
        # Invokes the step directly for an unwrapped model or arms temporary
        # redirection, calls the wrapper, executes specialization hooks, and
        # restores all identities even when client code raises.
        step: Callable[..., ModelOutput] = cast(
            Callable[..., ModelOutput],
            getattr(module, step_name)
        )
        if wrapper is module:
            return step(*args, **kwargs)
        if self._active:
            raise RuntimeError("A wrapped module step is already active.")
        original_forward: Callable[..., ModelOutput] = cast(
            Callable[..., ModelOutput],
            module.forward
        )
        self._active: bool = True
        self._module: Module | None = module
        self._original_forward: Callable[..., ModelOutput] | None = original_forward
        self._step: Callable[..., ModelOutput] | None = step
        self._wrapper: nn.Module | None = wrapper
        module.forward: Callable[..., ModelOutput] = self._redirected_forward
        try:
            output: ModelOutput = wrapper(*args, **kwargs)
            self._after_outer_forward(wrapper, module)
            return output
        finally:
            module.forward: Callable[..., ModelOutput] = original_forward
            self._active: bool = False
            self._module: Module | None = None
            self._original_forward: Callable[..., ModelOutput] | None = None
            self._step: Callable[..., ModelOutput] | None = None
            self._wrapper: nn.Module | None = None

    def _redirected_forward(
        self,
        *args: StepArgument,
        **kwargs: StepArgument
    ) -> ModelOutput:
        # Restores the true model forward before entering the stage step, allowing
        # self(...) inside client code to execute actual model computation rather
        # than recurse into redirection.
        module: Module | None = self._module
        original_forward: Callable[..., ModelOutput] | None = self._original_forward
        step: Callable[..., ModelOutput] | None = self._step
        wrapper: nn.Module | None = self._wrapper
        if module is None or original_forward is None or step is None or wrapper is None:
            raise RuntimeError("Wrapped module step redirection is not active.")
        module.forward: Callable[..., ModelOutput] = original_forward
        output: ModelOutput = step(*args, **kwargs)
        self._after_inner_forward(wrapper, module)
        return output

    def _after_inner_forward(self, wrapper: nn.Module, module: Module) -> None:
        # Default wrappers require no mutation after the redirected client step.
        del wrapper, module

    def _after_outer_forward(self, wrapper: nn.Module, module: Module) -> None:
        # Default wrappers require no mutation after their outer call completes.
        del wrapper, module
