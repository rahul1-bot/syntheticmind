# This module:
# 1. Owns explicit torch.compile configuration and reversible forward bindings
# 2. Preserves model, parameter, hook, and checkpoint identities during a stage
#
# Design decisions:
# - Only selected tensor forwards are compiled; Trainer hooks, optimizer policy,
#   and the temporary DDP step redirection remain outside the compiled region
# - Compilation is lazy. Successful binding is not a successful backend run or
#   a latency measurement; backend execution failures propagate unchanged
# - Stage teardown restores instance bindings without resetting global caches
# - The immutable configuration is shared with checkpoint and manifest records
#
# Author: Rahul Sawhney

from collections.abc import Callable
from typing import Annotated, ClassVar, Literal, cast

import torch
from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator
from torch import nn

from syntheticmind.primitives.types import ModelOutput

__all__: list[str] = ["CompilationConfiguration", "TorchCompiler"]

type CompilationMode = Literal[
    "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
]
type GraphBreakPolicy = Literal["allow", "forbid"]
type ShapePolicy = Literal["auto", "dynamic", "static"]
type _Forward = Callable[..., ModelOutput]
type _ForwardBinding = tuple[nn.Module, _Forward, _Forward, bool]


class CompilationConfiguration(BaseModel):
    # Portable, versioned compiler policy. It contains no compiled callable,
    # model reference, hardware handle, or process-specific cache.
    schema_version: Literal[1] = 1
    backend: Annotated[str, StringConstraints(min_length=1)]
    mode: CompilationMode
    graph_break_policy: GraphBreakPolicy
    shape_policy: ShapePolicy
    module_paths: tuple[str, ...]
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", strict=True, validate_default=True
    )

    @model_validator(mode="after")
    def _validate_regions(self) -> CompilationConfiguration:
        if not self.module_paths or len(set(self.module_paths)) != len(self.module_paths):
            raise ValueError("Compiler module_paths must be non-empty and unique.")
        path: str
        other: str
        part: str
        for path in self.module_paths:
            if path and any(not part for part in path.split(".")):
                raise ValueError(f"Invalid compiler module path: {path!r}.")
            for other in self.module_paths:
                if path != other and (not path or other.startswith(f"{path}.")):
                    raise ValueError("Compiler regions cannot overlap or nest.")
        return self


class TorchCompiler:
    # Strategy owns this stage-scoped binding lifecycle. Standalone clients may
    # call compile(module), execute module(...), then teardown in a finally block.
    # module_paths selects explicit regions; the default is the root forward.
    def __init__(
        self,
        backend: str = "inductor",
        mode: CompilationMode = "default",
        graph_break_policy: GraphBreakPolicy = "allow",
        shape_policy: ShapePolicy = "auto",
        module_paths: tuple[str, ...] | None = None
    ) -> None:
        self._configuration: CompilationConfiguration = CompilationConfiguration(
            backend=backend,
            mode=mode,
            graph_break_policy=graph_break_policy,
            shape_policy=shape_policy,
            module_paths=tuple([""]) if module_paths is None else module_paths
        )
        self._bindings: list[_ForwardBinding] = []
        self._root: nn.Module | None = None

    def compile(self, module: nn.Module) -> nn.Module:
        # Rebinds forwards only after resolving every requested region. Returning
        # the original module avoids _orig_mod checkpoint keys and DDP nesting.
        if not isinstance(module, nn.Module):
            raise TypeError("TorchCompiler requires a torch.nn.Module.")
        if self._root is module:
            return module
        if self._root is not None:
            raise RuntimeError("Compiler is active on another model; tear it down first.")
        path: str
        region: nn.Module
        regions: list[nn.Module] = [
            module.get_submodule(path) for path in self._configuration.module_paths
        ]
        if len({id(region) for region in regions}) != len(regions):
            raise ValueError("Compiler paths must identify distinct module instances.")
        for region in regions:
            if getattr(region, "_compiled_call_impl", None) is not None:
                raise ValueError("A selected region already owns an nn.Module.compile binding.")
        self._root: nn.Module | None = module
        try:
            for region in regions:
                self._bind_forward(region)
        except BaseException as exception:
            try:
                self.teardown()
            except BaseException as cleanup_error:
                exception.add_note(f"Compiler rollback also failed: {cleanup_error}")
            raise
        return module

    def teardown(self) -> None:
        # Removes only bindings owned by this compiler, attempts all restores,
        # and leaves unrelated client mutations visible rather than overwriting.
        failures: list[str] = []
        region: nn.Module
        original: _Forward
        compiled: _Forward
        was_instance_bound: bool
        for region, original, compiled, was_instance_bound in reversed(self._bindings):
            if region.forward is not compiled:
                failures.append(type(region).__qualname__)
                continue
            if was_instance_bound:
                region.forward: _Forward = original
            else:
                delattr(region, "forward")
        self._bindings: list[_ForwardBinding] = []
        self._root: nn.Module | None = None
        if failures:
            raise RuntimeError(f"Client replaced active compiled forwards: {failures}.")

    @property
    def configuration(self) -> CompilationConfiguration:
        return self._configuration

    @property
    def is_active(self) -> bool:
        return self._root is not None

    @property
    def backend(self) -> str:
        return self._configuration.backend

    @property
    def mode(self) -> CompilationMode:
        return self._configuration.mode

    @property
    def graph_break_policy(self) -> GraphBreakPolicy:
        return self._configuration.graph_break_policy

    @property
    def shape_policy(self) -> ShapePolicy:
        return self._configuration.shape_policy

    def _bind_forward(self, region: nn.Module) -> None:
        original: _Forward = cast(_Forward, region.forward)
        compiled: _Forward = cast(
            _Forward,
            torch.compile(
                original,
                fullgraph=self.graph_break_policy == "forbid",
                dynamic=self._dynamic_shapes(),
                backend=self.backend,
                mode=self.mode
            )
        )
        self._bindings.append((region, original, compiled, "forward" in vars(region)))
        region.forward: _Forward = compiled

    def _dynamic_shapes(self) -> bool | None:
        return None if self.shape_policy == "auto" else self.shape_policy == "dynamic"
