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

import hashlib
import os
import tempfile
from collections import deque
from collections.abc import Callable
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Annotated, ClassVar, Literal, cast

import torch
import torch._inductor
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from torch import nn
from torch._dynamo import config as dynamo_configuration
from torch._dynamo.backends.registry import lookup_backend
from torch._dynamo.output_graph import GraphCompileReason
from torch._guards import CompileContext, CompileId

from syntheticmind.primitives.types import ModelOutput

__all__: list[str] = ["CompilationConfiguration", "CompilationEvent", "CompilationReport", "TorchCompiler"]

type CompilationMode = Literal[
    "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
]
type GraphBreakPolicy = Literal["allow", "forbid"]
type ShapePolicy = Literal["auto", "dynamic", "static"]
type _Forward = Callable[..., ModelOutput]
type _GraphInput = torch.Tensor | torch.SymInt | torch.SymFloat | torch.SymBool | int | float | bool
type _GraphBackend = Callable[[torch.fx.GraphModule, list[_GraphInput]], _Forward]
type _CompilerOption = bool | int | float | str | None
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


class CompilationEvent(BaseModel):
    # A completed backend attempt, not total Dynamo capture or lazy backward time.
    # Null native frame IDs indicate unavailable metadata, not a first compile.
    region_path: str
    frame_id: Annotated[int, Field(ge=0)] | None
    frame_compile_id: Annotated[int, Field(ge=0)] | None
    graph_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    graph_break_reason: str | None
    backend_duration_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    failure: str | None
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", strict=True, validate_default=True
    )


class CompilationReport(BaseModel):
    # Bounded stage-local observations. A cache hit may produce no backend events;
    # absence of an event is not proof that execution never entered eager Python.
    schema_version: Literal[1] = 1
    torch_version: str
    configuration: CompilationConfiguration
    events: tuple[CompilationEvent, ...]
    dropped_event_count: Annotated[int, Field(ge=0)]
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", strict=True, validate_default=True
    )


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
        module_paths: tuple[str, ...] | None = None,
        *,
        diagnostics_limit: int = 256
    ) -> None:
        if type(diagnostics_limit) is not int or diagnostics_limit < 1:
            raise ValueError("diagnostics_limit must be a positive integer.")
        self._configuration: CompilationConfiguration = CompilationConfiguration(
            backend=backend,
            mode=mode,
            graph_break_policy=graph_break_policy,
            shape_policy=shape_policy,
            module_paths=tuple([""]) if module_paths is None else module_paths
        )
        self._bindings: list[_ForwardBinding] = []
        self._root: nn.Module | None = None
        self._events: deque[CompilationEvent] = deque(maxlen=diagnostics_limit)
        self._dropped_event_count: int = 0
        path: str
        self._backends: dict[str, _GraphBackend] = {
            path: partial(self._compile_graph, path) for path in self._configuration.module_paths
        }

    def compile(self, module: nn.Module) -> nn.Module:
        # Rebinds forwards only after resolving every requested region. Returning
        # the original module avoids _orig_mod checkpoint keys and DDP nesting.
        if not isinstance(module, nn.Module):
            raise TypeError("TorchCompiler requires a torch.nn.Module.")
        if self._root is module:
            return module
        if self._root is not None:
            raise RuntimeError("Compiler is active on another model; tear it down first.")
        if dynamo_configuration.suppress_errors:
            raise RuntimeError("Disable torch._dynamo.config.suppress_errors before selecting TorchCompiler.")
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
        self._events.clear()
        self._dropped_event_count: int = 0
        self._root: nn.Module | None = module
        try:
            for path, region in zip(self._configuration.module_paths, regions, strict=True):
                self._bind_forward(path, region)
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

    def save_report(self, path: Path) -> None:
        # Explicit local publication; no graph code, tensor payloads, or automatic
        # uploads. Call after the stage to include all observed backend attempts.
        destination: Path = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, delete=False) as stream:
                temporary: Path | None = Path(stream.name)
                stream.write(self.report.model_dump_json(indent=2))
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @property
    def report(self) -> CompilationReport:
        return CompilationReport(
            torch_version=str(torch.__version__), configuration=self._configuration,
            events=tuple(self._events), dropped_event_count=self._dropped_event_count
        )

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

    def _bind_forward(self, path: str, region: nn.Module) -> None:
        original: _Forward = cast(_Forward, region.forward)
        compiled: _Forward = cast(
            _Forward,
            torch.compile(
                original,
                fullgraph=self.graph_break_policy == "forbid",
                dynamic=self._dynamic_shapes(),
                backend=self._backends[path]
            )
        )
        self._bindings.append((region, original, compiled, "forward" in vars(region)))
        region.forward: _Forward = compiled

    def _compile_graph(
        self, path: str, graph: torch.fx.GraphModule, example_inputs: list[_GraphInput]
    ) -> _Forward:
        # Observe only this compiler's backend boundary, not global Dynamo counters.
        identity: CompileId | None = CompileContext.current_compile_id()
        reason: GraphCompileReason | None = getattr(graph, "compile_subgraph_reason", None)
        fingerprint: str = hashlib.sha256(str(graph.graph).encode("utf-8")).hexdigest()
        started: float = perf_counter()
        failure: str | None = None
        try:
            if self.backend == "inductor":
                options: dict[str, _CompilerOption] = cast(
                    dict[str, _CompilerOption],
                    torch._inductor.list_mode_options(self.mode, self._dynamic_shapes())
                )
                result: _Forward = cast(_Forward, torch._inductor.compile(graph, example_inputs, options=options))
            elif self.mode == "default":
                result: _Forward = cast(_Forward, lookup_backend(self.backend)(graph, example_inputs))
            else:
                result: _Forward = cast(_Forward, lookup_backend(self.backend)(graph, example_inputs, mode=self.mode))
            if not callable(result):
                raise TypeError(f"Compiler backend {self.backend!r} did not return executable code.")
            return result
        except BaseException as exception:
            failure: str | None = f"{type(exception).__name__}: {exception}"
            raise
        finally:
            event: CompilationEvent = CompilationEvent(
                region_path=path,
                frame_id=identity.frame_id if identity is not None else None,
                frame_compile_id=identity.frame_compile_id if identity is not None else None,
                graph_sha256=fingerprint,
                graph_break_reason=reason.reason if isinstance(reason, GraphCompileReason) and reason.graph_break else None,
                backend_duration_seconds=perf_counter() - started, failure=failure
            )
            if len(self._events) == self._events.maxlen:
                self._dropped_event_count: int = self._dropped_event_count + 1
            self._events.append(event)

    def _dynamic_shapes(self) -> bool | None:
        return None if self.shape_policy == "auto" else self.shape_policy == "dynamic"
