from typing import Literal, cast

import torch
from torch import nn

__all__: list[str] = ["TorchCompiler"]

type CompilationMode = Literal[
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs"
]
type GraphBreakPolicy = Literal["allow", "forbid"]
type ShapePolicy = Literal["auto", "dynamic", "static"]


class TorchCompiler:
    def __init__(
        self,
        backend: str = "inductor",
        mode: CompilationMode = "default",
        graph_break_policy: GraphBreakPolicy = "allow",
        shape_policy: ShapePolicy = "auto"
    ) -> None:
        if not backend:
            raise ValueError("Compiler backend must be a non-empty string.")
        if mode not in {
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs"
        }:
            raise ValueError(f"Unsupported torch.compile mode: {mode!r}.")
        if graph_break_policy not in {"allow", "forbid"}:
            raise ValueError(
                f"Unsupported graph break policy: {graph_break_policy!r}."
            )
        if shape_policy not in {"auto", "dynamic", "static"}:
            raise ValueError(f"Unsupported shape policy: {shape_policy!r}.")
        self._backend: str = backend
        self._mode: CompilationMode = mode
        self._graph_break_policy: GraphBreakPolicy = graph_break_policy
        self._shape_policy: ShapePolicy = shape_policy

    def compile(self, module: nn.Module) -> nn.Module:
        if not isinstance(module, nn.Module):
            raise TypeError(
                f"TorchCompiler requires torch.nn.Module, got {type(module).__name__}."
            )
        fullgraph: bool = self._graph_break_policy == "forbid"
        dynamic: bool | None = self._dynamic_shapes()
        try:
            compiled_module: nn.Module = cast(
                nn.Module,
                torch.compile(
                    module,
                    fullgraph=fullgraph,
                    dynamic=dynamic,
                    backend=self._backend,
                    mode=self._mode
                )
            )
        except BaseException as exception:
            raise RuntimeError(
                f"torch.compile failed for {type(module).__name__} with "
                f"backend={self._backend!r}, mode={self._mode!r}, "
                f"graph_break_policy={self._graph_break_policy!r}, "
                f"shape_policy={self._shape_policy!r}."
            ) from exception
        return compiled_module

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def mode(self) -> CompilationMode:
        return self._mode

    @property
    def graph_break_policy(self) -> GraphBreakPolicy:
        return self._graph_break_policy

    @property
    def shape_policy(self) -> ShapePolicy:
        return self._shape_policy

    def _dynamic_shapes(self) -> bool | None:
        match self._shape_policy:
            case "auto":
                return None
            case "dynamic":
                return True
            case "static":
                return False
