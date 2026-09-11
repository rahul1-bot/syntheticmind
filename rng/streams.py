# This module:
# 1. Owns process-wide seeding for Python, NumPy, torch CPU, CUDA, and MPS random
#    streams selected by the active accelerator topology
# 2. Defines the immutable random-stream snapshot used by checkpoints,
#    multiprocessing transfer, exact partial-epoch continuation, and reversible
#    model inspection
#
# Design decisions:
# - Only active accelerator families are captured, preventing a CPU run from
#   initializing CUDA or MPS merely to inspect random state and preserving fork
#   safety on process-sensitive backends
# - Tensor generator states are serialized as integer tuples so the strict
#   Pydantic snapshot crosses checkpoint and multiprocessing boundaries without
#   unsafe arbitrary tensor reconstruction
# - One owner seeds, captures, and restores every stream, preventing checkpoint,
#   launcher, and callback code from implementing subtly different trajectories
# - PYTHONHASHSEED is written for subsequently launched interpreters; Python's
#   current-process hash seed is fixed before this module can execute
# - Deterministic algorithm selection is excluded and belongs to Trainer's
#   deterministic execution policy rather than to stochastic stream state
#
# Author: Rahul Sawhney

import os
import random
from collections.abc import Iterable
from typing import ClassVar, cast

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

__all__: list[str] = []

type _PythonRandomState = tuple[int, tuple[int, ...], float | None]
type _NumPyRandomState = tuple[str, np.ndarray, int, int, float]
type _SerializedGeneratorState = tuple[int, ...]


class _RandomStateSnapshot(BaseModel):
    # Immutable transport schema for every supported process-wide random stream.
    # The Python and NumPy tuple layouts are expanded into named fields so schema
    # validation can reject malformed state before any global generator mutates.
    python_version: int
    python_state: tuple[int, ...]
    python_gauss: float | None
    numpy_bit_generator: str
    numpy_state: tuple[int, ...]
    numpy_position: int
    numpy_has_gauss: int
    numpy_cached_gaussian: float
    torch_cpu_state: _SerializedGeneratorState
    torch_cuda_states: tuple[_SerializedGeneratorState, ...]
    torch_mps_state: _SerializedGeneratorState | None
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        validate_default=True
    )


class _Randomness:
    # Stateful owner for the random streams relevant to one Trainer topology.
    # Trainer constructs it from selected device types, checkpointing captures
    # and restores it, launchers transfer it, and model inspection borrows it
    # transactionally so observation cannot perturb the training trajectory.
    def __init__(self, accelerator_types: Iterable[str]) -> None:
        # Binds the active accelerator families without initializing their
        # generators. Unsupported names fail before any global random state is
        # read or mutated.
        #
        # Args:
        #     accelerator_types: Device-family names participating in the run;
        #         accepted values are ``"cpu"``, ``"cuda"``, and ``"mps"``.
        #
        # Raises:
        #     ValueError: If any requested accelerator family is unsupported.
        resolved_accelerator_types: frozenset[str] = frozenset(accelerator_types)
        unsupported_types: frozenset[str] = resolved_accelerator_types.difference(
            {"cpu", "cuda", "mps"}
        )
        if unsupported_types:
            raise ValueError(
                f"Randomness received unsupported accelerator types "
                f"{sorted(unsupported_types)}."
            )
        self._accelerator_types: frozenset[str] = resolved_accelerator_types
        self._seed: int | None = None

    def seed(self, seed: int) -> None:
        # Applies one integer seed to every active stream and records it for
        # inspection. NumPy receives the seed modulo its unsigned 32-bit range,
        # while torch and Python retain their wider supported ranges.
        #
        # Raises:
        #     TypeError: If seed is not an integer or is a bool.
        #     ValueError: If seed lies outside the range accepted by torch.
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"Random seed must be an integer, got {seed!r}.")
        if not -(2 ** 63) <= seed <= (2 ** 64) - 1:
            raise ValueError(
                "Random seed must be between -(2**63) and 2**64 - 1, got "
                f"{seed}."
            )
        self._seed: int | None = seed
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if "cuda" in self._accelerator_types:
            torch.cuda.manual_seed_all(seed)
        if "mps" in self._accelerator_types:
            torch.mps.manual_seed(seed)
        os.environ.update({"PYTHONHASHSEED": str(seed)})

    def capture(self) -> _RandomStateSnapshot:
        # Captures a mutation-free snapshot of the current stochastic trajectory.
        # CUDA and MPS are consulted only when their family was bound at
        # construction, keeping inactive backends uninitialized.
        python_random_state: _PythonRandomState = cast(
            _PythonRandomState,
            random.getstate()
        )
        numpy_random_state: _NumPyRandomState = cast(
            _NumPyRandomState,
            np.random.get_state()
        )
        torch_cpu_state: _SerializedGeneratorState = self._serialize_tensor_state(
            torch.random.get_rng_state()
        )
        torch_cuda_states: tuple[_SerializedGeneratorState, ...] = tuple(
            self._serialize_tensor_state(state)
            for state in torch.cuda.get_rng_state_all()
        ) if "cuda" in self._accelerator_types else ()
        torch_mps_state: _SerializedGeneratorState | None = (
            self._serialize_tensor_state(torch.mps.get_rng_state())
            if "mps" in self._accelerator_types
            else None
        )
        return _RandomStateSnapshot(
            python_version=python_random_state[0],
            python_state=python_random_state[1],
            python_gauss=python_random_state[2],
            numpy_bit_generator=numpy_random_state[0],
            numpy_state=tuple(map(int, numpy_random_state[1])),
            numpy_position=numpy_random_state[2],
            numpy_has_gauss=numpy_random_state[3],
            numpy_cached_gaussian=numpy_random_state[4],
            torch_cpu_state=torch_cpu_state,
            torch_cuda_states=torch_cuda_states,
            torch_mps_state=torch_mps_state
        )

    def restore(self, snapshot: _RandomStateSnapshot) -> None:
        # Restores every active stream from one validated snapshot. Inactive
        # accelerator entries remain untouched even when the snapshot contains
        # state captured by a different active topology.
        #
        # Raises:
        #     TypeError: If snapshot is not the strict random-state model.
        if not isinstance(snapshot, _RandomStateSnapshot):
            raise TypeError("Randomness.restore requires a RandomStateSnapshot.")
        random.setstate(
            (
                snapshot.python_version,
                snapshot.python_state,
                snapshot.python_gauss
            )
        )
        np.random.set_state(
            (
                snapshot.numpy_bit_generator,
                np.asarray(snapshot.numpy_state, dtype=np.uint32),
                snapshot.numpy_position,
                snapshot.numpy_has_gauss,
                snapshot.numpy_cached_gaussian
            )
        )
        torch.random.set_rng_state(
            self._deserialize_tensor_state(snapshot.torch_cpu_state)
        )
        if "cuda" in self._accelerator_types:
            cuda_states: list[torch.Tensor] = [
                self._deserialize_tensor_state(state)
                for state in snapshot.torch_cuda_states
            ]
            torch.cuda.set_rng_state_all(cuda_states)
        if "mps" in self._accelerator_types and snapshot.torch_mps_state is not None:
            torch.mps.set_rng_state(
                self._deserialize_tensor_state(snapshot.torch_mps_state)
            )

    @property
    def seed_value(self) -> int | None:
        # Most recent seed explicitly applied through this owner, or None when
        # the inherited process streams have not been reseeded.
        return self._seed

    def _serialize_tensor_state(
        self,
        state: torch.Tensor
    ) -> _SerializedGeneratorState:
        # Converts a device generator byte tensor into a portable integer tuple.
        return tuple(map(int, state.detach().cpu().tolist()))

    def _deserialize_tensor_state(
        self,
        state: _SerializedGeneratorState
    ) -> torch.Tensor:
        # Reconstructs the uint8 tensor format expected by torch generator APIs.
        return torch.tensor(state, dtype=torch.uint8)
