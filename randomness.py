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
    def __init__(self, accelerator_types: Iterable[str]) -> None:
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
        return self._seed

    def _serialize_tensor_state(
        self,
        state: torch.Tensor
    ) -> _SerializedGeneratorState:
        return tuple(map(int, state.detach().cpu().tolist()))

    def _deserialize_tensor_state(
        self,
        state: _SerializedGeneratorState
    ) -> torch.Tensor:
        return torch.tensor(state, dtype=torch.uint8)
