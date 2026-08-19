# This module:
# 1. Assembles the complete checkpoint payload: model weights, per-optimizer and
#    per-scheduler state, random-number streams, callback state keyed by state_key,
#    optional datamodule state, mid-epoch resume markers, and environment metadata
# 2. Restores that payload back onto live objects while tolerating legacy
#    single-optimizer and single-scheduler layouts
# 3. Collects and restores the Python, NumPy, torch, and CUDA random streams
#
# Design decisions:
# - The payload writes both the list form (optimizer_state_dicts,
#   scheduler_state_dicts) and the first entry under the legacy singular keys, so
#   older readers and single-optimizer tooling keep working against new files
# - restore() pairs optimizers and schedulers with saved states through
#   zip(strict=False) because a resumed configuration may carry a different count
#   than the checkpoint; unmatched objects simply start fresh
# - A checkpoint without optimizer state is tolerated with a warning rather than
#   an error, because exception-time checkpoints can capture a partially
#   constructed run whose weights are still worth restoring
# - completed_batches and epoch_rng_state enter the payload only when mid-epoch
#   resume is actually in play, keeping epoch-boundary checkpoints minimal
# - torch and CUDA RNG tensors are moved to CPU before restoration because the
#   set_rng_state APIs require CPU byte tensors regardless of the producing device
# - Callback state is keyed by each callback's state_key so two instances of the
#   same class with different configurations never overwrite each other
#
# Author: Rahul Sawhney

import random
import warnings
from typing import cast

import numpy as np
import torch

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.datamodule import DataModule
from syntheticmind.core.hyperparameters import HyperparametersMixin
from syntheticmind.core.optimizer import _SchedulerInstance
from syntheticmind.precision.precision import Precision
from syntheticmind.utilities.types import (
    CheckpointDict,
    CheckpointValue,
    HyperparameterDict,
    StateDict
)

__all__: list[str] = ["CheckpointState"]


class CheckpointState:
    # Stateless assembler and restorer for checkpoint payloads, exposed as
    # classmethods because the payload schema, not instance state, is the thing
    # being managed. build() and restore() are exact inverses over that schema.

    @classmethod
    def build(
        cls,
        model: torch.nn.Module,
        optimizers: list[torch.optim.Optimizer],
        scheduler: _SchedulerInstance | None,
        schedulers: list[_SchedulerInstance] | None,
        epoch: int,
        global_step: int,
        callbacks: list[Callback],
        datamodule: DataModule | None = None,
        completed_batches: int = 0,
        total_batch_idx: int = -1,
        epoch_rng_state: CheckpointDict | None = None,
        combined_loader_states: list[dict[str, object]] | None = None,
        strategy_name: str = "SingleDeviceStrategy",
        world_size: int = 1,
        precision_plugin: Precision | None = None,
        weights_only: bool = False
    ) -> CheckpointDict:
        # Assembles the payload from live objects. Progress counters and model
        # weights always travel; optimizer state travels in list form plus the
        # legacy first-entry key; RNG streams are captured at call time so a
        # restored run continues the same random sequences.
        model_state_dict: CheckpointValue = cast(
            CheckpointValue,
            model.state_dict()
        )
        checkpoint: CheckpointDict = {
            "epoch": epoch,
            "global_step": global_step,
            "state_dict": model_state_dict,
            "model_state_dict": model_state_dict,
        }
        if isinstance(model, HyperparametersMixin):
            cls._store_hyperparameters(checkpoint, model)

        if not weights_only:
            checkpoint["rng_state"] = cls._collect_rng_states()
            optimizer_states: CheckpointValue = cast(
                CheckpointValue,
                [optimizer.state_dict() for optimizer in optimizers]
            )
            checkpoint["optimizer_states"] = optimizer_states
            if optimizers:
                checkpoint["optimizer_state_dict"] = cast(
                    CheckpointValue,
                    optimizers[0].state_dict()
                )
                checkpoint["optimizer_states"] = optimizer_states
                checkpoint["optimizer_state_dicts"] = optimizer_states

        if completed_batches > 0:
            checkpoint["completed_batches"] = completed_batches

        if total_batch_idx >= 0:
            checkpoint["total_batch_idx"] = total_batch_idx

        if epoch_rng_state is not None:
            checkpoint["epoch_rng_state"] = epoch_rng_state

        if combined_loader_states:
            checkpoint["loops"] = cast(
                CheckpointValue,
                {
                    "fit_loop": {
                        "state_dict": {
                            "combined_loader": combined_loader_states
                        }
                    }
                }
            )

        if not weights_only:
            scheduler_list: list[_SchedulerInstance] = []
            if schedulers is not None:
                scheduler_list = schedulers
            elif scheduler is not None:
                scheduler_list = [scheduler]
            scheduler_states: CheckpointValue = cast(
                CheckpointValue,
                [
                    current_scheduler.state_dict()
                    for current_scheduler in scheduler_list
                ]
            )
            checkpoint["lr_schedulers"] = scheduler_states
            if scheduler_list:
                checkpoint["scheduler_state_dicts"] = scheduler_states
                checkpoint["scheduler_state_dict"] = cast(
                    CheckpointValue,
                    scheduler_list[0].state_dict()
                )

            callback_states: dict[str, StateDict] = {}
            for callback in callbacks:
                state: StateDict = callback.state_dict()
                if state:
                    callback_states[callback.state_key] = state
            checkpoint["callback_states"] = cast(
                CheckpointValue,
                callback_states
            )
            checkpoint["callbacks"] = cast(
                CheckpointValue,
                callback_states
            )

        if datamodule is not None:
            cls._store_hyperparameters(checkpoint, datamodule)
            datamodule_state: StateDict = datamodule.state_dict()
            if datamodule_state:
                checkpoint["datamodule_state_dict"] = cast(
                    CheckpointValue,
                    datamodule_state
                )
                checkpoint[type(datamodule).__qualname__] = cast(
                    CheckpointValue,
                    datamodule_state
                )

        if precision_plugin is not None and not weights_only:
            precision_state: StateDict = precision_plugin.state_dict()
            if precision_state:
                checkpoint[type(precision_plugin).__qualname__] = cast(
                    CheckpointValue,
                    precision_state
                )

        checkpoint["metadata"] = {
            "torch_version": str(torch.__version__),
            "strategy": strategy_name,
            "world_size": world_size
        }

        return checkpoint

    @classmethod
    def _store_hyperparameters(
        cls,
        checkpoint: CheckpointDict,
        source: HyperparametersMixin
    ) -> None:
        source._clean_hparams()
        if not source.hparams:
            return
        checkpoint[source.CHECKPOINT_HYPER_PARAMS_NAME] = source._hparams_name
        if cls._is_omegaconf_container(source.hparams):
            checkpoint[source.CHECKPOINT_HYPER_PARAMS_KEY] = cast(
                CheckpointValue,
                source.hparams
            )
            checkpoint[source.CHECKPOINT_HYPER_PARAMS_TYPE] = cast(
                CheckpointValue,
                type(source.hparams)
            )
        else:
            stored_hyperparameters: HyperparameterDict = {
                key: value for key, value in source.hparams.items()
            }
            checkpoint[source.CHECKPOINT_HYPER_PARAMS_KEY] = cast(
                CheckpointValue,
                stored_hyperparameters
            )

    @classmethod
    def _is_omegaconf_container(cls, value: object) -> bool:
        try:
            from omegaconf import Container
        except ImportError:
            return False
        return isinstance(value, Container)

    @classmethod
    def restore(
        cls,
        checkpoint: CheckpointDict,
        model: torch.nn.Module,
        optimizers: list[torch.optim.Optimizer],
        scheduler: _SchedulerInstance | None,
        schedulers: list[_SchedulerInstance] | None,
        callbacks: list[Callback],
        datamodule: DataModule | None = None,
        precision_plugin: Precision | None = None,
        restore_datamodule_state: bool = True,
        strict: bool = True
    ) -> tuple[int, int, int, int, CheckpointDict | None]:
        # Rehydrates live objects from the payload and returns the resume
        # coordinates (epoch, global step, completed batches, epoch RNG state).
        # The list-form optimizer and scheduler keys are preferred; the legacy
        # singular keys are the fallback for older checkpoints.
        model_state_dict_raw: CheckpointValue | None = checkpoint.get(
            "state_dict",
            checkpoint.get("model_state_dict")
        )
        if not isinstance(model_state_dict_raw, dict):
            raise KeyError("Checkpoint is missing a model `state_dict`.")
        incompatible_keys: object = model.load_state_dict(
            model_state_dict_raw,
            strict=strict
        )  # type: ignore[arg-type]
        if not strict:
            missing_keys: object = getattr(incompatible_keys, "missing_keys", [])
            unexpected_keys: object = getattr(
                incompatible_keys,
                "unexpected_keys",
                []
            )
            if missing_keys:
                warnings.warn(
                    "Found keys that are in the model state dict but not in the "
                    f"checkpoint: {missing_keys}",
                    UserWarning,
                    stacklevel=3
                )
            if unexpected_keys:
                warnings.warn(
                    "Found keys that are not in the model state dict but in the "
                    f"checkpoint: {unexpected_keys}",
                    UserWarning,
                    stacklevel=3
                )
        optimizer_state_dicts_raw: CheckpointValue | None = checkpoint.get(
            "optimizer_states",
            checkpoint.get("optimizer_state_dicts")
        )
        if isinstance(optimizer_state_dicts_raw, list):
            for optimizer, optimizer_state_dict in zip(optimizers, optimizer_state_dicts_raw, strict=False):
                optimizer.load_state_dict(optimizer_state_dict)  # type: ignore[arg-type]
        elif "optimizer_state_dict" in checkpoint and optimizers:
            optimizers[0].load_state_dict(checkpoint["optimizer_state_dict"])  # type: ignore[arg-type]
        elif optimizers:
            from loguru import logger as restore_log
            restore_log.warning(
                "Checkpoint missing optimizer_state_dict (partial/exception checkpoint); "
                "optimizer will start fresh."
            )

        scheduler_state_dicts_raw: CheckpointValue | None = checkpoint.get(
            "lr_schedulers",
            checkpoint.get("scheduler_state_dicts")
        )
        scheduler_list: list[_SchedulerInstance] = []
        if schedulers is not None:
            scheduler_list: list[_SchedulerInstance] = schedulers
        elif scheduler is not None:
            scheduler_list: list[_SchedulerInstance] = [scheduler]
        if isinstance(scheduler_state_dicts_raw, list):
            for current_scheduler, scheduler_state_dict in zip(
                scheduler_list,
                scheduler_state_dicts_raw,
                strict=False
            ):
                current_scheduler.load_state_dict(scheduler_state_dict)  # type: ignore[arg-type]
        elif scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])  # type: ignore[arg-type]

        rng_state: CheckpointValue | None = checkpoint.get("rng_state")
        if isinstance(rng_state, dict):
            cls._restore_rng_states(rng_state)

        callback_states_raw: CheckpointValue | None = checkpoint.get(
            "callbacks",
            checkpoint.get("callback_states")
        )
        callback_states: dict[str, StateDict] = (
            cast(dict[str, StateDict], callback_states_raw)
            if isinstance(callback_states_raw, dict)
            else {}
        )
        for callback in callbacks:
            key: str = callback.state_key
            if key in callback_states:
                callback.load_state_dict(callback_states[key])

        if datamodule is not None and restore_datamodule_state:
            cls.load_datamodule_state(checkpoint, datamodule)

        if precision_plugin is not None:
            precision_state_raw: CheckpointValue | None = checkpoint.get(
                type(precision_plugin).__qualname__
            )
            if isinstance(precision_state_raw, dict):
                precision_plugin.load_state_dict(cast(StateDict, precision_state_raw))

        epoch_value: CheckpointValue | None = checkpoint.get("epoch", 0)
        global_step_value: CheckpointValue | None = checkpoint.get("global_step", 0)
        completed_batches_value: CheckpointValue | None = checkpoint.get("completed_batches", 0)
        total_batch_idx_value: CheckpointValue | None = checkpoint.get("total_batch_idx", -1)
        epoch: int = cls._integer_value(epoch_value, 0)
        global_step: int = cls._integer_value(global_step_value, 0)
        completed_batches: int = cls._integer_value(completed_batches_value, 0)
        total_batch_idx: int = cls._integer_value(total_batch_idx_value, -1)
        epoch_rng_state_raw: CheckpointValue | None = checkpoint.get("epoch_rng_state")
        epoch_rng_state: CheckpointDict | None = (
            cast(CheckpointDict, epoch_rng_state_raw)
            if isinstance(epoch_rng_state_raw, dict)
            else None
        )
        return epoch, global_step, completed_batches, total_batch_idx, epoch_rng_state

    @staticmethod
    def combined_loader_states(
        checkpoint: CheckpointDict
    ) -> list[dict[str, object]]:
        loops_value: CheckpointValue | None = checkpoint.get("loops")
        if loops_value is None:
            return []
        if not isinstance(loops_value, dict):
            raise TypeError("Checkpoint `loops` state must be a mapping.")
        fit_loop_value: CheckpointValue | None = loops_value.get("fit_loop")
        if fit_loop_value is None:
            return []
        if not isinstance(fit_loop_value, dict):
            raise TypeError("Checkpoint `loops.fit_loop` state must be a mapping.")
        state_dict_value: CheckpointValue | None = fit_loop_value.get("state_dict")
        if state_dict_value is None:
            return []
        if not isinstance(state_dict_value, dict):
            raise TypeError("Checkpoint fit-loop `state_dict` must be a mapping.")
        combined_loader_value: CheckpointValue | None = state_dict_value.get(
            "combined_loader"
        )
        if combined_loader_value is None:
            return []
        if not isinstance(combined_loader_value, list):
            raise TypeError("Checkpoint combined-loader state must be a list.")
        states: list[dict[str, object]] = []
        state: CheckpointValue
        for state in combined_loader_value:
            if not isinstance(state, dict):
                raise TypeError(
                    "Each checkpointed combined-loader state must be a mapping."
                )
            states.append(cast(dict[str, object], state))
        return states

    @staticmethod
    def load_datamodule_state(
        checkpoint: CheckpointDict,
        datamodule: DataModule
    ) -> None:
        datamodule_state_raw: CheckpointValue | None = checkpoint.get(
            type(datamodule).__qualname__,
            checkpoint.get("datamodule_state_dict")
        )
        if isinstance(datamodule_state_raw, dict):
            datamodule.load_state_dict(cast(StateDict, datamodule_state_raw))

    @staticmethod
    def _integer_value(value: CheckpointValue | None, default: int) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float | str):
            try:
                return int(value)
            except (OverflowError, ValueError):
                return default
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return int(value.item())
        return default

    @classmethod
    def _collect_rng_states(cls) -> CheckpointDict:
        # Captures every random stream the harness seeds: Python, NumPy, torch
        # CPU, and one state per visible CUDA device when CUDA is present.
        numpy_rng_state: tuple[str, np.ndarray, int, int, float] = cast(
            tuple[str, np.ndarray, int, int, float],
            np.random.get_state()
        )
        states: CheckpointDict = {
            "python": random.getstate(),
            "numpy": cast(
                CheckpointValue,
                {
                    "bit_generator": numpy_rng_state[0],
                    "state": torch.from_numpy(numpy_rng_state[1].copy()),
                    "position": numpy_rng_state[2],
                    "has_gauss": numpy_rng_state[3],
                    "cached_gaussian": numpy_rng_state[4]
                }
            ),
            "torch": torch.random.get_rng_state()
        }
        if torch.cuda.is_available():
            states["cuda"] = cast(
                CheckpointValue,
                torch.cuda.get_rng_state_all()
            )
        return states

    @classmethod
    def _restore_rng_states(cls, rng_state: CheckpointDict) -> None:
        # Restores each stream that is present, skipping absent keys so payloads
        # from CPU-only producers restore cleanly on CUDA hosts and vice versa.
        # Tensor states are moved to CPU first because the torch restore APIs
        # accept only CPU byte tensors.
        if "python" in rng_state:
            random.setstate(rng_state["python"])  # type: ignore[arg-type]
        if "numpy" in rng_state:
            numpy_rng_state: CheckpointValue = rng_state["numpy"]
            if isinstance(numpy_rng_state, dict):
                bit_generator: CheckpointValue | None = numpy_rng_state.get(
                    "bit_generator"
                )
                state_tensor: CheckpointValue | None = numpy_rng_state.get("state")
                position: CheckpointValue | None = numpy_rng_state.get("position")
                has_gauss: CheckpointValue | None = numpy_rng_state.get("has_gauss")
                cached_gaussian: CheckpointValue | None = numpy_rng_state.get(
                    "cached_gaussian"
                )
                if (
                    not isinstance(bit_generator, str)
                    or not isinstance(state_tensor, torch.Tensor)
                    or type(position) is not int
                    or type(has_gauss) is not int
                    or not isinstance(cached_gaussian, int | float)
                ):
                    raise TypeError("Checkpoint NumPy RNG state is malformed.")
                np.random.set_state(
                    (
                        bit_generator,
                        state_tensor.detach().cpu().numpy().astype(
                            np.uint32,
                            copy=True
                        ),
                        position,
                        has_gauss,
                        float(cached_gaussian)
                    )
                )
            else:
                np.random.set_state(numpy_rng_state)  # type: ignore[arg-type]
        if "torch" in rng_state:
            torch_rng: CheckpointValue = rng_state["torch"]
            if isinstance(torch_rng, torch.Tensor):
                torch_rng: CheckpointValue = torch_rng.cpu()
            torch.random.set_rng_state(torch_rng)  # type: ignore[arg-type]
        if "cuda" in rng_state and torch.cuda.is_available():
            cuda_states: CheckpointValue = rng_state["cuda"]
            if isinstance(cuda_states, list):
                cuda_states: CheckpointValue = [s.cpu() if isinstance(s, torch.Tensor) else s for s in cuda_states]
            torch.cuda.set_rng_state_all(cuda_states)  # type: ignore[arg-type]
