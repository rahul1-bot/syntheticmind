# This module:
# 1. Executes one complete prediction pass under torch.inference_mode: the
#    prediction lifecycle hooks, the batch-transfer hook chain, per-batch
#    predict-step execution, and optional collection of the step outputs
#
# Design decisions:
# - Prediction produces no metrics and touches no optimization state; the loop
#   therefore has no accumulator machinery and only clears the module's logging
#   buffer at the end as a defensive measure, since logging is blocked inside
#   prediction hooks by the metric-logging contract
# - The entire batch loop runs inside torch.inference_mode because prediction
#   must never build autograd graphs
# - return_predictions controls whether step outputs are retained. Disabling
#   retention keeps memory flat for prediction passes whose outputs are written
#   to disk by the module or a callback rather than returned
# - The batch-transfer hook chain applies the same datamodule-over-module
#   precedence rule as the other loops, decided per hook by comparing bound
#   implementations against the DataModule base attributes
#
# Author: Rahul Sawhney

import inspect
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.datamodule import DataModule
from syntheticmind.core.module import Module
from syntheticmind.loops.fetchers import _DataFetcher
from syntheticmind.loops.loop import Loop
from syntheticmind.state.trainer_state import TrainerState
from syntheticmind.strategies.strategy import Strategy
from syntheticmind.utilities.combined_loader import CombinedLoader
from syntheticmind.utilities.data_transfer import move_data_to_device
from syntheticmind.utilities.evaluation import evaluation_context
from syntheticmind.utilities.model_helpers import (
    ModuleMode,
    is_overridden,
    is_parameter_in_hook_signature
)
from syntheticmind.utilities.types import Batch, ModelOutput

if TYPE_CHECKING:
    from syntheticmind.core.trainer import Trainer

__all__: list[str] = ["PredictionLoop"]


class PredictionLoop(Loop):
    # Loop that executes one inference pass over the prediction dataloader each
    # time run() is invoked, returning the collected predict-step outputs in
    # batch order when retention is enabled.
    def __init__(
        self,
        return_predictions: bool = True,
        inference_mode: bool = True
    ) -> None:
        # Binds the optional batch ceiling and the output-retention policy.
        super().__init__()
        self.return_predictions: bool = return_predictions
        self.inference_mode: bool = inference_mode

    def run(self) -> list[ModelOutput] | list[list[ModelOutput]] | None:
        with evaluation_context(self.inference_mode):
            return self._run()

    def _run(self) -> list[ModelOutput] | list[list[ModelOutput]] | None:
        # Executes one full prediction pass. The sequence is: switch the module
        # to evaluation mode through its prediction hook, fire the stage-start
        # and epoch-start hook pairs, iterate batches under inference mode with
        # the batch-start hooks, the batch-transfer chain, the predict step,
        # and the batch-end hooks, then fire the epoch-end and stage-end hook
        # pairs and return the collected outputs.
        trainer: Trainer = self.trainer
        module: Module | None = trainer.module
        assert module is not None
        strategy: Strategy = trainer.strategy
        state: TrainerState = trainer.state
        callbacks: list[Callback] = trainer.callbacks
        combined_loader: CombinedLoader | None = trainer._predict_combined_loader
        assert combined_loader is not None
        device: torch.device = strategy.root_device
        model: nn.Module | None = strategy.model
        assert model is not None

        state.set_running_stage("predicting")
        module._running_stage = "predicting"
        num_batches: list[int | float] = trainer.num_predict_batches
        combined_loader.limits = num_batches
        total_batches: int | None = combined_loader._length()
        max_batches: int | None = total_batches
        trainer._raise_if_sigterm()
        if max_batches == 0:
            module._reset_logged_metrics()
            return None

        num_dataloaders: int = len(combined_loader.flattened)
        include_dataloader_idx: bool = num_dataloaders > 1
        self._verify_dataloader_idx_requirements(module, include_dataloader_idx)

        module_mode: ModuleMode = ModuleMode()
        module_mode.capture(module)
        trainer._call_module_hook(
            module,
            "on_predict_model_eval",
            module.on_predict_model_eval
        )

        trainer._call_callback_hooks(
            module,
            "on_predict_start",
            callback_group=callbacks
        )
        trainer._call_module_hook(
            module,
            "on_predict_start",
            module.on_predict_start
        )
        trainer._call_callback_hooks(
            module,
            "on_predict_epoch_start",
            callback_group=callbacks
        )
        trainer._call_module_hook(
            module,
            "on_predict_epoch_start",
            module.on_predict_epoch_start
        )

        predictions_by_dataloader: list[list[ModelOutput]] = [
            [] for _ in range(num_dataloaders)
        ]

        dataloader_iterator_start_time: float = time.perf_counter()
        with trainer.profiler.profile("setup_predict_dataloader"):
            data_fetcher: _DataFetcher[tuple[object, int, int]] = _DataFetcher(
                combined_loader,
                max_batches,
                sized=combined_loader._has_sized_iterables()
            )
            dataloader_iterator: Iterator[tuple[object, int, int]] = iter(data_fetcher)
        trainer._record_dataloader_iterator_creation_time(
            "predicting",
            time.perf_counter() - dataloader_iterator_start_time
        )
        while not data_fetcher.done:
            trainer._raise_if_sigterm(synchronize=False)
            dataloader_fetch_start_time: float = time.perf_counter()
            try:
                with trainer.profiler.profile(
                    f"[{type(self).__name__}].predict_next"
                ):
                    fetched_batch: object
                    fetched_batch, batch_idx, dataloader_idx = next(dataloader_iterator)
            except StopIteration:
                break
            trainer._record_dataloader_fetch_time(
                "predicting",
                time.perf_counter() - dataloader_fetch_start_time
            )
            batch: Batch = cast(Batch, fetched_batch)
            batch: Batch = trainer.precision_plugin.convert_input(batch)
            hook_dataloader_idx: int | None = (
                dataloader_idx if include_dataloader_idx else None
            )
            batch_hook_arguments: tuple[object, ...] = (
                (batch, batch_idx)
                if hook_dataloader_idx is None
                else (batch, batch_idx, hook_dataloader_idx)
            )
            step_arguments: tuple[object, ...] = batch_hook_arguments
            if not is_parameter_in_hook_signature(
                module.predict_step,
                "batch_idx",
                minimum_arguments=2
            ):
                step_arguments = (
                    (batch,)
                    if hook_dataloader_idx is None
                    else (batch, hook_dataloader_idx)
                )

            trainer._call_callback_hooks(
                module,
                "on_predict_batch_start",
                *batch_hook_arguments,
                callback_group=callbacks
            )
            trainer._call_module_hook(
                module,
                "on_predict_batch_start",
                cast(Callable[..., object], module.on_predict_batch_start),
                *batch_hook_arguments
            )

            dm: DataModule | None = trainer.datamodule
            if dm is not None and is_overridden("on_before_batch_transfer", dm, DataModule):
                batch: Batch = trainer._call_datamodule_hook(
                    dm,
                    "on_before_batch_transfer",
                    dm.on_before_batch_transfer,
                    batch,
                    dataloader_idx=dataloader_idx
                )
            else:
                batch = trainer._call_module_hook(
                    module,
                    "on_before_batch_transfer",
                    module.on_before_batch_transfer,
                    batch,
                    dataloader_idx=dataloader_idx
                )
            if dm is not None and is_overridden("transfer_batch_to_device", dm, DataModule):
                batch = trainer._call_datamodule_hook(
                    dm,
                    "transfer_batch_to_device",
                    dm.transfer_batch_to_device,
                    batch,
                    device,
                    dataloader_idx=dataloader_idx
                )
            else:
                batch = trainer._call_module_hook(
                    module,
                    "transfer_batch_to_device",
                    module.transfer_batch_to_device,
                    batch,
                    device,
                    dataloader_idx=dataloader_idx
                )
            if dm is not None and is_overridden("on_after_batch_transfer", dm, DataModule):
                batch = trainer._call_datamodule_hook(
                    dm,
                    "on_after_batch_transfer",
                    dm.on_after_batch_transfer,
                    batch,
                    dataloader_idx=dataloader_idx
                )
            else:
                batch = trainer._call_module_hook(
                    module,
                    "on_after_batch_transfer",
                    module.on_after_batch_transfer,
                    batch,
                    dataloader_idx=dataloader_idx
                )

            with trainer.precision_plugin.predict_step_context():
                with module._hook_context("predict_step"):
                    outputs: ModelOutput = trainer._call_strategy_hook(
                        "predict_step",
                        strategy.predict_step,
                        model,
                        *step_arguments
                    )
            if self.return_predictions:
                predictions_by_dataloader[dataloader_idx].append(
                    cast(
                        ModelOutput,
                        move_data_to_device(cast(Batch, outputs), torch.device("cpu"))
                    )
                )

            trainer._call_callback_hooks(
                module,
                "on_predict_batch_end",
                outputs,
                *batch_hook_arguments,
                callback_group=callbacks
            )
            trainer._call_module_hook(
                module,
                "on_predict_batch_end",
                cast(Callable[..., object], module.on_predict_batch_end),
                outputs,
                *batch_hook_arguments
            )

        trainer._call_callback_hooks(
            module,
            "on_predict_epoch_end",
            callback_group=callbacks
        )
        trainer._call_module_hook(
            module,
            "on_predict_epoch_end",
            module.on_predict_epoch_end
        )
        trainer._call_callback_hooks(
            module,
            "on_predict_end",
            callback_group=callbacks
        )
        trainer._call_module_hook(
            module,
            "on_predict_end",
            module.on_predict_end
        )
        module_mode.restore(module)

        module._reset_logged_metrics()
        trainer._raise_if_sigterm()
        if not self.return_predictions:
            return None
        if num_dataloaders == 1:
            return predictions_by_dataloader[0]
        return predictions_by_dataloader

    @staticmethod
    def _verify_dataloader_idx_requirements(
        module: Module,
        include_dataloader_idx: bool
    ) -> None:
        for hook_name in (
            "predict_step",
            "on_predict_batch_start",
            "on_predict_batch_end"
        ):
            hook: object = getattr(module, hook_name)
            if not callable(hook):
                continue
            parameters: dict[str, inspect.Parameter] = dict(
                inspect.signature(hook).parameters
            )
            has_varargs: bool = any(
                parameter.kind is inspect.Parameter.VAR_POSITIONAL
                for parameter in parameters.values()
            )
            dataloader_parameter: inspect.Parameter | None = parameters.get(
                "dataloader_idx"
            )
            if (
                include_dataloader_idx
                and dataloader_parameter is None
                and not has_varargs
            ):
                raise RuntimeError(
                    f"Multiple predict dataloaders require `dataloader_idx` in "
                    f"{type(module).__qualname__}.{hook_name}()."
                )
            if (
                not include_dataloader_idx
                and dataloader_parameter is not None
                and dataloader_parameter.default is inspect.Parameter.empty
            ):
                raise RuntimeError(
                    f"A single predict dataloader requires "
                    f"{type(module).__qualname__}.{hook_name}() to remove `dataloader_idx` "
                    "or give it the default value 0."
                )
