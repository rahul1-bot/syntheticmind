import inspect
from collections.abc import Callable, Iterator
from typing import cast, override

import syntheticmind
import torch
from torch import nn

from syntheticmind.callbacks.callback import Callback
from syntheticmind.core.module import Module
from syntheticmind.loops.fetchers import _DataFetcher
from syntheticmind.loops.loop import Loop
from syntheticmind.data.combined_loader import CombinedLoader
from syntheticmind.primitives.types import Batch, ModelOutput, RunningStage, StepArgument

__all__: list[str] = ["PredictionLoop"]

type _PredictionSetup = tuple[CombinedLoader, int | None, int, bool]
type _PredictionHookArguments = tuple[StepArgument, ...]
type _PredictionHook = Callable[..., ModelOutput | None]


class PredictionLoop(Loop):
    @override
    def __init__(
        self,
        return_predictions: bool = True,
        inference_mode: bool = True
    ) -> None:
        super().__init__()
        self.return_predictions: bool = return_predictions
        self.inference_mode: bool = inference_mode

    def run(self) -> list[ModelOutput] | list[list[ModelOutput]] | None:
        with self._evaluation_context(self.inference_mode):
            return self._run()

    def _run(self) -> list[ModelOutput] | list[list[ModelOutput]] | None:
        trainer: syntheticmind.Trainer = self.trainer
        module: Module | None = trainer.module
        assert module is not None
        callbacks: list[Callback] = trainer.callbacks
        prediction_setup: _PredictionSetup = self._prepare_prediction(
            trainer,
            module
        )
        combined_loader: CombinedLoader = prediction_setup[0]
        max_batches: int | None = prediction_setup[1]
        num_dataloaders: int = prediction_setup[2]
        include_dataloader_idx: bool = prediction_setup[3]
        if max_batches == 0:
            module._reset_logged_metrics()
            return None
        self._verify_dataloader_idx_requirements(module, include_dataloader_idx)

        training_modes: dict[str, bool] = module._capture_training_modes()
        self._run_stage_start(trainer, module, callbacks)

        predictions_by_dataloader: list[list[ModelOutput]] = [
            [] for _ in range(num_dataloaders)
        ]
        self._run_batches(
            trainer,
            module,
            callbacks,
            combined_loader,
            max_batches,
            include_dataloader_idx,
            predictions_by_dataloader
        )

        self._run_stage_end(trainer, module, callbacks, training_modes)

        module._reset_logged_metrics()
        trainer._raise_if_sigterm()
        if not self.return_predictions:
            return None
        if num_dataloaders == 1:
            return predictions_by_dataloader[0]
        return predictions_by_dataloader

    def _prepare_prediction(
        self,
        trainer: syntheticmind.Trainer,
        module: Module
    ) -> _PredictionSetup:
        combined_loader: CombinedLoader | None = (
            trainer._data_connector.predict_combined_loader
        )
        assert combined_loader is not None
        trainer.state.set_running_stage("predicting")
        module._running_stage: RunningStage | None = "predicting"
        num_batches: list[int | float] = trainer.num_predict_batches
        combined_loader.limits: list[int | float | None] | None = num_batches
        max_batches: int | None = combined_loader._length()
        trainer._raise_if_sigterm()
        num_dataloaders: int = len(combined_loader.flattened)
        include_dataloader_idx: bool = num_dataloaders > 1
        return combined_loader, max_batches, num_dataloaders, include_dataloader_idx

    def _run_stage_start(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        callbacks: list[Callback]
    ) -> None:
        trainer._hook_dispatcher.call_module(
            module,
            "on_predict_model_eval",
            module.on_predict_model_eval
        )
        trainer._hook_dispatcher.call_callbacks(
            module,
            "on_predict_start",
            callback_group=callbacks
        )
        trainer._hook_dispatcher.call_module(
            module,
            "on_predict_start",
            module.on_predict_start
        )
        trainer._hook_dispatcher.call_callbacks(
            module,
            "on_predict_epoch_start",
            callback_group=callbacks
        )
        trainer._hook_dispatcher.call_module(
            module,
            "on_predict_epoch_start",
            module.on_predict_epoch_start
        )

    def _run_stage_end(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        callbacks: list[Callback],
        training_modes: dict[str, bool]
    ) -> None:
        trainer._hook_dispatcher.call_callbacks(
            module,
            "on_predict_epoch_end",
            callback_group=callbacks
        )
        trainer._hook_dispatcher.call_module(
            module,
            "on_predict_epoch_end",
            module.on_predict_epoch_end
        )
        trainer._hook_dispatcher.call_callbacks(
            module,
            "on_predict_end",
            callback_group=callbacks
        )
        trainer._hook_dispatcher.call_module(
            module,
            "on_predict_end",
            module.on_predict_end
        )
        module._restore_training_modes(training_modes)

    def _run_batches(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        callbacks: list[Callback],
        combined_loader: CombinedLoader,
        max_batches: int | None,
        include_dataloader_idx: bool,
        predictions_by_dataloader: list[list[ModelOutput]]
    ) -> None:
        with trainer.profiler.profile("setup_predict_dataloader"):
            data_fetcher: _DataFetcher[tuple[Batch, int, int]] = _DataFetcher(
                combined_loader,
                max_batches,
                sized=combined_loader._has_sized_iterables()
            )
            dataloader_iterator: Iterator[tuple[Batch, int, int]] = iter(
                data_fetcher
            )
        while not data_fetcher.done:
            trainer._raise_if_sigterm(synchronize=False)
            try:
                with trainer.profiler.profile(
                    f"[{type(self).__name__}].predict_next"
                ):
                    fetched: tuple[Batch, int, int] = next(dataloader_iterator)
            except StopIteration:
                break
            self._advance_batch(
                trainer,
                module,
                callbacks,
                fetched,
                include_dataloader_idx,
                predictions_by_dataloader
            )

    def _advance_batch(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        callbacks: list[Callback],
        fetched: tuple[Batch, int, int],
        include_dataloader_idx: bool,
        predictions_by_dataloader: list[list[ModelOutput]]
    ) -> None:
        batch: Batch = trainer.precision_plugin.convert_input(fetched[0])
        batch_idx: int = fetched[1]
        dataloader_idx: int = fetched[2]
        hook_dataloader_idx: int | None = (
            dataloader_idx if include_dataloader_idx else None
        )
        batch: Batch = trainer._transfer_batch(
            module,
            batch,
            trainer.strategy.root_device,
            dataloader_idx
        )
        hook_arguments: _PredictionHookArguments = self._hook_arguments(
            batch,
            batch_idx,
            hook_dataloader_idx
        )
        step_arguments: _PredictionHookArguments = self._step_arguments(
            module,
            batch,
            batch_idx,
            hook_dataloader_idx
        )
        trainer._hook_dispatcher.call_callbacks(
            module,
            "on_predict_batch_start",
            *hook_arguments,
            callback_group=callbacks
        )
        trainer._hook_dispatcher.call_module(
            module,
            "on_predict_batch_start",
            module.on_predict_batch_start,
            *hook_arguments
        )
        outputs: ModelOutput = self._run_prediction_step(
            trainer,
            module,
            step_arguments
        )
        if self.return_predictions:
            predictions_by_dataloader[dataloader_idx].append(
                cast(
                    ModelOutput,
                    module._move_data_to_device(
                        cast(Batch, outputs),
                        torch.device("cpu")
                    )
                )
            )
        trainer._hook_dispatcher.call_callbacks(
            module,
            "on_predict_batch_end",
            outputs,
            *hook_arguments,
            callback_group=callbacks
        )
        trainer._hook_dispatcher.call_module(
            module,
            "on_predict_batch_end",
            module.on_predict_batch_end,
            outputs,
            *hook_arguments
        )

    def _hook_arguments(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int | None
    ) -> _PredictionHookArguments:
        if dataloader_idx is None:
            return batch, batch_idx
        return batch, batch_idx, dataloader_idx

    def _step_arguments(
        self,
        module: Module,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int | None
    ) -> _PredictionHookArguments:
        if module._hook_accepts_parameter(
            module.predict_step,
            "batch_idx",
            minimum_arguments=2
        ):
            return self._hook_arguments(batch, batch_idx, dataloader_idx)
        if dataloader_idx is None:
            return tuple([batch])
        return batch, dataloader_idx

    def _run_prediction_step(
        self,
        trainer: syntheticmind.Trainer,
        module: Module,
        step_arguments: _PredictionHookArguments
    ) -> ModelOutput:
        model: nn.Module | None = trainer.strategy.model
        assert model is not None
        with trainer.precision_plugin.predict_step_context():
            with module._hook_context("predict_step"):
                return trainer._hook_dispatcher.call_strategy(
                    "predict_step",
                    trainer.strategy.predict_step,
                    model,
                    *step_arguments
                )

    def _verify_dataloader_idx_requirements(
        self,
        module: Module,
        include_dataloader_idx: bool
    ) -> None:
        for hook_name in (
            "predict_step",
            "on_predict_batch_start",
            "on_predict_batch_end"
        ):
            hook: _PredictionHook | None = cast(
                _PredictionHook | None,
                getattr(module, hook_name, None)
            )
            if hook is None or not callable(hook):
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
