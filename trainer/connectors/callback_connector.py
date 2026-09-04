import warnings
from collections.abc import Callable, Sequence
from datetime import timedelta

from syntheticmind.callbacks.callback import Callback
from syntheticmind.callbacks.checkpointing import Checkpoint, ModelCheckpoint
from syntheticmind.callbacks.model_summary import ModelSummary
from syntheticmind.callbacks.progress_bar import ProgressBarCallback
from syntheticmind.callbacks.timer import Timer
from syntheticmind.core.module import Module
from syntheticmind.primitives.exceptions import MisconfigurationError
from syntheticmind.primitives.types import TrainerStage

__all__: list[str] = []

type _CallbackConfiguration = Callback | Sequence[Callback] | None


class _CallbackConnector:
    def __init__(
        self,
        callbacks: Callback | list[Callback] | None,
        enable_checkpointing: bool,
        enable_progress_bar: bool,
        enable_model_summary: bool,
        max_time: str | timedelta | dict[str, int] | None
    ) -> None:
        self._callbacks: list[Callback] = self._resolve_callbacks(callbacks)
        self._model_callbacks: list[Callback] = []
        self._model_configuration: tuple[TrainerStage, Module] | None = None
        self._configure_checkpointing(enable_checkpointing)
        self._configure_progress(enable_progress_bar, max_time)
        self._configure_model_summary(enable_model_summary)
        self._callbacks: list[Callback] = self._reorder(self._callbacks)
        self.validate_state_keys()

    def configure_model_callbacks(
        self,
        module: Module,
        stage: TrainerStage,
        configuration_operation: Callable[[], _CallbackConfiguration]
    ) -> None:
        active_configuration: tuple[TrainerStage, Module] | None = (
            self._model_configuration
        )
        if (
            active_configuration is not None
            and active_configuration[0] == stage
            and active_configuration[1] is module
        ):
            return
        self._model_configuration: tuple[TrainerStage, Module] | None = (
            stage,
            module
        )
        try:
            configured_callbacks: _CallbackConfiguration = configuration_operation()
            model_callbacks: list[Callback] = self._normalize_model_callbacks(
                configured_callbacks
            )
            if not model_callbacks:
                return
            previous_model_callback_ids: set[int] = {
                id(callback) for callback in self._model_callbacks
            }
            trainer_callbacks: list[Callback] = [
                callback
                for callback in self._callbacks
                if id(callback) not in previous_model_callback_ids
            ]
            model_callback_types: set[type[Callback]] = {
                type(callback) for callback in model_callbacks
            }
            trainer_callback_types: set[type[Callback]] = {
                type(callback) for callback in trainer_callbacks
            }
            trainer_callback_types.discard(Callback)
            override_types: set[type[Callback]] = {
                trainer_callback_type
                for model_callback_type in model_callback_types
                for trainer_callback_type in trainer_callback_types
                if issubclass(model_callback_type, trainer_callback_type)
            }
            merged_callbacks: list[Callback] = [
                callback
                for callback in trainer_callbacks
                if type(callback) not in override_types
            ]
            merged_callbacks.extend(model_callbacks)
            self.validate_state_keys(merged_callbacks)
            self._callbacks: list[Callback] = self._reorder(merged_callbacks)
            self._model_callbacks: list[Callback] = model_callbacks
        except BaseException:
            self._model_configuration: tuple[TrainerStage, Module] | None = None
            raise

    def clear_model_configuration(self, stage: TrainerStage) -> None:
        active_configuration: tuple[TrainerStage, Module] | None = (
            self._model_configuration
        )
        if active_configuration is not None and active_configuration[0] == stage:
            self._model_configuration: tuple[TrainerStage, Module] | None = None

    def validate_state_keys(
        self,
        callbacks: Sequence[Callback] | None = None
    ) -> None:
        callback_group: Sequence[Callback] = (
            self._callbacks if callbacks is None else callbacks
        )
        seen: dict[str, str] = {}
        callback: Callback
        for callback in callback_group:
            if (
                type(callback).state_dict is not Callback.state_dict
                or type(callback).load_state_dict is not Callback.load_state_dict
            ):
                key: str = callback.state_key
                if key in seen:
                    raise MisconfigurationError(
                        f"Duplicate callback state_key {key!r} between "
                        f"{seen[key]} and {callback!r}. Each stateful callback "
                        "must have a unique state_key."
                    )
                seen[key] = repr(callback)

    @property
    def callbacks(self) -> list[Callback]:
        return self._callbacks

    @callbacks.setter
    def callbacks(self, callbacks: list[Callback]) -> None:
        if not isinstance(callbacks, list) or not all(
            isinstance(callback, Callback) for callback in callbacks
        ):
            raise TypeError("callbacks must contain only Callback instances.")
        self.validate_state_keys(callbacks)
        self._callbacks: list[Callback] = self._reorder(callbacks)

    def _resolve_callbacks(
        self,
        callbacks: Callback | list[Callback] | None
    ) -> list[Callback]:
        if callbacks is None:
            return []
        if isinstance(callbacks, Callback):
            return [callbacks]
        if isinstance(callbacks, list) and all(
            isinstance(callback, Callback) for callback in callbacks
        ):
            return list(callbacks)
        raise TypeError(
            "`callbacks` must be a Callback, a list of Callback instances, or None."
        )

    def _configure_checkpointing(self, enabled: bool) -> None:
        checkpoint_callbacks: list[Checkpoint] = [
            callback
            for callback in self._callbacks
            if isinstance(callback, Checkpoint)
        ]
        if checkpoint_callbacks and not enabled:
            raise MisconfigurationError(
                "Trainer was configured with enable_checkpointing=False but found "
                "a Checkpoint callback in callbacks."
            )
        if not checkpoint_callbacks and enabled:
            self._callbacks.append(ModelCheckpoint())

    def _configure_progress(
        self,
        enabled: bool,
        max_time: str | timedelta | dict[str, int] | None
    ) -> None:
        progress_callbacks: list[ProgressBarCallback] = [
            callback
            for callback in self._callbacks
            if isinstance(callback, ProgressBarCallback)
        ]
        if len(progress_callbacks) > 1:
            raise MisconfigurationError(
                "Only one progress bar callback is supported. Remove duplicate "
                "ProgressBarCallback instances from callbacks."
            )
        if not enabled and progress_callbacks:
            raise MisconfigurationError(
                "Trainer was configured with enable_progress_bar=False but received "
                "a ProgressBarCallback in callbacks."
            )
        timer_callbacks: list[Timer] = [
            callback
            for callback in self._callbacks
            if isinstance(callback, Timer)
        ]
        if max_time is not None:
            if timer_callbacks:
                warnings.warn(
                    "Ignoring Trainer(max_time=...) because callbacks already contain a Timer.",
                    UserWarning,
                    stacklevel=3
                )
            else:
                timer: Timer = Timer(
                    duration=max_time,
                    interval="step",
                    verbose=False
                )
                self._callbacks.append(timer)
                timer_callbacks.append(timer)
        if not enabled:
            return
        progress_index: int | None = next(
            (
                index
                for index, callback in enumerate(self._callbacks)
                if isinstance(callback, ProgressBarCallback)
            ),
            None
        )
        if not timer_callbacks:
            timer: Timer = Timer(verbose=False)
            if progress_index is None:
                self._callbacks.append(timer)
            else:
                self._callbacks.insert(progress_index, timer)
        if not progress_callbacks:
            self._callbacks.append(ProgressBarCallback())

    def _configure_model_summary(self, enabled: bool) -> None:
        if enabled and not any(
            isinstance(callback, ModelSummary) for callback in self._callbacks
        ):
            self._callbacks.append(ModelSummary())

    def _normalize_model_callbacks(
        self,
        configured_callbacks: _CallbackConfiguration
    ) -> list[Callback]:
        if configured_callbacks is None:
            return []
        if isinstance(configured_callbacks, Callback):
            return [configured_callbacks]
        if not isinstance(configured_callbacks, Sequence):
            raise TypeError(
                "configure_callbacks() must return a Callback or a sequence of callbacks."
            )
        model_callbacks: list[Callback] = []
        callback: Callback
        for callback in configured_callbacks:
            if not isinstance(callback, Callback):
                raise TypeError(
                    "configure_callbacks() must return only Callback instances."
                )
            model_callbacks.append(callback)
        return model_callbacks

    def _reorder(self, callbacks: Sequence[Callback]) -> list[Callback]:
        regular_callbacks: list[Callback] = [
            callback
            for callback in callbacks
            if not isinstance(callback, Checkpoint)
        ]
        checkpoint_callbacks: list[Callback] = [
            callback
            for callback in callbacks
            if isinstance(callback, Checkpoint)
        ]
        return regular_callbacks + checkpoint_callbacks
