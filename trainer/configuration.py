from datetime import timedelta
from typing import Annotated, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator
)

from syntheticmind.exceptions import MisconfigurationError
from syntheticmind.types import ClipAlgorithm

__all__: list[str] = []

type _ValidationIntervalParts = tuple[tuple[str, int], ...]
type _ValidationInterval = (
    int
    | float
    | str
    | timedelta
    | _ValidationIntervalParts
    | None
)
type _ValidationIntervalInput = (
    int
    | float
    | str
    | timedelta
    | dict[str, int]
    | None
)


class _TrainerConfiguration(BaseModel):
    fast_dev_run: int | bool = False
    max_epochs: Annotated[int, Field(ge=-1)] | None = None
    min_epochs: Annotated[int, Field(ge=0)] | None = None
    max_steps: Annotated[int, Field(ge=-1)] = -1
    min_steps: Annotated[int, Field(ge=0)] | None = None
    limit_train_batches: int | float | None = None
    limit_val_batches: int | float | None = None
    limit_test_batches: int | float | None = None
    limit_predict_batches: int | float | None = None
    overfit_batches: int | float = 0.0
    val_check_interval: _ValidationInterval = None
    check_val_every_n_epoch: Annotated[int, Field(ge=1)] | None = 1
    num_sanity_val_steps: Annotated[int, Field(ge=-1)] | None = None
    log_every_n_steps: Annotated[int, Field(ge=0)] | None = None
    enable_checkpointing: bool | None = None
    enable_progress_bar: bool | None = None
    enable_model_summary: bool | None = None
    accumulate_grad_batches: Annotated[int, Field(ge=1)] = 1
    gradient_clip_val: Annotated[int | float, Field(ge=0)] | None = None
    gradient_clip_algorithm: ClipAlgorithm | None = None
    deterministic: bool | Literal["warn"] | None = None
    benchmark: bool | None = None
    inference_mode: bool = True
    use_distributed_sampler: bool = True
    detect_anomaly: bool = False
    barebones: bool = False
    sync_batchnorm: bool = False
    reload_dataloaders_every_n_epochs: Annotated[int, Field(ge=0)] = 0
    enable_autolog_hparams: bool = True
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        validate_default=True
    )

    @field_validator("fast_dev_run")
    @classmethod
    def _normalize_fast_dev_run(cls, value: int | bool) -> int | bool:
        del cls
        if value < 0:
            raise MisconfigurationError(
                f"`fast_dev_run={value!r}` is invalid. Expected a non-negative "
                "integer or bool."
            )
        return True if value == 1 else value

    @field_validator(
        "limit_train_batches",
        "limit_val_batches",
        "limit_test_batches",
        "limit_predict_batches",
        "overfit_batches"
    )
    @classmethod
    def _validate_batch_limit(
        cls,
        value: int | float | None,
        info: ValidationInfo
    ) -> int | float | None:
        del cls
        if value is None:
            return None
        if 0 <= value <= 1:
            return value
        if value > 1:
            if isinstance(value, int):
                return value
            if value.is_integer():
                return int(value)
        raise MisconfigurationError(
            f"Invalid value {value!r} for `{info.field_name}`. Expected a float "
            "in [0.0, 1.0] or a non-negative integer."
        )

    @field_validator("gradient_clip_algorithm", mode="before")
    @classmethod
    def _normalize_gradient_clip_algorithm(
        cls,
        value: str | None
    ) -> ClipAlgorithm | None:
        del cls
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("`gradient_clip_algorithm` must be a string or None.")
        normalized_value: str = value.lower()
        if normalized_value not in {"norm", "value"}:
            raise MisconfigurationError(
                f"`gradient_clip_algorithm` {value} is invalid. Allowed algorithms: "
                "('norm', 'value')."
            )
        return "norm" if normalized_value == "norm" else "value"

    @field_validator("val_check_interval", mode="before")
    @classmethod
    def _freeze_validation_interval(
        cls,
        value: _ValidationIntervalInput
    ) -> _ValidationInterval:
        del cls
        if not isinstance(value, dict):
            return value
        frozen_parts: _ValidationIntervalParts = tuple(value.items())
        return frozen_parts

    @model_validator(mode="after")
    def _validate_barebones_policy(self) -> _TrainerConfiguration:
        if not self.barebones:
            return self
        if self.enable_checkpointing:
            raise MisconfigurationError(
                "`Trainer(barebones=True, enable_checkpointing=True)` was passed. "
                "Checkpointing is disabled in barebones mode."
            )
        if self.enable_progress_bar:
            raise MisconfigurationError(
                "`Trainer(barebones=True, enable_progress_bar=True)` was passed. "
                "Progress presentation is disabled in barebones mode."
            )
        if self.log_every_n_steps not in {None, 0}:
            raise MisconfigurationError(
                "`Trainer(barebones=True, log_every_n_steps>0)` was passed. "
                "Metric persistence is disabled in barebones mode."
            )
        if self.enable_model_summary:
            raise MisconfigurationError(
                "`Trainer(barebones=True, enable_model_summary=True)` was passed. "
                "Model inspection is disabled in barebones mode."
            )
        if self.num_sanity_val_steps not in {None, 0}:
            raise MisconfigurationError(
                "`Trainer(barebones=True, num_sanity_val_steps>0)` was passed. "
                "Sanity validation is disabled in barebones mode."
            )
        if self.fast_dev_run is not False and self.fast_dev_run != 0:
            raise MisconfigurationError(
                "`Trainer(barebones=True, fast_dev_run=...)` was passed. "
                "Development runs are disabled in barebones mode."
            )
        if self.detect_anomaly:
            raise MisconfigurationError(
                "`Trainer(barebones=True, detect_anomaly=True)` was passed. "
                "Anomaly detection is disabled in barebones mode."
            )
        return self

    @property
    def fast_dev_run_enabled(self) -> bool:
        return bool(self.fast_dev_run)

    @property
    def overfit_batches_enabled(self) -> bool:
        return self.overfit_batches > 0

    @property
    def resolved_max_epochs(self) -> int:
        if self.fast_dev_run_enabled:
            return 1
        if self.max_epochs is not None:
            return self.max_epochs
        return 1000 if self.max_steps == -1 else -1

    @property
    def resolved_min_epochs(self) -> int:
        if self.min_epochs is not None:
            return self.min_epochs
        return 1 if self.min_steps is not None else 0

    @property
    def resolved_max_steps(self) -> int:
        return int(self.fast_dev_run) if self.fast_dev_run_enabled else self.max_steps

    @property
    def resolved_limit_train_batches(self) -> int | float:
        if self.overfit_batches_enabled:
            return self.overfit_batches
        if self.fast_dev_run_enabled:
            return int(self.fast_dev_run)
        return self._resolve_batch_limit(self.limit_train_batches)

    @property
    def resolved_limit_val_batches(self) -> int | float:
        if self.overfit_batches_enabled:
            return self.overfit_batches
        if self.fast_dev_run_enabled:
            return int(self.fast_dev_run)
        return self._resolve_batch_limit(self.limit_val_batches)

    @property
    def resolved_limit_test_batches(self) -> int | float:
        if self.fast_dev_run_enabled:
            return int(self.fast_dev_run)
        return self._resolve_batch_limit(self.limit_test_batches)

    @property
    def resolved_limit_predict_batches(self) -> int | float:
        if self.fast_dev_run_enabled:
            return int(self.fast_dev_run)
        return self._resolve_batch_limit(self.limit_predict_batches)

    @property
    def resolved_val_check_interval(
        self
    ) -> int | float | str | timedelta | dict[str, int] | None:
        if self.fast_dev_run_enabled:
            return 1.0
        if isinstance(self.val_check_interval, tuple):
            return dict(self.val_check_interval)
        return self.val_check_interval

    @property
    def resolved_check_val_every_n_epoch(self) -> int | None:
        return 1 if self.fast_dev_run_enabled else self.check_val_every_n_epoch

    @property
    def resolved_num_sanity_val_steps(self) -> int | float:
        if self.barebones or self.fast_dev_run_enabled:
            return 0
        if self.num_sanity_val_steps is None:
            return 2
        return (
            float("inf")
            if self.num_sanity_val_steps == -1
            else self.num_sanity_val_steps
        )

    @property
    def resolved_log_every_n_steps(self) -> int:
        if self.barebones:
            return 0
        return 50 if self.log_every_n_steps is None else self.log_every_n_steps

    @property
    def resolved_enable_checkpointing(self) -> bool:
        if self.barebones:
            return False
        return True if self.enable_checkpointing is None else self.enable_checkpointing

    @property
    def resolved_enable_progress_bar(self) -> bool:
        if self.barebones:
            return False
        return True if self.enable_progress_bar is None else self.enable_progress_bar

    @property
    def resolved_enable_model_summary(self) -> bool:
        if self.barebones:
            return False
        return True if self.enable_model_summary is None else self.enable_model_summary

    def _resolve_batch_limit(self, value: int | float | None) -> int | float:
        return 1.0 if value is None else value
