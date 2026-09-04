# This module:
# 1. Defines the strict immutable Trainer construction policy and all scalar
#    normalization for limits, validation, debugging modes, clipping,
#    observability, determinism, sampling, and reload cadence
# 2. Defines the effective configuration snapshot stored in resumable checkpoints
#    and validates trajectory-critical compatibility on continuation
#
# Design decisions:
# - Configuration contains no live model, optimizer, scheduler, dataloader,
#   logger, callback, profiler, accelerator, Strategy, plugin, device, path
#   handle, or resource; Trainer and connectors retain behavioral ownership
# - Pydantic validates intrinsic deterministic invariants only and performs no
#   filesystem, hardware, environment, time, randomness, logging, or orchestration
# - Batch limits accept non-negative integer counts or fractions in [0, 1];
#   integral floats above one normalize to counts before data-specific resolution
# - Mutable duration mappings freeze into ordered tuples inside the model and are
#   reconstructed only when handed to ValidationSchedule
# - fast_dev_run and barebones are explicit policy rewrites with conflict
#   validation rather than hidden call-site branches
# - Checkpoint snapshots omit run ceilings from mismatch rejection so a client
#   may continue toward larger limits, while accumulation, data, validation,
#   clipping, deterministic, sampling, reload, and precision semantics may not
#   drift silently
#
# Author: Rahul Sawhney

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

from syntheticmind.primitives.exceptions import MisconfigurationError
from syntheticmind.primitives.types import ClipAlgorithm, PrecisionType

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


class _TrainerConfigurationSnapshot(BaseModel):
    # Effective trajectory policy persisted with a full checkpoint. It records
    # resolved rather than raw defaults so continuation compares actual behavior.
    schema_version: Literal[1] = 1
    fast_dev_run: int | bool
    max_epochs: int
    min_epochs: int
    max_steps: int
    min_steps: int | None
    limit_train_batches: int | float
    limit_val_batches: int | float
    limit_test_batches: int | float
    limit_predict_batches: int | float
    overfit_batches: int | float
    validation_interval_identity: str
    check_val_every_n_epoch: int | None
    num_sanity_val_steps: int | float
    log_every_n_steps: int
    enable_checkpointing: bool
    enable_progress_bar: bool
    enable_model_summary: bool
    accumulate_grad_batches: int
    gradient_clip_val: int | float | None
    gradient_clip_algorithm: ClipAlgorithm | None
    deterministic: bool | Literal["warn"] | None
    inference_mode: bool
    use_distributed_sampler: bool
    detect_anomaly: bool
    barebones: bool
    sync_batchnorm: bool
    reload_dataloaders_every_n_epochs: int
    enable_autolog_hparams: bool
    precision: PrecisionType
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        validate_default=True
    )

    def resume_mismatches(
        self,
        current: _TrainerConfigurationSnapshot
    ) -> tuple[str, ...]:
        # Returns trajectory-changing fields that differ from the current run.
        # Maximum and minimum ceilings are intentionally absent so continuation
        # can extend a previously bounded experiment.
        mismatches: list[str] = []
        if self.fast_dev_run != current.fast_dev_run:
            mismatches.append("fast_dev_run")
        if self.limit_train_batches != current.limit_train_batches:
            mismatches.append("limit_train_batches")
        if self.limit_val_batches != current.limit_val_batches:
            mismatches.append("limit_val_batches")
        if self.overfit_batches != current.overfit_batches:
            mismatches.append("overfit_batches")
        if self.validation_interval_identity != current.validation_interval_identity:
            mismatches.append("val_check_interval")
        if self.check_val_every_n_epoch != current.check_val_every_n_epoch:
            mismatches.append("check_val_every_n_epoch")
        if self.accumulate_grad_batches != current.accumulate_grad_batches:
            mismatches.append("accumulate_grad_batches")
        if self.gradient_clip_val != current.gradient_clip_val:
            mismatches.append("gradient_clip_val")
        if self.gradient_clip_algorithm != current.gradient_clip_algorithm:
            mismatches.append("gradient_clip_algorithm")
        if self.deterministic != current.deterministic:
            mismatches.append("deterministic")
        if self.use_distributed_sampler != current.use_distributed_sampler:
            mismatches.append("use_distributed_sampler")
        if self.barebones != current.barebones:
            mismatches.append("barebones")
        if self.sync_batchnorm != current.sync_batchnorm:
            mismatches.append("sync_batchnorm")
        if (
            self.reload_dataloaders_every_n_epochs
            != current.reload_dataloaders_every_n_epochs
        ):
            mismatches.append("reload_dataloaders_every_n_epochs")
        if self.precision != current.precision:
            mismatches.append("precision")
        return tuple(mismatches)

    @classmethod
    def capture(
        cls,
        configuration: _TrainerConfiguration,
        precision: PrecisionType,
        max_epochs: int
    ) -> _TrainerConfigurationSnapshot:
        # Materializes one immutable snapshot from resolved construction policy,
        # selected precision, and the effective maximum epoch horizon.
        return cls(
            fast_dev_run=configuration.fast_dev_run,
            max_epochs=max_epochs,
            min_epochs=configuration.resolved_min_epochs,
            max_steps=configuration.resolved_max_steps,
            min_steps=configuration.min_steps,
            limit_train_batches=configuration.resolved_limit_train_batches,
            limit_val_batches=configuration.resolved_limit_val_batches,
            limit_test_batches=configuration.resolved_limit_test_batches,
            limit_predict_batches=configuration.resolved_limit_predict_batches,
            overfit_batches=configuration.overfit_batches,
            validation_interval_identity=cls._validation_interval_identity(
                configuration.val_check_interval
            ),
            check_val_every_n_epoch=(
                configuration.resolved_check_val_every_n_epoch
            ),
            num_sanity_val_steps=configuration.resolved_num_sanity_val_steps,
            log_every_n_steps=configuration.resolved_log_every_n_steps,
            enable_checkpointing=configuration.resolved_enable_checkpointing,
            enable_progress_bar=configuration.resolved_enable_progress_bar,
            enable_model_summary=configuration.resolved_enable_model_summary,
            accumulate_grad_batches=configuration.accumulate_grad_batches,
            gradient_clip_val=configuration.gradient_clip_val,
            gradient_clip_algorithm=configuration.gradient_clip_algorithm,
            deterministic=configuration.deterministic,
            inference_mode=configuration.inference_mode,
            use_distributed_sampler=configuration.use_distributed_sampler,
            detect_anomaly=configuration.detect_anomaly,
            barebones=configuration.barebones,
            sync_batchnorm=configuration.sync_batchnorm,
            reload_dataloaders_every_n_epochs=(
                configuration.reload_dataloaders_every_n_epochs
            ),
            enable_autolog_hparams=configuration.enable_autolog_hparams,
            precision=precision
        )

    @classmethod
    def _validation_interval_identity(
        cls,
        interval: _ValidationInterval
    ) -> str:
        # Creates a stable, type-qualified cadence identity; float hex encoding
        # avoids equality drift from decimal rendering.
        del cls
        if interval is None:
            return "default"
        if isinstance(interval, bool):
            raise TypeError("Validation interval cannot be a bool.")
        if isinstance(interval, int):
            return f"batches:{interval}"
        if isinstance(interval, float):
            return f"fraction:{interval.hex()}"
        if isinstance(interval, str):
            return f"duration:{interval}"
        if isinstance(interval, timedelta):
            return f"duration_seconds:{interval.total_seconds().hex()}"
        parts: str = ",".join(
            f"{name}={value}"
            for name, value in interval
        )
        return f"duration_parts:{parts}"


class _TrainerConfiguration(BaseModel):
    # Private scalar input model for Trainer construction. Properties expose the
    # effective policy after fast-development, overfit, barebones, and default
    # interactions have been resolved.
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
        # Accepts a bool or non-negative batch count and normalizes integer one to
        # the canonical enabled bool.
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
        # Accepts fractions, counts, or None and converts integral floats above one
        # into explicit counts.
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
        # Normalizes case and closes clipping policy to norm or value.
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
        # Replaces mutable duration mappings with an owned immutable tuple.
        del cls
        if not isinstance(value, dict):
            return value
        frozen_parts: _ValidationIntervalParts = tuple(value.items())
        return frozen_parts

    @model_validator(mode="after")
    def _validate_barebones_policy(self) -> _TrainerConfiguration:
        # Rejects explicit observability, development, and diagnostic features
        # that barebones promises to disable rather than silently ignoring them.
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
        # Selects one development epoch, an explicit ceiling, an ordinary default,
        # or an unbounded epoch horizon when another stop policy governs the run.
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
        # Applies overfit and fast-development precedence before ordinary limit.
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
        # Disables sanity work in barebones and development modes, resolves the
        # standard two-step default, and maps -1 to an unbounded pass.
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
