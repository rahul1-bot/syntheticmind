from datetime import timedelta
from pathlib import Path
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__: list[str] = []


class _ModelCheckpointConfiguration(BaseModel):
    dirpath: str | Path | None = None
    filename: str | None = None
    monitor: str | None = None
    verbose: bool = False
    save_last: bool | Literal["link"] | None = None
    save_top_k: Annotated[int, Field(ge=-1)] = 1
    save_on_exception: bool = False
    save_weights_only: bool = False
    mode: Literal["min", "max"] = "min"
    auto_insert_metric_name: bool = True
    every_n_train_steps: Annotated[int, Field(ge=0)] | None = None
    train_time_interval: timedelta | None = None
    every_n_epochs: Annotated[int, Field(ge=0)] | None = None
    save_on_train_epoch_end: bool | None = None
    enable_version_counter: bool = True
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        strict=True,
        validate_default=True
    )

    @field_validator("filename", "monitor")
    @classmethod
    def _validate_optional_name(cls, value: str | None) -> str | None:
        del cls
        if value is not None and not value.strip():
            raise ValueError("Checkpoint filename and monitor must be non-empty when provided.")
        return value

    @field_validator("train_time_interval")
    @classmethod
    def _validate_time_interval(
        cls,
        value: timedelta | None
    ) -> timedelta | None:
        del cls
        if value is not None and value.total_seconds() <= 0.0:
            raise ValueError("Checkpoint time interval must be positive.")
        return value

    @model_validator(mode="after")
    def _validate_trigger_and_ranking(self) -> _ModelCheckpointConfiguration:
        trigger_count: int = sum(
            (
                self.every_n_train_steps is not None
                and self.every_n_train_steps > 0,
                self.every_n_epochs is not None and self.every_n_epochs > 0,
                self.train_time_interval is not None
            )
        )
        if trigger_count > 1:
            raise ValueError("Step, epoch, and time checkpoint triggers are mutually exclusive.")
        if self.monitor is None and self.save_top_k not in (-1, 0, 1):
            raise ValueError("save_top_k must be -1, 0, or 1 without a monitor.")
        return self

    @property
    def resolved_every_n_train_steps(self) -> int:
        if (
            self.every_n_train_steps is None
            and self.every_n_epochs is None
            and self.train_time_interval is None
        ):
            return 0
        return self.every_n_train_steps or 0

    @property
    def resolved_every_n_epochs(self) -> int:
        if (
            self.every_n_train_steps is None
            and self.every_n_epochs is None
            and self.train_time_interval is None
        ):
            return 1
        return self.every_n_epochs or 0
