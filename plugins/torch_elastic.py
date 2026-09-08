import os
from typing import override

from syntheticmind.plugins.environments import LocalEnvironment

__all__: list[str] = ["TorchElasticEnvironment"]


class TorchElasticEnvironment(LocalEnvironment):
    @classmethod
    @override
    def detect(cls) -> bool:
        del cls
        return (
            "TORCHELASTIC_RUN_ID" in os.environ
            and "LOCAL_RANK" in os.environ
            and "RANK" in os.environ
            and "WORLD_SIZE" in os.environ
        )

    @property
    @override
    def creates_processes_externally(self) -> bool:
        return self.detect()

    @property
    def restart_count(self) -> int:
        return int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"))

    @property
    def maximum_restarts(self) -> int:
        return int(os.environ.get("TORCHELASTIC_MAX_RESTARTS", "0"))

    @property
    def run_id(self) -> str:
        run_id: str | None = os.environ.get("TORCHELASTIC_RUN_ID")
        if not run_id:
            raise RuntimeError(
                "TorchElasticEnvironment requires TORCHELASTIC_RUN_ID."
            )
        return run_id
