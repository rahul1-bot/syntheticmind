from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from syntheticmind.plugins.environments import ClusterEnvironment
from syntheticmind.plugins.io import CheckpointIO
from syntheticmind.plugins.layer_sync import LayerSync, TorchSyncBatchNorm
from syntheticmind.precision.precision import Precision
from syntheticmind.utilities.exceptions import MisconfigurationError

__all__: list[str] = ["Plugin", "PluginConfiguration", "PluginResolver"]

type Plugin = Precision | CheckpointIO | ClusterEnvironment | LayerSync


class PluginConfiguration(BaseModel):
    precision: Precision | None
    checkpoint_io: CheckpointIO | None
    cluster_environment: ClusterEnvironment | None
    layer_sync: LayerSync | None
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        strict=True,
        validate_default=True
    )


class PluginResolver:
    @classmethod
    def resolve(
        cls: type[PluginResolver],
        plugins: Plugin | list[Plugin] | None,
        sync_batchnorm: bool
    ) -> PluginConfiguration:
        del cls
        if type(sync_batchnorm) is not bool:
            raise TypeError("sync_batchnorm must be a bool.")
        plugin_list: list[Plugin]
        if plugins is None:
            plugin_list: list[Plugin] = []
        elif isinstance(
            plugins,
            (Precision, CheckpointIO, ClusterEnvironment, LayerSync)
        ):
            plugin_list: list[Plugin] = [plugins]
        elif isinstance(plugins, list):
            plugin_list: list[Plugin] = list(plugins)
        else:
            raise TypeError(
                "plugins must be a plugin instance, a plugin list, or None."
            )
        precision: Precision | None = None
        checkpoint_io: CheckpointIO | None = None
        cluster_environment: ClusterEnvironment | None = None
        layer_sync: LayerSync | None = None
        duplicate_categories: list[str] = []
        plugin: Plugin
        for plugin in plugin_list:
            if isinstance(plugin, Precision):
                if precision is not None:
                    duplicate_categories.append("Precision")
                precision: Precision | None = plugin
            elif isinstance(plugin, CheckpointIO):
                if checkpoint_io is not None:
                    duplicate_categories.append("CheckpointIO")
                checkpoint_io: CheckpointIO | None = plugin
            elif isinstance(plugin, ClusterEnvironment):
                if cluster_environment is not None:
                    duplicate_categories.append("ClusterEnvironment")
                cluster_environment: ClusterEnvironment | None = plugin
            elif isinstance(plugin, LayerSync):
                if layer_sync is not None:
                    duplicate_categories.append("LayerSync")
                layer_sync: LayerSync | None = plugin
            else:
                raise MisconfigurationError(
                    f"Unsupported plugin {plugin!r}. Expected Precision, CheckpointIO, "
                    "ClusterEnvironment, or LayerSync."
                )
        if duplicate_categories:
            categories: str = ", ".join(dict.fromkeys(duplicate_categories))
            raise MisconfigurationError(
                f"Received multiple values for {categories} in plugins."
            )
        if sync_batchnorm:
            if layer_sync is not None and not isinstance(layer_sync, TorchSyncBatchNorm):
                raise MisconfigurationError(
                    "sync_batchnorm conflicts with the configured LayerSync plugin."
                )
            layer_sync: LayerSync | None = layer_sync or TorchSyncBatchNorm()
        return PluginConfiguration(
            precision=precision,
            checkpoint_io=checkpoint_io,
            cluster_environment=cluster_environment,
            layer_sync=layer_sync
        )
