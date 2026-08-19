from dataclasses import dataclass

from syntheticmind.plugins.environments import ClusterEnvironment
from syntheticmind.plugins.io import CheckpointIO
from syntheticmind.plugins.layer_sync import LayerSync, TorchSyncBatchNorm
from syntheticmind.precision.precision import Precision
from syntheticmind.utilities.exceptions import MisconfigurationError

__all__: list[str] = ["Plugin", "PluginConfiguration", "PluginResolver"]

type Plugin = Precision | CheckpointIO | ClusterEnvironment | LayerSync


@dataclass(frozen=True, slots=True)
class PluginConfiguration:
    precision: Precision | None
    checkpoint_io: CheckpointIO | None
    cluster_environment: ClusterEnvironment | None
    layer_sync: LayerSync | None


class PluginResolver:
    @classmethod
    def resolve(
        cls,
        plugins: Plugin | list[Plugin] | None,
        sync_batchnorm: bool
    ) -> PluginConfiguration:
        del cls
        plugin_list: list[Plugin]
        if plugins is None:
            plugin_list = []
        elif isinstance(
            plugins,
            (Precision, CheckpointIO, ClusterEnvironment, LayerSync)
        ):
            plugin_list = [plugins]
        elif isinstance(plugins, list):
            plugin_list = plugins
        else:
            raise TypeError(
                "plugins must be a plugin instance, a list of plugin instances, or None."
            )

        precision: Precision | None = None
        checkpoint_io: CheckpointIO | None = None
        cluster_environment: ClusterEnvironment | None = None
        layer_sync: LayerSync | None = None
        duplicate_categories: list[str] = []
        for plugin in plugin_list:
            if isinstance(plugin, Precision):
                if precision is not None:
                    duplicate_categories.append("Precision")
                precision = plugin
            elif isinstance(plugin, CheckpointIO):
                if checkpoint_io is not None:
                    duplicate_categories.append("CheckpointIO")
                checkpoint_io = plugin
            elif isinstance(plugin, ClusterEnvironment):
                if cluster_environment is not None:
                    duplicate_categories.append("ClusterEnvironment")
                cluster_environment = plugin
            elif isinstance(plugin, LayerSync):
                if layer_sync is not None:
                    duplicate_categories.append("LayerSync")
                layer_sync = plugin
            else:
                raise MisconfigurationError(
                    f"Unsupported plugin {plugin!r}. Expected Precision, CheckpointIO, "
                    "ClusterEnvironment, or LayerSync."
                )
        if duplicate_categories:
            categories: str = ", ".join(dict.fromkeys(duplicate_categories))
            raise MisconfigurationError(
                f"Received multiple values for {categories} in plugins. "
                "Configure at most one plugin from each family."
            )
        if sync_batchnorm:
            if layer_sync is not None and not isinstance(
                layer_sync,
                TorchSyncBatchNorm
            ):
                raise MisconfigurationError(
                    "Trainer(sync_batchnorm=True) conflicts with a custom LayerSync plugin. "
                    "Configure one synchronization policy."
                )
            layer_sync = layer_sync or TorchSyncBatchNorm()
        return PluginConfiguration(
            precision=precision,
            checkpoint_io=checkpoint_io,
            cluster_environment=cluster_environment,
            layer_sync=layer_sync
        )
