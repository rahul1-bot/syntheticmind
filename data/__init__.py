# This module:
# 1. Exposes the public data lifecycle and loader-composition contracts from
#    their semantic package
#
# Design decisions:
# - DataModule, DatasetDataModule, CombinedLoader, and CombinedLoaderMode are
#   re-exported together because they form the client-facing data surface
# - The initializer owns no loader construction, traversal, registration, or
#   side effects; those remain with the concrete modules
#
# Author: Rahul Sawhney

from syntheticmind.data.combined_loader import CombinedLoader, CombinedLoaderMode
from syntheticmind.data.datamodule import DataModule, DatasetDataModule

__all__: list[str] = [
    "CombinedLoader",
    "CombinedLoaderMode",
    "DataModule",
    "DatasetDataModule"
]
