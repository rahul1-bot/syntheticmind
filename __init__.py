# This module:
# 1. Exposes the stable package-root surface used by downstream research code:
#    Trainer, Module, DataModule, DatasetDataModule, and CombinedLoader
#
# Design decisions:
# - The root is a composition surface only; implementations remain in their
#   semantic packages so importing syntheticmind does not create a second owner
# - Re-exported classes preserve one canonical public identity while internal
#   modules import from their owning paths and never route through this facade
#
# Author: Rahul Sawhney

from syntheticmind.data.combined_loader import CombinedLoader
from syntheticmind.data.datamodule import DataModule, DatasetDataModule
from syntheticmind.core.module import Module
from syntheticmind.trainer.trainer import Trainer

__all__: list[str] = [
    "CombinedLoader",
    "DataModule",
    "DatasetDataModule",
    "Module",
    "Trainer"
]
