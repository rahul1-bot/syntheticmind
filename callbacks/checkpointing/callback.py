# This module:
# 1. Defines Checkpoint, the semantic callback category for checkpoint-producing
#    lifecycle extensions
#
# Design decisions:
# - The category adds no parallel hook surface; it lets callback composition and
#   client extensions identify checkpoint policy without coupling to the
#   concrete ModelCheckpoint implementation
# - Serialization, topology, and transport remain owned by Trainer, Strategy,
#   and CheckpointIO respectively
#
# Author: Rahul Sawhney

from syntheticmind.callbacks.callback import Callback

__all__: list[str] = ["Checkpoint"]


class Checkpoint(Callback):
    # Marker base for checkpoint policies that operate through Callback hooks.
    pass
