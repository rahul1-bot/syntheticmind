# This module:
# 1. Exposes Trainer as the public lifecycle orchestration entry point
#
# Design decisions:
# - Configuration, connectors, loops, state, rendering, determinism, and hook
#   dispatch remain private implementation concepts behind one Trainer identity
# - Importing the package performs no registration, construction, or global
#   runtime mutation beyond loading the class definition
#
# Author: Rahul Sawhney

from syntheticmind.trainer.trainer import Trainer

__all__: list[str] = ["Trainer"]
