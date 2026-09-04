# This module:
# 1. Marks optimization as the internal automatic and manual training-step
#    normalization package owned by TrainingEpochLoop
#
# Design decisions:
# - The package exports no client API; Module and SyntheticOptimizer provide the
#   supported optimization surface
# - Automatic and manual paths remain separate because loss ownership, backward,
#   zeroing, and stepping have fundamentally different lifecycle semantics
#
# Author: Rahul Sawhney

__all__: list[str] = []
