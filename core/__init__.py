# This module:
# 1. Marks core as the model-facing contract package for hooks, Module,
#    optimization, metrics, hyperparameters, and checkpoint reconstruction
#
# Design decisions:
# - The package initializer exports no aggregate surface; public client classes
#   are exposed from syntheticmind while internal consumers import their owner
# - Trainer orchestration and data lifecycle remain outside core so model-facing
#   contracts do not depend on outward execution composition
#
# Author: Rahul Sawhney

__all__: list[str] = []
