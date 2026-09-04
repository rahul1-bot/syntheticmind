# This module:
# 1. Marks the internal full and sharded model-parallel checkpoint capability
#    owned exclusively by ModelParallelStrategy
#
# Design decisions:
# - The package exports no client-facing transport or storage API; clients select
#   checkpoint format through ModelParallelStrategy
# - Generation publication, distributed state representation, and sharded store
#   behavior remain separate internal owners under one semantic capability
#
# Author: Rahul Sawhney

__all__: list[str] = []
