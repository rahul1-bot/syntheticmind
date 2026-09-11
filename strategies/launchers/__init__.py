# This module:
# 1. Marks launchers as the internal process-creation and parent-worker state
#    reconciliation package owned by parallel Strategies
#
# Design decisions:
# - No launcher is a public standalone runtime; clients select a Strategy start
#   method and Trainer delegates through that Strategy
# - The initializer exports nothing, preventing process machinery from becoming
#   a second user-facing execution API
#
# Author: Rahul Sawhney

__all__: list[str] = []
