# This module:
# 1. Marks loops as the iterative fit, epoch, evaluation, prediction, validation
#    cadence, data-fetch, and optimization execution package
#
# Design decisions:
# - The initializer exports no parallel loop API; Trainer constructs and connects
#   the canonical loop classes directly
# - Loops own iteration and cadence only, while Trainer owns outer stage setup,
#   shared batch transfer, resources, and teardown
#
# Author: Rahul Sawhney

__all__: list[str] = []
