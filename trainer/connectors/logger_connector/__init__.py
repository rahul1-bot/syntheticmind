# This module:
# 1. Marks the internal experiment-observation package for metric accumulation,
#    consumer views, logger routing, and hyperparameter emission
#
# Design decisions:
# - The package exports no second logging API; clients produce observations only
#   through Module.log and concrete Logger implementations own persistence
#
# Author: Rahul Sawhney

__all__: list[str] = []
