# This module:
# 1. Marks connectors as Trainer's private vertical composition boundaries for
#    accelerator, callback, checkpoint, data, logger, and signal lifecycles
#
# Design decisions:
# - Connectors are internal and export no alternate Trainer API
# - A connector exists only when one configuration family crosses several
#   components and requires a single validation, ordering, and attachment owner
#
# Author: Rahul Sawhney

__all__: list[str] = []
