# This module:
# 1. Marks syntheticmind.callbacks as the namespace containing lifecycle
#    observers and policies
#
# Design decisions:
# - The namespace does not flatten concrete callbacks; clients import each
#   concept from its defining module or semantic subpackage, keeping optional
#   dependencies and callback ownership visible
#
# Author: Rahul Sawhney

__all__: list[str] = []
