# This module:
# 1. Marks primitives as the dependency-inward package for framework-wide type
#    and exception vocabulary
#
# Design decisions:
# - The package initializer exports no aggregate namespace; consumers import
#   the exact types or exceptions module so dependencies remain visible
# - Algorithms, mutable runtime state, resources, and concrete implementations
#   are prohibited from entering this package
#
# Author: Rahul Sawhney

__all__: list[str] = []
