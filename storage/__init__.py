# This module:
# 1. Marks storage as the path-bound local and remote artifact infrastructure
#    package
#
# Design decisions:
# - The initializer exports no aggregate facade; framework consumers import
#   StoragePath from its owning module and public ML APIs continue accepting
#   ordinary str or Path values
#
# Author: Rahul Sawhney

__all__: list[str] = []
