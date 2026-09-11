# This module:
# 1. Exposes the dependency-free ModelSummary callback from the model-summary
#    concept package
#
# Design decisions:
# - RichModelSummary is not imported here because doing so would turn its
#   optional Rich dependency into a requirement for every model-summary client
#
# Author: Rahul Sawhney

from syntheticmind.callbacks.model_summary.callback import ModelSummary

__all__: list[str] = ["ModelSummary"]
