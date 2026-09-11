# This module:
# 1. Marks rng as the stochastic-stream package used by Trainer, checkpoint,
#    launcher, and observational model-inspection paths
#
# Design decisions:
# - The package exports no public convenience facade because clients seed runs
#   through Trainer; internal consumers import the concrete streams owner
# - Deterministic kernel policy is deliberately absent and remains owned by
#   trainer/determinism.py, since controlled streams and deterministic
#   algorithms are related reproducibility inputs but not the same mechanism
#
# Author: Rahul Sawhney

__all__: list[str] = []
