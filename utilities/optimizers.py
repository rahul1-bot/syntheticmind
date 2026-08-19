# This module:
# 1. Maps a validated optimizer configuration onto the concrete torch.optim
#    implementation for the AdamW, Adam, and SGD families
#
# Design decisions:
# - Construction dispatches over the closed optimizer-name vocabulary with
#   match/case, and the default arm raises so an unknown name can never fall
#   through to a silently wrong optimizer
# - Each arm passes only the hyperparameters its optimizer understands: betas
#   for the Adam family, momentum and nesterov for SGD; the configuration
#   object carries the superset and this function performs the projection
# - The parameter iterator is taken once from the module and consumed by
#   exactly one constructor, so no arm ever receives an exhausted iterator
#
# Author: Rahul Sawhney

from collections.abc import Iterator, MutableMapping

import torch
from torch import nn

from syntheticmind.core.optimizer import _OptimizerFactoryConfig

__all__: list[str] = ["build_optimizer"]


def _move_optimizer_state_value(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, MutableMapping):
        for key, nested_value in value.items():
            value[key] = _move_optimizer_state_value(nested_value, device)
        return value
    if isinstance(value, list):
        return [_move_optimizer_state_value(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_optimizer_state_value(item, device) for item in value)
    return value


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> None:
    for parameter, state in optimizer.state.items():
        if not isinstance(state, MutableMapping):
            optimizer.state[parameter] = _move_optimizer_state_value(state, device)
            continue
        for key, value in state.items():
            if key != "step":
                state[key] = _move_optimizer_state_value(value, device)


def _optimizers_to_device(
    optimizers: list[torch.optim.Optimizer],
    device: torch.device
) -> None:
    for optimizer in optimizers:
        _optimizer_to_device(optimizer, device)


def build_optimizer(config: _OptimizerFactoryConfig, model: nn.Module) -> torch.optim.Optimizer:
    # Constructs the optimizer named by the configuration over the module's
    # parameters. The configuration is validated upstream, so this function
    # only projects fields onto the selected torch.optim constructor.
    parameters: Iterator[nn.Parameter] = model.parameters()

    match config.name:
        case "adamw":
            return torch.optim.AdamW(
                parameters,
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=config.betas
            )
        case "adam":
            return torch.optim.Adam(
                parameters,
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=config.betas
            )
        case "sgd":
            return torch.optim.SGD(
                parameters,
                lr=config.lr,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
                nesterov=config.nesterov
            )
        case _:
            raise ValueError(f"Unknown optimizer: {config.name}")
