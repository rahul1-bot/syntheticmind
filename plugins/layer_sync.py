# This module:
# 1. Defines LayerSync, the reversible model-transformation contract applied
#    around distributed fitting
# 2. Implements conversion between ordinary BatchNorm layers and PyTorch
#    SyncBatchNorm while preserving parameters, buffers, modes, children, and
#    quantization configuration
#
# Design decisions:
# - Layer synchronization is a plugin because it transforms model structure but
#   remains independent from DDP process and collective ownership
# - apply delegates to PyTorch's recursive conversion; revert reconstructs
#   ordinary batch normalization so the client model returned after fitting does
#   not retain a distributed-only layer type
# - Reversion preserves live Parameter identities and running buffers rather than
#   copying values into unrelated state, keeping optimizer and checkpoint links
#   coherent
# - _BatchNormXd disables dimensionality specialization because the original
#   concrete BatchNorm rank is not recoverable from SyncBatchNorm alone
#
# Author: Rahul Sawhney

from typing import Protocol, cast, override, runtime_checkable

import torch
from torch import nn
from torch.ao.quantization.qconfig import QConfigAny

__all__: list[str] = ["LayerSync", "TorchSyncBatchNorm"]


@runtime_checkable
class LayerSync(Protocol):
    # Structural reversible transformation supplied through Trainer plugins.
    # Applies the distributed-only transformation before Strategy wrapping.
    def apply(self, model: nn.Module) -> nn.Module: ...

    # Reconstructs the ordinary client model after distributed execution.
    def revert(self, model: nn.Module) -> nn.Module: ...


class TorchSyncBatchNorm(LayerSync):
    # PyTorch synchronized-batch-normalization policy used with DDP.
    @override
    def apply(self, model: nn.Module) -> nn.Module:
        # Converts every eligible batch-normalization layer before DDP wrapping.
        return nn.SyncBatchNorm.convert_sync_batchnorm(model)

    @override
    def revert(self, model: nn.Module) -> nn.Module:
        # Recursively reconstructs ordinary batch normalization while retaining
        # learned affine parameters, running statistics, modes, and child modules.
        converted_model: nn.Module = model
        if isinstance(model, nn.SyncBatchNorm):
            converted_batch_norm: _BatchNormXd = _BatchNormXd(
                model.num_features,
                model.eps,
                model.momentum,
                model.affine,
                model.track_running_stats
            )
            if model.affine:
                with torch.no_grad():
                    converted_batch_norm.weight: nn.Parameter | None = model.weight
                    converted_batch_norm.bias: nn.Parameter | None = model.bias
            converted_batch_norm.running_mean: torch.Tensor | None = model.running_mean
            converted_batch_norm.running_var: torch.Tensor | None = model.running_var
            converted_batch_norm.num_batches_tracked: torch.Tensor | None = (
                model.num_batches_tracked
            )
            converted_batch_norm.train(model.training)
            if hasattr(model, "qconfig"):
                qconfig: QConfigAny = cast(QConfigAny, getattr(model, "qconfig"))
                converted_batch_norm.qconfig: QConfigAny = qconfig
            converted_model: nn.Module = converted_batch_norm
        child_name: str
        child: nn.Module
        for child_name, child in model.named_children():
            converted_model.add_module(child_name, self.revert(child))
        return converted_model


class _BatchNormXd(nn.modules.batchnorm._BatchNorm):
    # Dimension-agnostic ordinary BatchNorm used only for reversible conversion.
    @override
    def _check_input_dim(self, input: torch.Tensor) -> None:
        # Accepts any rank because SyncBatchNorm does not preserve which concrete
        # BatchNorm dimensional subclass produced it.
        del input
