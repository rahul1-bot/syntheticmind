from typing import override

import torch
from torch import nn

__all__: list[str] = ["LayerSync", "TorchSyncBatchNorm"]


class LayerSync:
    def apply(self, model: nn.Module) -> nn.Module:
        raise NotImplementedError

    def revert(self, model: nn.Module) -> nn.Module:
        raise NotImplementedError


class TorchSyncBatchNorm(LayerSync):
    @override
    def apply(self, model: nn.Module) -> nn.Module:
        return nn.SyncBatchNorm.convert_sync_batchnorm(model)

    @override
    def revert(self, model: nn.Module) -> nn.Module:
        converted_model: nn.Module = model
        if isinstance(model, nn.SyncBatchNorm):
            converted_model = _BatchNormXd(
                model.num_features,
                model.eps,
                model.momentum,
                model.affine,
                model.track_running_stats
            )
            if model.affine:
                with torch.no_grad():
                    converted_model.weight = model.weight
                    converted_model.bias = model.bias
            converted_model.running_mean = model.running_mean
            converted_model.running_var = model.running_var
            converted_model.num_batches_tracked = model.num_batches_tracked
            converted_model.train(model.training)
            if hasattr(model, "qconfig"):
                converted_model.qconfig = model.qconfig
        for name, child in model.named_children():
            converted_model.add_module(name, self.revert(child))
        return converted_model


class _BatchNormXd(nn.modules.batchnorm._BatchNorm):
    @override
    def _check_input_dim(self, input: torch.Tensor) -> None:
        del input
