from typing import Protocol, cast, override, runtime_checkable

import torch
from torch import nn
from torch.ao.quantization.qconfig import QConfigAny

__all__: list[str] = ["LayerSync", "TorchSyncBatchNorm"]


@runtime_checkable
class LayerSync(Protocol):
    def apply(self, model: nn.Module) -> nn.Module: ...

    def revert(self, model: nn.Module) -> nn.Module: ...


class TorchSyncBatchNorm(LayerSync):
    @override
    def apply(self, model: nn.Module) -> nn.Module:
        return nn.SyncBatchNorm.convert_sync_batchnorm(model)

    @override
    def revert(self, model: nn.Module) -> nn.Module:
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
    @override
    def _check_input_dim(self, input: torch.Tensor) -> None:
        del input
