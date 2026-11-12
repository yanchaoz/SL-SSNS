"""Confidence-weighted consistency losses with defined empty-mask behavior."""

from torch.nn.modules.loss import _Loss
import torch


def masked_mean(loss, weight):
    """Retain sum(weight) normalization; no selected voxels means zero loss."""
    denominator = weight.sum()
    denominator = torch.where(
        denominator == 0, torch.ones_like(denominator), denominator
    )
    return loss.sum() / denominator


class MSELoss_unlabel(_Loss):
    def __init__(self):
        super(MSELoss_unlabel, self).__init__()

    def forward(self, input_y, target, weight):
        # assert target.requires_grad is False
        weight = weight.float()
        target = target.float()
        loss = weight * ((input_y - target) ** 2)
        return masked_mean(loss, weight)


class BCELoss_unlabel(_Loss):
    def __init__(self):
        super(BCELoss_unlabel, self).__init__()

    def forward(self, input_y, target, weight):
        assert target.requires_grad is False
        input_y = torch.clamp(input_y, min=0.000001, max=0.999999)
        weight = weight.float()
        target = target.float()
        loss = -weight * (
            target * torch.log(input_y) + (1 - target) * torch.log(1 - input_y)
        )
        return masked_mean(loss, weight)
