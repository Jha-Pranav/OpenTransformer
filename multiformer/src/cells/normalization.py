import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm with bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))  # Multiplicative
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None  # Additive

    def forward(self, x):
        device = x.device

        return F.layer_norm(
            x, self.weight.shape, self.weight, self.bias, 1e-5
        )