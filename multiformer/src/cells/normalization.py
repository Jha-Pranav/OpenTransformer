import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm with bias=False"""

    def __init__(self, ndim, bias, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim)).to(device)  # Multiplicative
        self.bias = (
            nn.Parameter(torch.zeros(ndim)).to(device) if bias else None
        )  # Additive

    def forward(self, x):
        device = x.device

        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rvariance = torch.rsqrt(
            (x.pow(2)).mean(dim=-1, keepdim=True) + self.eps
        )  # 1/variance
        norm = (x * rvariance).type_as(x)
        return self.w * norm
