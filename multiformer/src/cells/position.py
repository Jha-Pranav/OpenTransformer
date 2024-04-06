import torch
import torch.nn as nn
import math

from typing import Union


class PostionalEncoding(nn.Module):
    """
    Implementation of PE
    """

    def __init__(self, embedding_dim, max_len, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, embedding_dim)  # (max_len,embedding_dim)
        constant = math.log(10000)
        pos = torch.arange(0, max_len).unsqueeze(1)  # (max_len,1)
        # ignore multiplication with 2
        inv_den = torch.exp(
            (torch.arange(0, embedding_dim, 2, dtype=torch.float) / embedding_dim)
            * -constant
        )  # (embedding_dim/2)

        pe[:, 0::2] = torch.sin(pos * inv_den)  # (max_len,embedding_dim/2)
        pe[:, 1::2] = torch.cos(pos * inv_den)  # (max_len,embedding_dim/2)
        pe = pe.unsqueeze(0)  # (1,max_len,embedding_dim)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # here x will be the embedding vector    # (batch,seq_len,embedding_dim)
        x = x + self.pe[:, : x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(
        self,
        h_dim: int,
        seq_len: int,
        base: Union[int, float] = 10000,
        device: str = None,
        scaling_factor: Union[int, float] = 1.0,
    ):
        super().__init__()

        # Long-termdecay: θi=10000**−2i/d.
        theta = 1.0 / base ** (
            torch.arange(0, h_dim, 2, dtype=torch.int16, device=device) / h_dim
        )  #  (h_dim/2)
        m = torch.arange(seq_len, dtype=torch.int16, device=device).type_as(
            theta
        )  #  (seq_len)
        m /= scaling_factor
        m0 = torch.outer(m, theta)  # (seq_len,h_dim/2)
        self.m0 = torch.polar(torch.ones_like(m0), m0)  # e^i0 = cos0 +isin0

    def forward(self, x: torch.Tensor):

        x_complex = torch.view_as_complex(
            x.float().reshape(*x.shape[:-1], -1, 2)
        )  # (b,seq_len,d_model,head,head_dim/2,2)
        m0_transform = self.m0.unsqueeze(0).unsqueeze(
            2
        )  # (seq_len,h_dim/2) -> (1,seq_len,1,h_dim/2)
        x_rotated = x_complex * m0_transform  # (b,seq_len,d_model,head,head_dim/2,2)
        x_out = torch.view_as_real(x_rotated)
        return x_out.reshape(*x.shape)


if __name__ == "__main__":
    # Sanity check
    x = torch.rand(16, 1024, 8, 64)
    result = RotaryEmbedding(64, 1024)(x)
    print(result.shape)
