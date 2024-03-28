from importlib.metadata import version
import torch

print("TORCH VERSION :", version("torch"))
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backend.mps.is_available() else "cpu"
)
print("Processor : ", device)

import torch.nn as nn
import math


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
