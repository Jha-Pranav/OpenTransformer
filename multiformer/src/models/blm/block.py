import torch
import torch.nn as nn
from src.cells.attention import GQMultiHeadAttention
from src.cells.feedforward import FeedForward
from src.cells.normalization import RMSLayerNorm


class Block(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.norms = RMSLayerNorm(args.emebdding_dim, eps=args.rms_norm_eps)
        self.attention = GQMultiHeadAttention(args, is_causal=True, attn_mask=None)
        self.mlp = FeedForward(args.emebdding_dim, args.mlp_hidden_size, dropout=args.mlp_dropout)

    def forward(self, x, rope_q, rope_k):
        x = x + self.attention(self.norms(x), rope_q, rope_k)
        x = x + self.mlp(self.norms(x))
        return x
