import torch
import torch.nn as nn
from src.cells.attention import GQMultiHeadAttention
from src.cells.feedforward import FeedForward
from src.cells.normalization import RMSLayerNorm


class Block(nn.Module):
    def __init__(self, args, is_causal=True, attn_mask=None) -> None:
        super().__init__()
        self.is_causal = is_causal
        self.attn_mask = attn_mask
        self.norms = RMSLayerNorm(args.embedding_dim, eps=args.rms_norm_eps)
        self.attention = GQMultiHeadAttention(args)
        self.mlp = FeedForward(args.embedding_dim, args.mlp_hidden_size, dropout=args.mlp_dropout)

    def forward(self, x, rope_q, rope_k):
        x = x + self.attention(self.norms(x), rope_q, rope_k, self.is_causal, self.attn_mask)
        x = x + self.mlp(self.norms(x))
        return x
