import torch
import torch.nn as nn
import torch.nn.functional as F
from src.cells.attention import GPT2CausalSelfAttention
from src.cells.feedforward import GPT2MLP
from src.cells.normalization import LayerNorm


class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias, device=config.device)
        self.attn = GPT2CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias, device=config.device)
        self.mlp = GPT2MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
