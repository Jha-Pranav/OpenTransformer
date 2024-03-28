import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT2CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert (
            config.n_embd % config.n_head == 0
        ), "embedding dim should be divisible by head dim"
        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias, device=config.device
        )

        self.c_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias, device=config.device
        )

        self.dropout = config.dropout
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # In case of pytorch < 2.0
            NotImplementedError
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        return self.resid_dropout(self.c_proj(y))
