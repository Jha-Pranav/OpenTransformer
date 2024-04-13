import torch
import torch.nn as nn
import torch.nn.functional as F
from src.cells.position import RotaryEmbedding


class GPT2CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0, "embedding dim should be divisible by head dim"
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

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

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


class GQMultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()

        args.num_attention_heads % args.num_key_value_heads == 0, "Head counts should be divisible KV Heads counts"
        self.group_factor = args.num_attention_heads // args.num_key_value_heads
        self.head_dim = args.emebdding_dim // args.num_attention_heads

        self.wq = nn.Linear(args.emebdding_dim, args.emebdding_dim, bias=args.attention_bias)
        self.wk = nn.Linear(
            args.emebdding_dim,
            args.emebdding_dim // self.group_factor,
            bias=args.attention_bias,
        )
        self.wv = nn.Linear(
            args.emebdding_dim,
            args.emebdding_dim // self.group_factor,
            bias=args.attention_bias,
        )
        self.wo = nn.Linear(args.emebdding_dim, args.emebdding_dim, bias=args.attention_bias)

        self.dropout = args.attention_dropout
        self.residual_dropout = nn.Dropout(args.residual_dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_q: RotaryEmbedding,
        rope_k: RotaryEmbedding,
        is_causal=True,
        attn_mask=None,
    ):
        b, seqlen, _ = x.shape
        # QKV
        xq, xk, xv = (
            self.wq(x).view(b, seqlen, -1, self.head_dim),
            self.wk(x).view(b, seqlen, -1, self.head_dim),
            self.wv(x).view(b, seqlen, -1, self.head_dim),
        )
        # RoPE on Q,K
        xq = rope_q(xq).transpose(1, 2)
        xk = rope_k(xk).transpose(1, 2)

        xv = xv.transpose(1, 2)
        if hasattr(F, "scaled_dot_product_attention"):
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            raise NotImplementedError("Upgrade to pytorch version >= 2.0")

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(b, seqlen, -1)
        return self.residual_dropout(self.wo(output))
