import torch
from torch import nn
import torch.nn.functional as F
import math

from src.models.blm.config import ModelArgs
from src.models.blm.block import Block

from src.cells.normalization import RMSLayerNorm
from typing import Optional
from src.cells.position import RotaryEmbedding


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.tok_embd = nn.Embedding(args.vocab_size, args.emebdding_dim)
        self.dropout = nn.Dropout(args.embedding_dropout)
        self.rope_q = RotaryEmbedding(
            args.emebdding_dim // args.num_attention_heads,
            args.max_seq_len,
            device=args.device,
        )
        self.rope_k = RotaryEmbedding(
            args.emebdding_dim // args.num_key_value_heads,
            args.max_seq_len,
            device=args.device,
        )

        # Freeze the parameters rope_q and rope_k
        self.rope_q.requires_grad_(False)
        self.rope_k.requires_grad_(False)

        self.layers = nn.ModuleList([Block(args) for lid in range(args.num_layers)])

        self.norm = RMSLayerNorm(args.emebdding_dim, eps=args.rms_norm_eps)
        self.output = nn.Linear(args.emebdding_dim, args.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embd.weight = (
            self.output.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("wo.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * args.num_layers)
                )

    def __repr__(self):
        return f"{self.get_num_params()} Million Params Model"

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_embd.weight.numel()
        return n_params / 1e6  # In Million

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.dropout(self.tok_embd(tokens))
        for layer in self.layers:
            x = layer(
                x, self.rope_q, self.rope_k
            )  ## How about we add residual connection here also ?
        x = self.norm(x)

        loss = None
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
        return logits, loss
