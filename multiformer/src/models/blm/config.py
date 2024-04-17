from dataclasses import dataclass

import torch


@dataclass
class ModelArgs:
    vocab_size: int = 32000  # llama2 tokenizer has 32k vocab size
    embedding_dim: int = 4096
    max_seq_len: int = 2048
    embedding_dropout: float = 0.0

    rms_norm_eps: float = 1e-05

    rope_scaling: float = 1.0
    rope_theta: float = 10000.0

    attention_bias: bool = False
    attention_dropout: float = 0.0
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    use_cache: bool = True
    use_sliding_window: bool = True
    residual_dropout: float = 0.1

    mlp_hidden_size: int = int(
        1.3 * embedding_dim
    )  # set some lambda function or scaling factor
    mlp_dropout: float = 0.0

    num_layers: int = 32

    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backend.mps.is_available() else "cpu"
    )
    padding_idx: int = 2
