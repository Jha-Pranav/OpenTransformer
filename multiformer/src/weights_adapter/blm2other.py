import re
from typing import Dict

import torch

_FROM_HF = {
    "model.embed_tokens.weight": "tok_embd.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.mlp.linear1.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.mlp.linear3.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.mlp.linear2.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.norms.w",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.norms.w",
    "model.norm.weight": "norm.w",
    "lm_head.weight": "output.weight",
}


_FROM_META = {
    "tok_embeddings.weight": "tok_embd.weight",
    "norm.weight": "norm.w",
    "output.weight": "output.weight",
    "layers.{}.attention.wk.weight": "layers.{}.attention.wq.weight",
    "layers.{}.attention.wq.weight": "layers.{}.attention.wk.weight",
    "layers.{}.attention.wv.weight": "layers.{}.attention.wv.weight",
    "layers.{}.attention.wo.weight": "layers.{}.attention.wo.weight",
    "layers.{}.attention_norm.weight": "layers.{}.norms.w",
    "layers.{}.ffn_norm.weight": "layers.{}.norms.w",
    "layers.{}.feed_forward.w1.weight": "layers.{}.mlp.linear1.weight",
    "layers.{}.feed_forward.w2.weight": "layers.{}.mlp.linear2.weight",
    "layers.{}.feed_forward.w3.weight": "layers.{}.mlp.linear3.weight",
}


def _get_mapped_key(key: str, mapping_dict: Dict[str, str]) -> str:
    try:
        if "layers" in key:
            # Replace layer number with "{}" to create key for lookup
            abstract_key = re.sub(r"(\.\d+)", ".{}", key)
            layer_num = re.search(r"\d+", key).group(0)

            new_key = mapping_dict[abstract_key]

            new_key = new_key.format(layer_num)
        else:
            new_key = mapping_dict[key]
    except KeyError as e:
        raise Exception(
            f'Error converting the state dict. Found unexpected key: "{key}". '
            "Please make sure you're loading a checkpoint with the right format. "
        ) from e

    return new_key


def blm_to_hf(
    state_dict: Dict[str, torch.Tensor],
    num_heads: int = 12,
    num_kv_heads: int = 12,
    dim: int = 768,
):
    """

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in blm's format.
        num_heads (int): Number of heads in the model.
        num_kv_heads (int): Number of heads in the key/value projection layers.
        dim (int): Dimension of the model.

    Returns:
        Dict[str, torch.Tensor]: State dict in Meta's format.
    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _FROM_HF.items()}
    head_dim = dim // num_heads

    def _permute(t, n_heads):
        return (
            t.view(n_heads, head_dim // 2, 2, dim)
            .transpose(1, 2)
            .reshape((head_dim * n_heads), dim)
        )

    for key, value in state_dict.items():
        new_key = _get_mapped_key(key, inverted_mapping_dict)
        if "q_proj" in key:
            value = _permute(value, num_heads)
        elif "k_proj" in key:
            value = _permute(value, num_kv_heads)
        converted_state_dict[new_key] = value

    return converted_state_dict


def meta_to_blm(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in Meta's format.

    Returns:
        Dict[str, torch.Tensor]: State dict in blm's format.
    """
    converted_state_dict = {}
    for key, value in state_dict.items():
        if key not in ["rope.freqs"]:  # Skip loading the position embeddings
            new_key = _get_mapped_key(key, _FROM_META)
            converted_state_dict[new_key] = value

    return converted_state_dict


def blm_to_meta(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in blm's format.

    Returns:
        Dict[str, torch.Tensor]: State dict in Meta's format.
    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _FROM_META.items()}

    for key, value in state_dict.items():
        new_key = _get_mapped_key(key, inverted_mapping_dict)
        converted_state_dict[new_key] = value

    return converted_state_dict


def hf_to_blm(
    state_dict: Dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 32,
    dim: int = 4096,
    head_dim: int = None,
) -> Dict[str, torch.Tensor]:
    """Convert a state dict from HF's format to blm's format. State dicts from multiple checkpoint
    files should be consolidated into a single state dict before calling this function.

    Eg of HF-format state dict can be found in the ``meta-llama/Llama-2-7b-hf``
    repo in HF (https://huggingface.co/meta-llama/Llama-2-7b-hf).

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in Meta's format.
        num_heads (int): Number of heads in the model.
        num_kv_heads (int): Number of heads in the key/value projection layers.
        dim (int): Dimension of the model.
        head_dim (int): Dimension of the head. If not provided, it will be calculated
            as dim // num_heads.

    Returns:
        Dict[str, torch.Tensor]: State dict in blm's format.
    """
    converted_state_dict = {}
    if head_dim is None:
        head_dim = dim // num_heads

    def _permute(t, n_heads):
        return (
            t.view(n_heads, 2, head_dim // 2, dim)
            .transpose(1, 2)
            .reshape((head_dim * n_heads), dim)
        )

    for key, value in state_dict.items():
        if "rotary_emb.inv_freq" not in key:  # Skip loading the position embeddings
            new_key = _get_mapped_key(key, _FROM_HF)
            if "q_proj" in key:
                value = _permute(value, num_heads)
            elif "k_proj" in key:
                value = _permute(value, num_kv_heads)
            converted_state_dict[new_key] = value
    return converted_state_dict