import os

import torch
from torch import nn

_TORCH_COMPILE_WRAPPER_PREFIX = "_orig_mod."


def torch_compile(model: nn.Module, dynamic=True, TORCH_COMPILE_BACKEND="torch_tensorrt") -> None:
    """Wraps the model with torch.compile. This function will also register a state_dict post hook
    that allows state_dicts produced with torch.compile training to behave as regular eager mode
    models. In particular, it strips away a torch.compile specific prefix added to the state_dict
    by torch.compile.

    Args:
            model (nn.Module): model to wrap with compile.

    Returns:
        None
    """
    # TORCH_COMPILE_BACKEND can be set as an env var to override default torch.compile backend.
    # Currently only used in unittesting to work around https://github.com/pytorch/torchtune/issues/676
    backend = os.environ.get("TORCH_COMPILE_BACKEND", TORCH_COMPILE_BACKEND)
    model = torch.compile(model, backend=backend, dynamic=True)
    model._register_state_dict_hook(_remove_torch_compile_prefix)
    return model


def _remove_torch_compile_prefix(model, state_dict, *args, **kwargs):
    keys = list(state_dict.keys())
    for key in keys:
        if key.startswith(_TORCH_COMPILE_WRAPPER_PREFIX):
            newkey = key[len(_TORCH_COMPILE_WRAPPER_PREFIX) :]
            state_dict[newkey] = state_dict.pop(key)
