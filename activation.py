import torch


def get_act_fn(act_mode):
    if act_mode == "relu":
        return torch.nn.ReLU
    if act_mode == "gelu":
        return torch.nn.GELU
    if act_mode == "silu":
        return torch.nn.SiLU
    raise ValueError(f"Unsupported activation mode: {act_mode}")
