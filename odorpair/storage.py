import os
import torch
import numpy
import json
import numpy as np


def make_serializable(data):
    if isinstance(data, np.ndarray):
        return data.tolist()  # Convert ndarray to a list
    if isinstance(data, np.float32):  # Handle float32 specifically
        return float(data)
    if isinstance(data, dict):
        return {
            k: make_serializable(v) for k, v in data.items()
        }  # Recurse for dictionaries
    if isinstance(data, list):
        return [make_serializable(v) for v in data]  # Recurse for lists
    if isinstance(data, torch.Tensor):  # Handle PyTorch tensors
        return data.cpu().detach().numpy().tolist()
    return data  # Return as is for other supported types


def save(path, ckpt, model):
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(ckpt, f)

    torch.save(model.state_dict(), os.path.join(path, "model.pt"))
