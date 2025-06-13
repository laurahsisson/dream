import json
import torch
from odorpair import gcn
import importlib.resources as pkg_resources
from pathlib import Path

def _load_model(model_dir: str, device: str):
    # Access the files inside odorpair.Production.model_dir
    base_path = pkg_resources.files("odorpair.Production").joinpath(model_dir)
    config_path = base_path / "config.json"
    model_path = base_path / "model.pt"

    with config_path.open("r") as f:
        config = json.load(f)

    graph_model = gcn.GCN(**config)
    model_weights = torch.load(
        model_path,
        weights_only=True,
        map_location=device,
    )
    graph_model.load_state_dict(model_weights)
    graph_model.eval()
    return graph_model, config

def load_pretrained(device="cuda"):
    return _load_model("pretrained", device)

def load_similarity(device="cuda"):
    return _load_model("similarity", device)
