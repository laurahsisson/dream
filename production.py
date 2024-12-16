import json
import gcn
import os
import torch

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_model(model_path, device):
    # Build an absolute path from the script location
    production_path = os.path.join(BASE_DIR, "Production", model_path)

    with open(os.path.join(production_path, "config.json")) as f:
        config = json.load(f)

    graph_model = gcn.GCN(**config)
    model_weights = torch.load(os.path.join(production_path, "model.pt"), weights_only=True, map_location=device)
    graph_model.load_state_dict(model_weights)
    graph_model.eval()
    return graph_model, config

def load_pretrained(device='cuda'):
    return _load_model("pretrained",device)

def load_similarity(device='cuda'):
    return _load_model("similarity",device)
