import json
import gcn
import os

def _load_model(model_path):
    production_path = os.path.join("Production",model_path)

    with open(os.path.join(production_path,"config.json")) as f:
      config = json.load(f)

    graph_model = gcn.GCN(**config)
    model_weights = torch.load(os.path.join(production_path,"model.pt"),weights_only=True)
    graph_model.load_state_dict(model_weights)
    graph_model.eval()
    return graph_model, config

def load_pretrained():
    return _load_model("pretrained")

def load_similarity():
    return _load_model("similarity")