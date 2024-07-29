import tqdm
import torch
from ogb.utils import smiles2graph
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.loader import DataLoader
import numpy as np
import data

NOTES_DIM = 130

def make(pair_dataset, disable_tqdm=False, limit=None):
    all_notes = set()
    for d in pair_dataset:
        all_notes.update(d["blend_notes"])
    all_notes = list(all_notes)

    # Create a dictionary mapping each label to a unique index
    label_to_index = {label: idx for idx, label in enumerate(all_notes)}

    def multi_hot(notes):
        # Initialize a zero tensor of the appropriate size
        multi_hot_vector = torch.zeros(len(all_notes))

        # Set the corresponding positions in the tensor to 1 for each label of the item
        for label in notes:
            index = label_to_index[label]
            multi_hot_vector[index] = 1
        return multi_hot_vector

    all_multihots = dict()
    for d in pair_dataset:
        all_multihots[(d["mol1"], d["mol2"])] = multi_hot(d["blend_notes"])

    all_smiles = set()
    for d in pair_dataset:
        all_smiles.add(d["mol1"])
        all_smiles.add(d["mol2"])

    def to_torch(graph):
        tensor_keys = ["edge_index", 'edge_feat', 'node_feat']
        for key in tensor_keys:
            graph[key] = torch.from_numpy(graph[key])
        return Data(x=graph["node_feat"].float(),
                    edge_attr=graph["edge_feat"].float(),
                    edge_index=graph["edge_index"])

    errored = 0
    graph_data = dict()
    for smiles in all_smiles:
        try:
            graph_data[smiles] = to_torch(smiles2graph(smiles))
        except AttributeError as e:
            errored += 1

    pair_to_data = dict()
    for i, d in enumerate(
            tqdm.tqdm(pair_dataset, smoothing=0, disable=disable_tqdm)):
        if not d["mol1"] in graph_data or not d["mol2"] in graph_data:
            continue
        pair = (d["mol1"], d["mol2"])
        g1 = graph_data[d["mol1"]]
        g2 = graph_data[d["mol2"]]
        pair_to_data[pair] = data.combine_graphs([g1, g2])

        if limit and i > limit:
            break
    valid_pairs = set(pair_to_data.keys()).intersection(
        set(all_multihots.keys()))

    dataset = []
    for (pair, graph) in pair_to_data.items():
        dataset.append({
            "pair": pair,
            "graph": graph,
            "notes": all_multihots[pair]
        })

    return dataset

def load_dream_h5(fname):
    dream_data = []
    with h5py.File(fname, 'r') as f:
      for label in tqdm(f.keys()):
        group = f[label]
        graph1 = dream.data.read_graph(group['graph1'])
        graph2 = dream.data.read_graph(group['graph2'])
        # Index using () for scalar dataset
        y = group["y"][()]
        ds = group["dataset"][()]
        dream_data.append({"graph1":graph1,"graph2":graph2,"y":torch.tensor(y),"dataset":ds.decode()})
    return dream_data
