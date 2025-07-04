import numpy as np
import torch
from ogb.utils import smiles2graph
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.loader import DataLoader
from tqdm.notebook import tqdm

from odorpair import data


def to_torch(graph):
    tensor_keys = ["edge_index", "edge_feat", "node_feat"]
    for key in tensor_keys:
        graph[key] = torch.from_numpy(graph[key])
    return Data(
        x=graph["node_feat"].float(),
        edge_attr=graph["edge_feat"].float(),
        edge_index=graph["edge_index"],
    )


def smiles2torch(smiles):
    return to_torch(smiles2graph(smiles))


def convert(datapoint):
    return {
        "mol1": datapoint["edge"][0],
        "mol2": datapoint["edge"][1],
        "blend_notes": datapoint["blend_notes"],
    }


def make(pair_dataset,
         all_notes=None,
         convert_first=False,
         disable_tqdm=False,
         limit=None):
    if convert_first:
        pair_dataset = [convert(d) for d in pair_dataset]

    if all_notes is None:
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
            if not label in label_to_index:
                continue
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

    errored = 0
    graph_data = dict()
    for smiles in all_smiles:
        try:
            graph_data[smiles] = smiles2torch(smiles)
        except AttributeError as e:
            print(e)
            errored += 1

    pair_to_data = dict()
    for i, d in enumerate(tqdm(pair_dataset, smoothing=0,
                               disable=disable_tqdm)):
        if not d["mol1"] in graph_data or not d["mol2"] in graph_data:
            continue
        pair = (d["mol1"], d["mol2"])
        g1 = graph_data[d["mol1"]]
        g2 = graph_data[d["mol2"]]
        pair_to_data[pair] = data.combine_graphs([g1, g2])

    valid_pairs = set(pair_to_data.keys()).intersection(
        set(all_multihots.keys()))

    dataset = []
    for pair, graph in pair_to_data.items():
        dataset.append({
            "pair": pair,
            "graph": graph,
            "notes": all_multihots[pair]
        })

    return dataset
