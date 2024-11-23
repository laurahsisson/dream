import h5py
import data
import torch
import tqdm


def read_string(group, key):
    # Index using () for scalar dataset
    return group[key][()].decode()


def read_tensor(group, key):
    # Index using () for scalar dataset
    return torch.tensor(group[key][()]).float()


def load_dream_h5(fname):
    dream_data = []
    with h5py.File(fname, 'r') as f:
        for idx in tqdm.tqdm(f.keys()):
            group = f[idx]
            graph1 = data.read_graph(group['graph1'])
            graph2 = data.read_graph(group['graph2'])
            dataset = read_string(group, "dataset")
            label = read_string(group, "label")
            mixture1 = read_string(group, "mixture1")
            mixture2 = read_string(group, "mixture2")
            entry = {
                "idx": int(idx),
                "label": label,
                "mixture1": mixture1,
                "mixture2": mixture2,
                "graph1": graph1,
                "graph2": graph2,
                "dataset": dataset,
                "overlap": read_tensor(group, "overlap")
            }
            if "y" in group:
                entry["y"] = read_tensor(group, "y")
            dream_data.append(entry)

    dream_data = sorted(dream_data, key=lambda d: d['idx'])
    return dream_data
