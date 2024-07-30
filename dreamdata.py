import h5py
import data
import torch
import tqdm

def load_dream_h5(fname):
    dream_data = []
    with h5py.File(fname, 'r') as f:
      for idx in tqdm.tqdm(f.keys()):
        group = f[idx]
        graph1 = data.read_graph(group['graph1'])
        graph2 = data.read_graph(group['graph2'])
        # Index using () for scalar dataset
        ds = group["dataset"][()]
        label = group["label"][()]
        mixture1 = group["mixture1"][()]
        mixture2 = group["mixture2"][()]
        entry = {"idx":int(idx),"label":label,"mixture1":mixture1,"mixture2":mixture2,"graph1":graph1,"graph2":graph2,"dataset":ds.decode()}
        if "y" in group:
            y = group["y"][()]
            entry["y"] = torch.tensor(y)
        dream_data.append(entry)

    dream_data = sorted(dream_data,key=lambda d:d['idx'])
    return dream_data
