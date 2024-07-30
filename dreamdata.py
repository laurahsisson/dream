import h5py
import data
import torch
import tqdm

def load_dream_h5(fname):
    dream_data = []
    with h5py.File(fname, 'r') as f:
      for label in tqdm.tqdm(f.keys()):
        group = f[label]
        graph1 = data.read_graph(group['graph1'])
        graph2 = data.read_graph(group['graph2'])
        ds = group["dataset"][()]
        # Index using () for scalar dataset
        if "y" in group:
            y = group["y"][()]
            dream_data.append({"label":label,"graph1":graph1,"graph2":graph2,"y":torch.tensor(y),"dataset":ds.decode()})
        else:
            dream_data.append({"label":label,"graph1":graph1,"graph2":graph2,"dataset":ds.decode()})
    return dream_data
