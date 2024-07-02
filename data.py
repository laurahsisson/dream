import torch_geometric as tg
import torch
import h5py
import numpy as np

INDEX_KEYS = {"edge_index","mol_batch","blend_batch"}

class BlendData(tg.data.Data):
    def __inc__(self, key, value, *args, **kwargs):
        # Used for indexing the molecule into each batch
        # Each blend has only 1 blend (by definition)
        if key == 'blend_batch':
            return 1
        return super().__inc__(key, value, *args, **kwargs)

def combine_graphs(graphs):
    combined_batch = next(iter(tg.loader.DataLoader(graphs, batch_size=len(graphs))))
    # Index of the molecule, for each atom
    mol_batch = combined_batch.batch
    # Index of the blend, for each molecule (increment during batch)
    blend_batch = torch.zeros(len(graphs),dtype=torch.long)
    return BlendData(x=combined_batch.x,edge_attr=combined_batch.edge_attr,edge_index=combined_batch.edge_index,mol_batch=mol_batch,blend_batch=blend_batch)

def read_graph(graph_group: h5py._hl.group.Group):
    graph_data = {k: torch.tensor(np.array(v)) for k, v in graph_group.items()}
    graph_data = {k: v.long() if k in INDEX_KEYS else v.float() for k, v in graph_data.items()}
    return tg.data.Data(**graph_data)

class PairData(tg.data.Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)

    @classmethod
    def _make_batches(cls, pair_graph):
        graph1 = tg.data.Batch(x=pair_graph.x_s,
                               edge_index=pair_graph.edge_index_s,
                               edge_attr=pair_graph.edge_attr_s,
                               batch=pair_graph.x_s_batch,
                               ptr=pair_graph.x_s_ptr)

        graph2 = tg.data.Batch(x=pair_graph.x_t,
                               edge_index=pair_graph.edge_index_t,
                               edge_attr=pair_graph.edge_attr_t,
                               batch=pair_graph.x_t_batch,
                               ptr=pair_graph.x_t_ptr)
        return graph1, graph2

    @classmethod
    def _make_data(cls, pair_graph):
        graph1 = tg.data.Data(
            x=pair_graph.x_s,
            edge_index=pair_graph.edge_index_s,
            edge_attr=pair_graph.edge_attr_s,
        )

        graph2 = tg.data.Data(
            x=pair_graph.x_t,
            edge_index=pair_graph.edge_index_t,
            edge_attr=pair_graph.edge_attr_t,
        )
        return graph1, graph2

    @classmethod
    def split(cls, pair_graph):
        if isinstance(pair_graph, tg.data.Batch):
            return cls._make_batch(pair_graph)
        else:
            return cls._make_data(pair_graph)
