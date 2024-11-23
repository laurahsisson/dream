import torch_geometric as tg
import torch
import h5py
import numpy as np

INDEX_KEYS = {"edge_index", "mol_batch", "blend_batch"}


class BlendData(tg.data.Data):
    def __inc__(self, key, value, *args, **kwargs):
        # Used for indexing the molecule into each batch
        # Each blend has only 1 blend (by definition)
        if key == 'blend_batch':
            return 1
        return super().__inc__(key, value, *args, **kwargs)

def combine_graphs(graphs):
    # Start with empty tensors for concatenation
    x_list, edge_index_list, edge_attr_list = [], [], []
    mol_batch_list = []
    current_node_index = 0

    for i, graph in enumerate(graphs):
        x_list.append(graph.x)
        edge_attr_list.append(graph.edge_attr)
        edge_index_list.append(graph.edge_index + current_node_index)
        
        # `mol_batch` needs to mark the molecule index for each node
        mol_batch_list.append(torch.full((graph.x.size(0),), i, dtype=torch.long))

        # Update node index offset for the next graph
        current_node_index += graph.x.size(0)

    # Concatenate all lists into single tensors
    x = torch.cat(x_list, dim=0)
    edge_attr = torch.cat(edge_attr_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    mol_batch = torch.cat(mol_batch_list, dim=0)
    
    # Create a blend_batch tensor of zeros for each node
    blend_batch = torch.zeros(len(graphs), dtype=torch.long)

    return BlendData(x=x, edge_attr=edge_attr, edge_index=edge_index, mol_batch=mol_batch, blend_batch=blend_batch)


def read_graph(graph_group: h5py._hl.group.Group):
    graph_data = {k: torch.tensor(np.array(v)) for k, v in graph_group.items()}
    graph_data = {
        k: v.long() if k in INDEX_KEYS else v.float()
        for k, v in graph_data.items()
    }
    return BlendData(**graph_data)


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
