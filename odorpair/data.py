import torch_geometric as tg
import torch
import h5py
import numpy as np

INDEX_KEYS = {"edge_index", "mol_batch", "blend_batch"}

# For use in the backbone GNN model, holds an arbitrary number of
# molecules.
class BlendData(tg.data.Data):
    def __inc__(self, key, value, *args, **kwargs):
        # Used for indexing the molecule into each batch
        # Each blend has only 1 blend (by definition)
        if key == 'blend_batch':
            return 1
        return super().__inc__(key, value, *args, **kwargs)

# Specifically for loading ordered pairs of molecules, primarily
# for use in similarity prediction.
class PairData(tg.data.Data):
    def __inc__(self, key, value, *args, **kwargs):
        # Adjust increments for edge indices to handle source and target separately
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        # Ensure that x_s and x_t follow the correct batching
        if key in ['x_s', 'x_t']:
            return 0  # Concatenate along the node feature dimension
        if key == 'y':
            return None  # Avoid concatenation for y
        return super().__cat_dim__(key, value, *args, **kwargs)

    def _get_graph(self, key):
        """
        Retrieve a source or target graph as a Data object.

        Args:
            key (str): The graph key, either 's' (source) or 't' (target).

        Returns:
            torch_geometric.data.Data: The extracted graph.

        Raises:
            ValueError: If the key is not 's' or 't'.
        """
        if key not in ['s', 't']:
            raise ValueError(f"Invalid key '{key}'. Expected 's' or 't'.")

        suffix = f"_{key}"
        graph_data = {}
        for attr, value in self.items():
            if attr.endswith(suffix):
                graph_data[attr[:-2]] = value  # Remove the suffix (_s or _t)

        return graph_data

    @classmethod
    def factory(cls, data_s, data_t, y=None):
        """
        Factory function to create a BlendData object from two source graphs and optionally a label.

        Args:
            data_s (torch_geometric.data.Data): Source graph with arbitrary attributes.
            data_t (torch_geometric.data.Data): Target graph with arbitrary attributes.
            y (optional, torch.Tensor): Label or target value for the pair.

        Returns:
            BlendData: A BlendData object containing all attributes from the source and target graphs.
        """
        blend_data = cls()
        
        # Set attributes for source graph with `_s` suffix
        for key, value in data_s.items():
            setattr(blend_data, f"{key}_s", value)
        
        # Set attributes for target graph with `_t` suffix
        for key, value in data_t.items():
            setattr(blend_data, f"{key}_t", value)
        
        # Optionally set the label
        blend_data.y = y
        
        return blend_data

    @classmethod
    def split(cls, pair_graph):
        graph_s = pair_graph._get_graph("s") 
        graph_t = pair_graph._get_graph("t")
        if isinstance(pair_graph, tg.data.Batch):
            graph_s["batch"] = pair_graph["x_s_batch"]
            graph_t["batch"] = pair_graph["x_t_batch"]
            return tg.data.Batch(**graph_s), tg.data.Batch(**graph_t)
        else:
            return tg.data.Data(**graph_s), tg.data.Data(**graph_t)

    @classmethod
    def loader(cls, dataset, batch_size, shuffle=True):
        return tg.loader.DataLoader(dataset,
                     batch_size=batch_size,
                     follow_batch=['x_s','x_t'],
                     shuffle=shuffle)


# TODO: Refactor into BlendData
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

