import torch
import torch_geometric as tg


class VectorTokenizer:
    def __init__(self, unique_vector_tuples):
        self.unique_vector_tuples = unique_vector_tuples
        # Create a dictionary to map each vector to its index
        self.vector_to_index = {
            vector: idx
            for idx, vector in enumerate(self.unique_vector_tuples)
        }

    def tokenize(self, vector):
        vector_tuple = tuple(vector.numpy())
        # Break on out of dictionary.
        idx = self.vector_to_index.get(vector_tuple,len(vector_tuple))
        return torch.tensor(idx, dtype=torch.long)

    def tokenize_batch(self, vectors):
        return torch.stack([self.tokenize(v) for v in vectors])

    def __len__(self):
        return len(self.vector_to_index) + 1


class GraphTokenizer:

    def __init__(self, dictionary):
        self.x_tokenizer = VectorTokenizer(dictionary["x"])
        self.edge_attr_tokenizer = VectorTokenizer(dictionary["edge_attr"])

    def tokenize(self, graph: tg.data.Data):
        tokenized = graph.clone()
        tokenized.x = self.x_tokenizer.tokenize_batch(tokenized.x)
        tokenized.edge_attr = self.edge_attr_tokenizer.tokenize_batch(
            tokenized.edge_attr)
        return tokenized

    def unique_x(self):
        return len(self.x_tokenizer)

    def unique_edge_attr(self):
        return len(self.edge_attr_tokenizer)
