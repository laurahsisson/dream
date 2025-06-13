import torch
import torch_geometric as tg

from odorpair import activation


class Embedding(torch.nn.Module):

    def __init__(
        self,
        use_embed,
        unique_x,
        unique_edge_attr,
        embedding_dim_x,
        embedding_dim_edge_attr,
        act_mode,
    ):
        super(Embedding, self).__init__()
        act_fn = activation.get_act_fn(act_mode)
        self.use_embed = use_embed

        if self.use_embed:
            self.embed_x = torch.nn.Embedding(unique_x, embedding_dim_x)
            self.embed_edge_attr = torch.nn.Embedding(
                unique_edge_attr, embedding_dim_edge_attr
            )
        else:
            # unique_x should be the hidden size here
            self.embed_x = torch.nn.Sequential(
                torch.nn.Linear(unique_x, embedding_dim_x), act_fn()
            )
            self.embed_edge_attr = torch.nn.Sequential(
                torch.nn.Linear(unique_edge_attr, embedding_dim_edge_attr), act_fn()
            )

    def forward(self, graph: tg.data.Data):
        return self.embed_x(graph.x), self.embed_edge_attr(graph.edge_attr)
