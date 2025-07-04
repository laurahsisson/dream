import torch
import torch_geometric as tg

from odorpair import aggregate, embedding, mpnn, utils


class Encoder(torch.nn.Module):

    def __init__(self, graph_tokenizer, mpnn_configs, embedding_dim_x,
                 embedding_dim_edge_attr, do_edge_update, do_two_stage,
                 num_sabs, heads, dropout, act_mode, aggr_mode, **kwargs):
        super(Encoder, self).__init__()

        use_embed = not graph_tokenizer is None
        if use_embed:
            unique_x, unique_edge_attr = (
                graph_tokenizer.unique_x(),
                graph_tokenizer.unique_edge_attr(),
            )
        else:
            unique_x, unique_edge_attr = 9, 3

        self.embed = embedding.Embedding(
            use_embed,
            unique_x,
            unique_edge_attr,
            embedding_dim_x,
            embedding_dim_edge_attr,
            act_mode,
        )

        dim_x, dim_edge_attr = embedding_dim_x, embedding_dim_edge_attr
        self.convs = torch.nn.ModuleList()
        for config in mpnn_configs:
            gnn = mpnn.from_config(
                config,
                node_in_feats=dim_x,
                edge_in_feats=dim_edge_attr,
                dropout=dropout,
                do_edge_update=do_edge_update,
                act_mode=act_mode,
                aggr_mode=aggr_mode,
            )
            dim_x, dim_edge_attr = gnn.node_out_feats, gnn.edge_out_feats
            self.convs.append(gnn)

        self.readout = aggregate.BlendAggregator(
            do_two_stage,
            in_channels=dim_x,
            heads=heads,
            num_sabs=num_sabs,
            dropout=dropout,
        )

    # Cannot get gradient checkpointing to work b/c the SetTransformerAggregation
    # computes dropout internally and doesn't allow checkpointing. So for now
    # we will use dropout and no checkpointing.
    def forward(self, graph):
        x, edge_attr = self.embed(graph)

        for conv in self.convs:
            x, edge_attr = conv(graph, x, edge_attr)

        return self.readout(x, graph)

    def count_parameters(self):
        return {
            "total": utils.count_parameters(self),
            "embed": utils.count_parameters(self.embed),
            "convs": [utils.count_parameters(conv) for conv in self.convs],
            "readout": utils.count_parameters(self.readout),
        }
