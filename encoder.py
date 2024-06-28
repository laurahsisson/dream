import torch
import torch_geometric as tg
import mpnn
import embedding
import utils


class Encoder(torch.nn.Module):

    def __init__(self, graph_tokenizer, mpnn_configs, embedding_dim_x,
                 embedding_dim_edge_attr, do_edge_update, num_sabs, heads,
                 dropout, **kwargs):
        super(Encoder, self).__init__()

        use_embed = not graph_tokenizer is None
        if use_embed:
            unique_x, unique_edge_attr = graph_tokenizer.unique_x(
            ), graph_tokenizer.unique_edge_attr()
        else:
            unique_x, unique_edge_attr = 9, 3

        self.embed = embedding.Embedding(use_embed, unique_x, unique_edge_attr,
                                         embedding_dim_x,
                                         embedding_dim_edge_attr)

        dim_x, dim_edge_attr = embedding_dim_x, embedding_dim_edge_attr
        self.convs = torch.nn.ModuleList()
        for config in mpnn_configs:
            gnn = mpnn.from_config(config,
                                   node_in_feats=dim_x,
                                   edge_in_feats=dim_edge_attr,
                                   dropout=dropout,
                                   do_edge_update=do_edge_update)
            dim_x, dim_edge_attr = gnn.node_out_feats, gnn.edge_out_feats
            self.convs.append(gnn)

        self.readout = tg.nn.aggr.set_transformer.SetTransformerAggregation(
            dim_x,
            heads=heads,
            num_encoder_blocks=num_sabs,
            num_decoder_blocks=num_sabs,
            dropout=dropout)

    # Cannot get gradient checkpointing to work b/c the SetTransformerAggregation
    # computes dropout internally and doesn't allow checkpointing.
    # We can't combine checkpointing and gradient checkpointing without causing issues.
    # So we could commit to no dropout.
    def forward(self, graph):
        x, edge_attr = self.embed(graph)

        for conv in self.convs:
            x, edge_attr = conv(graph, x, edge_attr)

        if isinstance(graph, tg.data.Batch):
            return self.readout(x, graph.batch)

        return self.readout(x)

    def count_parameters(self):
        return {
            "total": utils.count_parameters(self),
            "convs": [utils.count_parameters(conv) for conv in self.convs],
            "readout": utils.count_parameters(self.readout)
        }
