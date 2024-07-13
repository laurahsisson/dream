import torch
import torch_geometric as tg


class TwoStageAggregator(torch.nn.Module):

    def __init__(self, in_channels, heads, num_sabs, dropout):
        super(TwoStageAggregator, self).__init__()
        self.mol_readout = tg.nn.aggr.set_transformer.SetTransformerAggregation(
            in_channels,
            heads=heads,
            num_encoder_blocks=num_sabs,
            num_decoder_blocks=num_sabs,
            dropout=dropout)

        self.blend_readout = tg.nn.aggr.set_transformer.SetTransformerAggregation(
            in_channels,
            heads=heads,
            num_encoder_blocks=num_sabs,
            num_decoder_blocks=num_sabs,
            dropout=dropout)

    def forward(self, x, graph):
        # Aggregation over atoms in a molecule
        x = self.mol_readout(x, index=graph.mol_batch)
        # Aggregatation over molecules in a blend
        return self.blend_readout(x, index=graph.blend_batch)


class BlendAggregator(torch.nn.Module):

    def __init__(self, do_two_stage, in_channels, heads, num_sabs, dropout):
        super(BlendAggregator, self).__init__()
        self.do_two_stage = do_two_stage
        self.in_channels = in_channels
        if do_two_stage:
            self.readout = TwoStageAggregator(in_channels, heads, num_sabs,
                                              dropout)
        else:
            self.readout = tg.nn.aggr.set_transformer.SetTransformerAggregation(
                in_channels,
                heads=heads,
                num_encoder_blocks=num_sabs,
                num_decoder_blocks=num_sabs,
                dropout=dropout)

    def forward(self, x, graph):
        if self.do_two_stage:
            return self.readout(x, graph)

        if isinstance(graph, tg.data.Batch):
            return self.readout(x, graph.batch)

        return self.readout(x)
