import torch_geometric as tg


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
