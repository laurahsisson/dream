import torch
import torch_geometric as tg
import collections
import activation

# Config does not contain dropout/do_edge_update/node_in_feats/edge_in_feats/aggr_mode
# because these are fixed at the encoder level (not at MPNN level)
Config = collections.namedtuple(
    'Config',
    ['node_out_feats', 'edge_hidden_feats', 'num_step_message_passing'])


class MPNNGNN(torch.nn.Module):
    """MPNN

    MPNN is introduced in `Neural Message Passing for Quantum Chemistry
    <https://arxiv.org/abs/1704.01212>`

    This class performs message passing in MPNN and returns the updated node representations.

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    node_out_feats : int
        Size for the output node representations.
    edge_in_feats : int
        Size for the input edge features.
    edge_hidden_feats : int
        Size for the hidden edge representations.
    num_step_message_passing : int
        Number of message passing steps.
    do_edge_update : bool
        Whether to perform edge updates.
    aggr : str
        Aggregation scheme for NNConv.
    """

    def __init__(self, node_in_feats, edge_in_feats, node_out_feats,
                 edge_hidden_feats, num_step_message_passing, dropout, aggr_mode, act_mode, do_edge_update=False):
        super(MPNNGNN, self).__init__()

        act_fn = activation.get_act_fn(act_mode)
        self.do_edge_update = do_edge_update

        # Node feature projection
        self.project_node_feats = torch.nn.Sequential(
            torch.nn.Linear(node_in_feats, node_out_feats), act_fn(),
            torch.nn.Dropout(dropout))
        self.num_step_message_passing = num_step_message_passing

        if self.do_edge_update:
            # Edge feature projection and update network
            self.project_edge_feats = torch.nn.Sequential(
                torch.nn.Linear(edge_in_feats, edge_hidden_feats), act_fn(),
                torch.nn.Dropout(dropout))

            self.edge_update_network = torch.nn.Sequential(
                torch.nn.Linear(edge_hidden_feats + 2 * node_out_feats, edge_hidden_feats),
                act_fn(),
                torch.nn.Linear(edge_hidden_feats, edge_hidden_feats),
                act_fn(),
                torch.nn.Dropout(dropout))

            edge_network = torch.nn.Sequential(
                torch.nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats),
                act_fn(),
                torch.nn.Dropout(dropout)
            )
            self.edge_out_feats = edge_hidden_feats
        else:
            edge_network = torch.nn.Sequential(
                torch.nn.Linear(edge_in_feats, edge_hidden_feats),
                act_fn(),
                torch.nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats),
                act_fn(),
                torch.nn.Dropout(dropout)
            )
            self.edge_out_feats = edge_in_feats

        self.gnn_layer = tg.nn.conv.NNConv(in_channels=node_out_feats,
                                           out_channels=node_out_feats,
                                           nn=edge_network,
                                           aggr=aggr_mode)

        self.node_out_feats = node_out_feats

        self.gru = torch.nn.GRU(node_out_feats,
                                node_out_feats,
                                bidirectional=False)
        self.final_dropout = torch.nn.Dropout(dropout)
        self.act_fn = act_fn()

    def forward(self, graph, node_feats, edge_attr):
        """Performs message passing.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        edge_attr : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges in the batch of graphs.

        Returns
        -------
        node_feats : float32 tensor of shape (V, node_out_feats)
            Output node representations.
        edge_feats : float32 tensor of shape (V, edge_hidden_feats or edge_in_feats)
            Updated or unchanged edge representations.
        """
        
        node_feats = self.project_node_feats(node_feats)  # (V, node_out_feats)
        
        hidden_feats = node_feats.unsqueeze(0)  # (1, V, node_out_feats)

        if self.do_edge_update:
            edge_attr = self.project_edge_feats(edge_attr)  # (V, edge_hidden_feats)

        for _ in range(self.num_step_message_passing):
            node_feats = self.act_fn(
                self.gnn_layer(node_feats, graph.edge_index, edge_attr))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0),
                                                hidden_feats)
            node_feats = self.final_dropout(node_feats.squeeze(0))

            if self.do_edge_update:
                # Update edge attributes using node features of both ends of the edges
                row, col = graph.edge_index
                edge_input = torch.cat(
                    [node_feats[row], node_feats[col], edge_attr], dim=1)
                edge_attr = self.edge_update_network(edge_input)

        return node_feats, edge_attr

def from_config(config, node_in_feats, edge_in_feats, dropout, do_edge_update, act_mode, aggr_mode):
    return MPNNGNN(**config._asdict(),
                   node_in_feats=node_in_feats,
                   edge_in_feats=edge_in_feats,
                   dropout=dropout,
                   do_edge_update=do_edge_update,
                   act_mode=act_mode,
                   aggr_mode=aggr_mode)
