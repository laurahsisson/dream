import torch
import torch_geometric as tg
import collections

# Config does not contain dropout/do_edge_update/node_in_feats/edge_in_feats
# because these are fixed at the encoder level (not at MPNN level)
Config = collections.namedtuple(
    'Config',
    ['node_out_feats', 'edge_hidden_feats', 'num_step_message_passing'])


# Shamelessly stolen from (and converted to PytorchGeometric)
# https://lifesci.dgl.ai/_modules/dgllife/model/gnn/mpnn.html
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
    """

    def __init__(self, node_in_feats, edge_in_feats, node_out_feats,
                 edge_hidden_feats, num_step_message_passing, dropout):
        super(MPNNGNN, self).__init__()

        # This should be changed to node wise dropout. But maybe not?
        # See https://arxiv.org/pdf/1411.4280
        self.project_node_feats = torch.nn.Sequential(
            torch.nn.Linear(node_in_feats, node_out_feats), torch.nn.ReLU(),
            torch.nn.Dropout(dropout))
        self.num_step_message_passing = num_step_message_passing

        edge_network = torch.nn.Sequential(
            torch.nn.Linear(edge_in_feats, edge_hidden_feats),
            torch.nn.ReLU(),  # Could add dropout after this.
            torch.nn.Linear(edge_hidden_feats,
                            node_out_feats * node_out_feats),
            torch.nn.Dropout(dropout)  # This one is after the largest by far.
        )
        self.gnn_layer = tg.nn.conv.NNConv(in_channels=node_out_feats,
                                           out_channels=node_out_feats,
                                           nn=edge_network,
                                           aggr='sum')

        self.edge_out_feats = edge_in_feats
        self.node_out_feats = node_out_feats

        # If we add a second layer, we could add dropout.
        self.gru = torch.nn.GRU(node_out_feats,
                                node_out_feats,
                                bidirectional=False)
        self.final_dropout = torch.nn.Dropout(dropout)

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
        edge_feats : float32 tensor of shape (V, edge_in_feats)
            Unchanged edge representations.
        """
        node_feats = self.project_node_feats(node_feats)  # (V, node_out_feats)
        hidden_feats = node_feats.unsqueeze(0)  # (1, V, node_out_feats)

        for _ in range(self.num_step_message_passing):
            node_feats = torch.nn.functional.relu(
                self.gnn_layer(node_feats, graph.edge_index, edge_attr))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0),
                                                hidden_feats)
            node_feats = self.final_dropout(node_feats.squeeze(0))

        return node_feats, edge_attr


class MPNNGNNEdgeUpdate(torch.nn.Module):
    # As above but with an Edge Update network
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats,
                 edge_hidden_feats, num_step_message_passing, dropout):
        super(MPNNGNNEdgeUpdate, self).__init__()
        self.base = MPNNGNN(node_in_feats, edge_in_feats, node_out_feats,
                            edge_hidden_feats, num_step_message_passing,
                            dropout)

        self.project_edge_feats = torch.nn.Sequential(
            torch.nn.Linear(edge_in_feats, edge_hidden_feats), torch.nn.ReLU(),
            torch.nn.Dropout(dropout))

        # Inspired by https://arxiv.org/pdf/1806.01261
        self.edge_update_network = torch.nn.Sequential(
            torch.nn.Linear(edge_hidden_feats + 2 * node_out_feats,
                            edge_hidden_feats), torch.nn.ReLU(),
            torch.nn.Linear(edge_hidden_feats, edge_hidden_feats),
            torch.nn.Dropout(dropout))

        edge_network = torch.nn.Sequential(
            torch.nn.Linear(edge_hidden_feats,
                            node_out_feats * node_out_feats),
            torch.nn.Dropout(dropout)  # This one is after the largest by far.
        )
        self.gnn_layer = tg.nn.conv.NNConv(in_channels=node_out_feats,
                                           out_channels=node_out_feats,
                                           nn=edge_network,
                                           aggr='sum')

        self.edge_out_feats = edge_hidden_feats
        self.node_out_feats = self.base.node_out_feats

    def forward(self, graph, node_feats, edge_attr):
        """Performs message passing .

        Returns
        -------
        node_feats : float32 tensor of shape (V, node_out_feats)
            Output node representations.
        edge_attr : float32 tensor of shape (V, edge_hidden_feats)
            Output edge representations.
        """
        node_feats = self.base.project_node_feats(
            node_feats)  # (V, node_out_feats)
        edge_attr = self.project_edge_feats(edge_attr)  # (V, node_out_feats)
        hidden_feats = node_feats.unsqueeze(0)  # (1, V, node_out_feats)

        for _ in range(self.base.num_step_message_passing):
            node_feats = torch.nn.functional.relu(
                self.gnn_layer(node_feats, graph.edge_index, edge_attr))
            node_feats, hidden_feats = self.base.gru(node_feats.unsqueeze(0),
                                                     hidden_feats)
            node_feats = self.base.final_dropout(node_feats.squeeze(0))

            # Update edge attributes using node features of both ends of the edges
            row, col = graph.edge_index
            edge_input = torch.cat(
                [node_feats[row], node_feats[col], edge_attr], dim=1)
            edge_attr = self.edge_update_network(edge_input)

        return node_feats, edge_attr


def from_config(config, node_in_feats, edge_in_feats, dropout, do_edge_update):
    if do_edge_update:
        return MPNNGNNEdgeUpdate(**config._asdict(),
                                 node_in_feats=node_in_feats,
                                 edge_in_feats=edge_in_feats,
                                 dropout=dropout)
    else:
        return MPNNGNN(**config._asdict(),
                       node_in_feats=node_in_feats,
                       edge_in_feats=edge_in_feats,
                       dropout=dropout)
