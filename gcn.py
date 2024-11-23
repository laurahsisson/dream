import torch
import torch_geometric as pyg
import utils
import activation
import aggregate


class GCN(torch.nn.Module):

    def __init__(self, in_features, num_convs, num_mpnns, num_hidden_layers,
                 hidden_dim, dropout, do_two_stage, num_heads, num_sabs,
                 notes_dim, act_mode, **kwargs):
        """
        Initializes the GCN model.

        Args:
            in_features (int): Number of input features for each node.
            num_convs (int): Number of convolutional layers per GINConv block.
            num_hidden_layers (int): Number of hidden layers in the MLP of GINConv.
            hidden_dim (int): Dimension of hidden layers.
            dropout (float): Dropout rate to apply.
            notes_dim (int): Output dimension for note predictions.
        """
        super(GCN, self).__init__()

        # Activation function
        self.act_fn = activation.get_act_fn(act_mode)
        self.num_convs = num_convs

        # Feature normalization
        self.feature_norm = torch.nn.BatchNorm1d(in_features)

        # Initial projection of node features to hidden dimension
        self.project_node_feats = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_dim), self.act_fn(),
            torch.nn.Dropout(dropout))

        # GNN layers with hidden state normalization
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_mpnns):
            # Build the MLP for GINConv
            nn = utils.build_layers(in_dim=hidden_dim,
                                    hidden_dim=hidden_dim,
                                    out_dim=hidden_dim,
                                    act_fn=self.act_fn,
                                    num_hidden_layers=num_hidden_layers)
            # Graph convolution layer (GINConv) with MLP
            self.convs.append(pyg.nn.GINConv(nn))
            # Add normalization layer for hidden states
            self.norms.append(torch.nn.BatchNorm1d(hidden_dim))

        self.dropout = torch.nn.Dropout(dropout)

        # Readout layer
        self.readout = aggregate.BlendAggregator(do_two_stage,
                                                 in_channels=hidden_dim,
                                                 heads=num_heads,
                                                 num_sabs=num_sabs,
                                                 dropout=dropout)

        # Linear layer for note prediction
        self.notes_predictor = torch.nn.Linear(hidden_dim, notes_dim)

    def do_conv(self, conv_idx, x, residual, graph):
        conv, norm = self.convs[conv_idx], self.norms[conv_idx]
        for _ in range(self.num_convs):
            x = conv(x, graph.edge_index)  # GINConv
            x = norm(x)  # Hidden state normalization
            x = self.act_fn()(x)  # Activation
            x = self.dropout(x)  # Dropout after activation
        x = x + residual  # Add residual connection
        residual = x  # Update residual for next block

        return x, residual

    def forward(self, graph):
        """
        Defines the forward pass of the GCN.

        Args:
            graph: A PyG data object with attributes `x` (node features) and `edge_index` (edge connectivity).

        Returns:
            dict: Contains 'embed' (node embeddings) and 'logits' (note predictions).
        """

        # Feature normalization
        x = self.feature_norm(graph.x)

        # Initial projection of node features
        x = self.project_node_feats(x)

        # Residual connections and graph convolutions
        residual = x
        for conv_idx in range(len(self.convs)):
            x, residual = self.do_conv(conv_idx, x, residual, graph)

        # Save the embedding
        x = self.readout(x, graph)
        embedding = x.clone(
        )  # Clone to ensure isolation from further modifications

        # Predict notes using the final embedding
        predictions = self.notes_predictor(x)

        # Return both the embeddings and predictions
        return {"embed": embedding, "logits": predictions}
