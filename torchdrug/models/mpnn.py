import torch
from torch import nn

from torchdrug import core, layers
from torchdrug.core import Registry as R


@R.register("models.MPNN")
class MessagePassingNeuralNetwork(nn.Module, core.Configurable):
    """
    Message Passing Neural Network proposed in `Neural Message Passing for Quantum Chemistry`_.

    This implements the enn-s2s variant in the original paper.

    .. _Neural Message Passing for Quantum Chemistry:
        https://arxiv.org/pdf/1704.01212.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dim (int): hidden dimension
        edge_input_dim (int): dimension of edge features
        num_layer (int, optional): number of hidden layers
        num_gru_layer (int, optional): number of GRU layers in each node update
        num_mlp_layer (int, optional): number of MLP layers in each message function
        num_s2s_step (int, optional): number of processing steps in set2set
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
    """

    def __init__(self, input_dim, hidden_dim, edge_input_dim, num_layer=1, num_gru_layer=1, num_mlp_layer=2,
                 num_s2s_step=3, short_cut=False, batch_norm=False, activation="relu", concat_hidden=False):
        super(MessagePassingNeuralNetwork, self).__init__()

        self.input_dim = input_dim
        self.edge_input_dim = edge_input_dim
        if concat_hidden:
            feature_dim = hidden_dim * num_layer
        else:
            feature_dim = hidden_dim
        self.output_dim = feature_dim * 2
        self.node_output_dim = feature_dim
        self.num_layer = num_layer
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.layer = layers.MessagePassing(hidden_dim, edge_input_dim, [hidden_dim] * (num_mlp_layer - 1),
                                           batch_norm, activation)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_gru_layer)

        self.readout = layers.Set2Set(feature_dim, num_s2s_step)

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).

        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        hiddens = []
        layer_input = self.linear(input)
        hx = layer_input.repeat(self.gru.num_layers, 1, 1)

        for i in range(self.num_layer):
            x = self.layer(graph, layer_input)
            hidden, hx = self.gru(x.unsqueeze(0), hx)
            hidden = hidden.squeeze(0)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }
