from collections.abc import Sequence

import torch
from torch import nn

from torchdrug import core, layers
from torchdrug.core import Registry as R


@R.register("models.SchNet")
class SchNet(nn.Module, core.Configurable):
    """
    SchNet from `SchNet: A continuous-filter convolutional neural network for modeling quantum interactions`_.

    .. _SchNet\: A continuous-filter convolutional neural network for modeling quantum interactions:
        https://arxiv.org/pdf/1706.08566.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        edge_input_dim (int, optional): dimension of edge features
        cutoff (float, optional): maximal scale for RBF kernels
        num_gaussian (int, optional): number of RBF kernels
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
    """

    def __init__(self, input_dim, hidden_dims, edge_input_dim=None, cutoff=5, num_gaussian=100, short_cut=True,
                 batch_norm=False, activation="shifted_softplus", concat_hidden=False):
        super(SchNet, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layers.ContinuousFilterConv(self.dims[i], self.dims[i + 1], edge_input_dim, None, cutoff,
                                                           num_gaussian, batch_norm, activation))

        self.readout = layers.SumReadout()

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).

        Require the graph(s) to have node attribute ``node_position``.

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
        layer_input = input

        for layer in self.layers:
            hidden = layer(graph, layer_input)
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