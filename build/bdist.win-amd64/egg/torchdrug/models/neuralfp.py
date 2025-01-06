from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core, layers
from torchdrug.core import Registry as R


@R.register("models.NeuralFP")
class NeuralFingerprint(nn.Module, core.Configurable):
    """
    Neural Fingerprints from `Convolutional Networks on Graphs for Learning Molecular Fingerprints`_.

    .. _Convolutional Networks on Graphs for Learning Molecular Fingerprints:
        https://arxiv.org/pdf/1509.09292.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): fingerprint dimension
        hidden_dims (list of int): hidden dimensions
        edge_input_dim (int, optional): dimension of edge features
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, input_dim, output_dim, hidden_dims, edge_input_dim=None, short_cut=False, batch_norm=False,
                 activation="relu", concat_hidden=False, readout="sum"):
        super(NeuralFingerprint, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = output_dim * (len(hidden_dims) if concat_hidden else 1)
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.layers = nn.ModuleList()
        self.linears = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layers.NeuralFingerprintConv(self.dims[i], self.dims[i + 1], edge_input_dim,
                                                            batch_norm, activation))
            self.linears.append(nn.Linear(self.dims[i + 1], output_dim))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

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
        outputs = []
        layer_input = input

        for layer, linear in zip(self.layers, self.linears):
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            output = F.softmax(linear(hidden), dim=-1)
            hiddens.append(hidden)
            outputs.append(output)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
            graph_feature = torch.cat(outputs, dim=-1)
        else:
            node_feature = hiddens[-1]
            graph_feature = torch.stack(outputs).sum(dim=0)

        graph_feature = self.readout(graph, graph_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }