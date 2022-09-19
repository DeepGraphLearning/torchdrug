from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("models.ProteinLSTM")
class ProteinLSTM(nn.Module, core.Configurable):
    """
    Protein LSTM proposed in `Evaluating Protein Transfer Learning with TAPE`_.

    .. _Evaluating Protein Transfer Learning with TAPE:
        https://arxiv.org/pdf/1906.08230.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dim (int, optional): hidden dimension
        num_layers (int, optional): number of LSTM layers
        activation (str or function, optional): activation function
        layer_norm (bool, optional): apply layer normalization or not
        dropout (float, optional): dropout ratio of input features
    """

    def __init__(self, input_dim, hidden_dim, num_layers, activation='tanh', layer_norm=False, 
                dropout=0):
        super(ProteinLSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_dim    # output_dim for node feature is 2 * hidden_dim
        self.node_output_dim = 2 * hidden_dim
        self.num_layers = num_layers
        self.padding_id = input_dim - 1

        self.embedding = nn.Linear(input_dim, hidden_dim)
        if layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout,
                            bidirectional=True)

        self.reweight = nn.Linear(2 * num_layers, 1)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the residue representations and the graph representation(s).

        Parameters:
            graph (Protein): :math:`n` protein(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``residue_feature`` and ``graph_feature`` fields:
                residue representations of shape :math:`(|V_{res}|, d)`, graph representations of shape :math:`(n, d)`
        """
        input = graph.residue_feature.float()
        input = functional.variadic_to_padded(input, graph.num_residues, value=self.padding_id)[0]
        
        input = self.embedding(input)
        if self.layer_norm:
            input = self.layer_norm(input)
        if self.dropout:
            input = self.dropout(input)

        output, hidden = self.lstm(input)

        residue_feature = functional.padded_to_variadic(output, graph.num_residues)

        # (2 * num_layer, B, d)
        graph_feature = self.reweight(hidden[0].permute(1, 2, 0)).squeeze(-1)
        graph_feature = self.linear(graph_feature)
        graph_feature = self.activation(graph_feature)

        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature
        }
