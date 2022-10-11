from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core, layers
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("models.ProteinResNet")
class ProteinResNet(nn.Module, core.Configurable):
    """
    Protein ResNet proposed in `Evaluating Protein Transfer Learning with TAPE`_.

    .. _Evaluating Protein Transfer Learning with TAPE:
        https://arxiv.org/pdf/1906.08230.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        kernel_size (int, optional): size of convolutional kernel
        stride (int, optional): stride of convolution
        padding (int, optional): padding added to both sides of the input
        activation (str or function, optional): activation function
        short_cut (bool, optional): use short cut or not
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        layer_norm (bool, optional): apply layer normalization or not
        dropout (float, optional): dropout ratio of input features
        readout (str, optional): readout function. Available functions are ``sum``, ``mean`` and ``attention``.
    """

    def __init__(self, input_dim, hidden_dims, kernel_size=3, stride=1, padding=1,
                 activation="gelu", short_cut=False, concat_hidden=False, layer_norm=False,
                 dropout=0, readout="attention"):
        super(ProteinResNet, self).__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.padding_id = input_dim - 1

        self.embedding = nn.Linear(input_dim, hidden_dims[0])
        self.position_embedding = layers.SinusoidalPositionEmbedding(hidden_dims[0])
        if layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dims[0])
        else:
            self.layer_norm = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layers.ProteinResNetBlock(self.dims[i], self.dims[i + 1], kernel_size,
                                                         stride, padding, activation))

        if readout == "sum":
            self.readout = layers.SumReadout("residue")
        elif readout == "mean":
            self.readout = layers.MeanReadout("residue")
        elif readout == "attention":
            self.readout = layers.AttentionReadout(self.output_dim, "residue")
        else:
            raise ValueError("Unknown readout `%s`" % readout)

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
        input, mask = functional.variadic_to_padded(input, graph.num_residues, value=self.padding_id)
        mask = mask.unsqueeze(-1)

        input = self.embedding(input) + self.position_embedding(input).unsqueeze(0)
        if self.layer_norm:
            input = self.layer_norm(input)
        if self.dropout:
            input = self.dropout(input)
        input = input * mask
        
        hiddens = []
        layer_input = input
        
        for layer in self.layers:
            hidden = layer(layer_input, mask)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            hidden = torch.cat(hiddens, dim=-1)
        else:
            hidden = hiddens[-1]

        residue_feature = functional.padded_to_variadic(hidden, graph.num_residues)
        graph_feature = self.readout(graph, residue_feature)
        
        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature
        }


@R.register("models.ProteinConvolutionalNetwork")
class ProteinConvolutionalNetwork(nn.Module, core.Configurable):
    """
    Protein Shallow CNN proposed in `Is Transfer Learning Necessary for Protein Landscape Prediction?`_.

    .. _Is Transfer Learning Necessary for Protein Landscape Prediction?:
        https://arxiv.org/pdf/2011.03443.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        kernel_size (int, optional): size of convolutional kernel
        stride (int, optional): stride of convolution
        padding (int, optional): padding added to both sides of the input
        activation (str or function, optional): activation function
        short_cut (bool, optional): use short cut or not
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum``, ``mean``, ``max`` and ``attention``.
    """

    def __init__(self, input_dim, hidden_dims, kernel_size=3, stride=1, padding=1,
                activation='relu', short_cut=False, concat_hidden=False, readout="max"):
        super(ProteinConvolutionalNetwork, self).__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.padding_id = input_dim - 1

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                nn.Conv1d(self.dims[i], self.dims[i+1], kernel_size, stride, padding)
            )

        if readout == "sum":
            self.readout = layers.SumReadout("residue")
        elif readout == "mean":
            self.readout = layers.MeanReadout("residue")
        elif readout == "max":
            self.readout = layers.MaxReadout("residue")
        elif readout == "attention":
            self.readout = layers.AttentionReadout(self.output_dim, "residue")
        else:
            raise ValueError("Unknown readout `%s`" % readout)

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
        
        hiddens = []
        layer_input = input
        
        for layer in self.layers:
            hidden = layer(layer_input.transpose(1, 2)).transpose(1, 2)
            hidden = self.activation(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            hidden = torch.cat(hiddens, dim=-1)
        else:
            hidden = hiddens[-1]

        residue_feature = functional.padded_to_variadic(hidden, graph.num_residues)
        graph_feature = self.readout(graph, residue_feature)
        
        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature
        }
