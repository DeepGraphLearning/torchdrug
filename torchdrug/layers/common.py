import inspect
import warnings
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_mean

from torchdrug.layers import functional


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.

    Note there is no batch normalization, activation or dropout in the last layer.

    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, hidden_dims, short_cut=False, batch_norm=False, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + hidden_dims
        self.short_cut = short_cut

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

    def forward(self, input):
        """"""
        layer_input = input

        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        return hidden


class GaussianSmearing(nn.Module):
    r"""
    Gaussian smearing from
    `SchNet: A continuous-filter convolutional neural network for modeling quantum interactions`_.

    There are two modes for Gaussian smearing.

    Non-centered mode:

    .. math::

        \mu = [0, 1, ..., n], \sigma = [1, 1, ..., 1]

    Centered mode:

    .. math::

        \mu = [0, 0, ..., 0], \sigma = [0, 1, ..., n]

    .. _SchNet\: A continuous-filter convolutional neural network for modeling quantum interactions:
        https://arxiv.org/pdf/1706.08566.pdf

    Parameters:
        start (int, optional): minimal input value
        stop (int, optional): maximal input value
        num_kernel (int, optional): number of RBF kernels
        centered (bool, optional): centered mode or not
        learnable (bool, optional): learnable gaussian parameters or not
    """

    def __init__(self, start=0, stop=5, num_kernel=100, centered=False, learnable=False):
        super(GaussianSmearing, self).__init__()
        if centered:
            mu = torch.zeros(num_kernel)
            sigma = torch.linspace(start, stop, num_kernel)
        else:
            mu = torch.linspace(start, stop, num_kernel)
            sigma = torch.ones(num_kernel) * (mu[1] - mu[0])

        if learnable:
            self.mu = nn.Parameter(mu)
            self.sigma = nn.Parameter(sigma)
        else:
            self.register_buffer("mu", mu)
            self.register_buffer("sigma", sigma)

    def forward(self, x, y):
        """
        Compute smeared gaussian features between data.

        Parameters:
            x (Tensor): data of shape :math:`(..., d)`
            y (Tensor): data of shape :math:`(..., d)`
        Returns:
            Tensor: features of shape :math:`(..., num\_kernel)`
        """
        distance = (x - y).norm(2, dim=-1, keepdim=True)
        z = (distance - self.mu) / self.sigma
        prob = torch.exp(-0.5 * z * z)
        return prob


class PairNorm(nn.Module):
    """
    Pair normalization layer proposed in `PairNorm: Tackling Oversmoothing in GNNs`_.

    .. _PairNorm\: Tackling Oversmoothing in GNNs:
        https://openreview.net/pdf?id=rkecl1rtwB

    Parameters:
        scale_individual (bool, optional): additionally normalize each node representation to have the same L2-norm
    """

    eps = 1e-8

    def __init__(self, scale_individual=False):
        super(PairNorm, self).__init__()
        self.scale_individual = scale_individual

    def forward(self, graph, input):
        """"""
        if graph.batch_size > 1:
            warnings.warn("PairNorm is proposed for a single graph, but now applied to a batch of graphs.")

        x = input.flatten(1)
        x = x - x.mean(dim=0)
        if self.scale_individual:
            output = x / (x.norm(dim=-1, keepdim=True) + self.eps)
        else:
            output = x  * x.shape[0] ** 0.5 / (x.norm() + self.eps)
        return output.view_as(input)


class InstanceNorm(nn.modules.instancenorm._InstanceNorm):
    """
    Instance normalization for graphs. This layer follows the definition in
    `GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training`.

    .. _GraphNorm\: A Principled Approach to Accelerating Graph Neural Network Training:
        https://arxiv.org/pdf/2009.03294.pdf

    Parameters:
        input_dim (int): input dimension
        eps (float, optional): epsilon added to the denominator
        affine (bool, optional): use learnable affine parameters or not
    """
    def __init__(self, input_dim, eps=1e-5, affine=False):
        super(InstanceNorm, self).__init__(input_dim, eps, affine=affine)

    def forward(self, graph, input):
        """"""
        assert (graph.num_nodes >= 1).all()

        mean = scatter_mean(input, graph.node2graph, dim=0, dim_size=graph.batch_size)
        centered = input - mean[graph.node2graph]
        var = scatter_mean(centered ** 2, graph.node2graph, dim=0, dim_size=graph.batch_size)
        std = (var + self.eps).sqrt()
        output = centered / std[graph.node2graph]

        if self.affine:
            output = torch.addcmul(self.bias, self.weight, output)
        return output


class MutualInformation(nn.Module):
    """
    Mutual information estimator from
    `Learning deep representations by mutual information estimation and maximization`_.

    .. _Learning deep representations by mutual information estimation and maximization:
        https://arxiv.org/pdf/1808.06670.pdf

    Parameters:
        input_dim (int): input dimension
        num_mlp_layer (int, optional): number of MLP layers
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, num_mlp_layer=2, activation="relu"):
        super(MutualInformation, self).__init__()
        self.x_mlp = MultiLayerPerceptron(input_dim, [input_dim] * num_mlp_layer, activation=activation)
        self.y_mlp = MultiLayerPerceptron(input_dim, [input_dim] * num_mlp_layer, activation=activation)

    def forward(self, x, y, pair_index=None):
        """"""
        x = self.x_mlp(x)
        y = self.y_mlp(y)
        score = x @ y.t()
        score = score.flatten()

        if pair_index is None:
            assert len(x) == len(y)
            pair_index = torch.arange(len(x), device=x.device).unsqueeze(-1).expand(-1, 2)

        index = pair_index[:, 0] * len(y) + pair_index[:, 1]
        positive = torch.zeros_like(score, dtype=torch.bool)
        positive[index] = 1
        negative = ~positive

        mutual_info = - functional.shifted_softplus(-score[positive]).mean() \
                      - functional.shifted_softplus(score[negative]).mean()
        return mutual_info


class Sequential(nn.Sequential):
    """
    Improved sequential container.
    Modules will be called in the order they are passed to the constructor.

    Compared to the vanilla nn.Sequential, this layer additionally supports the following features.

    1. Multiple input / output arguments.

    >>> # layer1 signature: (...) -> (a, b)
    >>> # layer2 signature: (a, b) -> (...)
    >>> layer = layers.Sequential(layer1, layer2)

    2. Global arguments.

    >>> # layer1 signature: (graph, a) -> b
    >>> # layer2 signature: (graph, b) -> c
    >>> layer = layers.Sequential(layer1, layer2, global_args=("graph",))

    Note the global arguments don't need to be present in every layer.

    >>> # layer1 signature: (graph, a) -> b
    >>> # layer2 signature: b -> c
    >>> # layer3 signature: (graph, c) -> d
    >>> layer = layers.Sequential(layer1, layer2, global_args=("graph",))

    3. Dict outputs.

    >>> # layer1 signature: a -> {"b": b, "c": c}
    >>> # layer2 signature: b -> d
    >>> layer = layers.Sequential(layer1, layer2, allow_unused=True)

    When dict outputs are used with global arguments, the global arguments can be explicitly
    overwritten by any layer outputs.

    >>> # layer1 signature: (graph, a) -> {"graph": graph, "b": b}
    >>> # layer2 signature: (graph, b) -> c
    >>> # layer2 takes in the graph output by layer1
    >>> layer = layers.Sequential(layer1, layer2, global_args=("graph",))
    """

    def __init__(self, *args, global_args=None, allow_unused=False):
        super(Sequential, self).__init__(*args)
        if global_args is not None:
            self.global_args = set(global_args)
        else:
            self.global_args = {}
        self.allow_unused = allow_unused

    def forward(self, *args, **kwargs):
        """"""
        global_kwargs = {}
        for i, module in enumerate(self._modules.values()):
            sig = inspect.signature(module.forward)
            parameters = list(sig.parameters.values())
            param_names = [param.name for param in parameters]
            j = 0
            for name in param_names:
                if j == len(args):
                    break
                if name in kwargs:
                    continue
                if name in global_kwargs and name not in kwargs:
                    kwargs[name] = global_kwargs[name]
                    continue
                kwargs[name] = args[j]
                j += 1
            if self.allow_unused:
                param_names = set(param_names)
                # pop unused kwargs
                kwargs = {k: v for k, v in kwargs.items() if k in param_names}
            if j < len(args):
                raise TypeError("too many positional arguments")

            output = module(**kwargs)

            global_kwargs.update({k: v for k, v in kwargs.items() if k in self.global_args})
            args = []
            kwargs = {}
            if isinstance(output, dict):
                kwargs.update(output)
            elif isinstance(output, Sequence):
                args += list(output)
            else:
                args.append(output)

        return output