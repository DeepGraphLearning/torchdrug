import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import layers


class ConditionalFlow(nn.Module):
    """
    Conditional flow transformation from `Masked Autoregressive Flow for Density Estimation`_.

    .. _Masked Autoregressive Flow for Density Estimation:
        https://arxiv.org/pdf/1705.07057.pdf

    Parameters:
        input_dim (int): input & output dimension
        condition_dim (int): condition dimension
        hidden_dims (list of int, optional): hidden dimensions
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, condition_dim, hidden_dims=None, activation="relu"):
        super(ConditionalFlow, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim

        if hidden_dims is None:
            hidden_dims = []
        self.mlp = layers.MLP(condition_dim, list(hidden_dims) + [input_dim * 2], activation)
        self.rescale = nn.Parameter(torch.zeros(1))

    def forward(self, input, condition):
        """
        Transform data into latent representations.

        Parameters:
            input (Tensor): input representations
            condition (Tensor): conditional representations

        Returns:
            (Tensor, Tensor): latent representations, log-likelihood of the transformation
        """
        scale, bias = self.mlp(condition).chunk(2, dim=-1)
        scale = (F.tanh(scale) * self.rescale)
        output = (input + bias) * scale.exp()
        log_det = scale
        return output, log_det

    def reverse(self, latent, condition):
        """
        Transform latent representations into data.

        Parameters:
            latent (Tensor): latent representations
            condition (Tensor): conditional representations

        Returns:
            (Tensor, Tensor): input representations, log-likelihood of the transformation
        """
        scale, bias = self.mlp(condition).chunk(2, dim=-1)
        scale = (F.tanh(scale) * self.rescale)
        output = latent / scale.exp() - bias
        log_det = scale
        return output, log_det