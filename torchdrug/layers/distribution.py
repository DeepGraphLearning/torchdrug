import math
from collections.abc import Sequence

import torch
from torch import nn


class IndependentGaussian(nn.Module):
    """
    Independent Gaussian distribution.

    Parameters:
        mu (Tensor): mean of shape :math:`(N,)`
        sigma2 (Tensor): variance of shape :math:`(N,)`
        learnable (bool, optional): learnable parameters or not
    """

    def __init__(self, mu, sigma2, learnable=False):
        super(IndependentGaussian, self).__init__()
        if learnable:
            self.mu = nn.Parameter(torch.as_tensor(mu))
            self.sigma2 = nn.Parameter(torch.as_tensor(sigma2))
        else:
            self.register_buffer("mu", torch.as_tensor(mu))
            self.register_buffer("sigma2", torch.as_tensor(sigma2))
        self.dim = len(mu)

    def forward(self, input):
        """
        Compute the likelihood of input data.

        Parameters:
            input (Tensor): input data of shape :math:`(..., N)`
        """
        log_likelihood = -0.5 * (math.log(2 * math.pi) + self.sigma2.log() + (input - self.mu) ** 2 / self.sigma2)
        return log_likelihood

    def sample(self, *size):
        """
        Draw samples from the distribution.

        Parameters:
            size (tuple of int): shape of the samples
        """
        if len(size) == 1 and isinstance(size[0], Sequence):
            size = size[0]
        size = list(size) + [self.dim]

        sample = torch.randn(size, device=self.mu.device) * self.sigma2.sqrt() + self.mu
        return sample
