import math
import unittest

import torch
from torch.nn import functional as F

from torchdrug import data, layers


class GraphConvTest(unittest.TestCase):

    def setUp(self):
        self.num_node = 10
        self.num_relation = 3
        self.input_dim = 5
        self.output_dim = 8
        adjacency = torch.rand(self.num_node, self.num_node, self.num_relation)
        threshold = adjacency.flatten().kthvalue((self.num_node - 3) * self.num_node)[0]
        adjacency = adjacency * (adjacency > threshold)
        self.graph = data.Graph.from_dense(adjacency).cuda()
        self.input = torch.rand(self.num_node, self.input_dim).cuda()

    def attention(self, query, key, value, mask, activation, eps=1e-10):
        weight = F.linear(key, query).squeeze(-1)
        weight = activation(weight)
        infinite = torch.tensor(math.inf, device=value.device)
        weight = torch.where(mask > 0, weight, -infinite)
        attention = (weight - weight.max(dim=0, keepdim=True)[0]).exp()
        attention = attention * mask
        attention = attention / (attention.sum(dim=0, keepdim=True) + eps)
        return (attention.unsqueeze(-1) * value).sum(dim=0)

    def test_graph_conv(self):
        conv = layers.GraphConv(self.input_dim, self.output_dim).cuda()
        result = conv(self.graph, self.input)
        adjacency = self.graph.adjacency.to_dense().sum(dim=-1)
        adjacency = adjacency + torch.eye(self.num_node, device=adjacency.device)
        adjacency /= adjacency.sum(dim=0, keepdim=True).sqrt() * adjacency.sum(dim=1, keepdim=True).sqrt()
        x = adjacency.t() @ self.input
        truth = conv.activation(conv.linear(x))
        self.assertTrue(torch.allclose(result, truth, rtol=1e-2, atol=1e-3), "Incorrect graph convolution")

        num_head = 2
        conv = layers.GraphAttentionConv(self.input_dim, self.output_dim, num_head=num_head).cuda()
        result = conv(self.graph, self.input)
        adjacency = self.graph.adjacency.to_dense().sum(dim=-1)
        adjacency = adjacency + torch.eye(self.num_node, device=adjacency.device)
        hidden = conv.linear(self.input)
        outputs = []
        for h, query in zip(hidden.chunk(num_head, dim=-1), conv.query.chunk(num_head, dim=0)):
            value = h.unsqueeze(1).expand(-1, self.num_node, -1)
            key = torch.stack([h.unsqueeze(1).expand(-1, self.num_node, -1),
                               h.unsqueeze(0).expand(self.num_node, -1, -1)], dim=-1).flatten(-2)
            output = self.attention(query, key, value, adjacency, conv.leaky_relu, conv.eps)
            outputs.append(output)
        truth = torch.cat(outputs, dim=-1)
        truth = conv.activation(truth)
        self.assertTrue(torch.allclose(result, truth, rtol=1e-2, atol=1e-3), "Incorrect graph attention convolution")

        eps = 1
        conv = layers.GraphIsomorphismConv(self.input_dim, self.output_dim, eps=eps).cuda()
        result = conv(self.graph, self.input)
        adjacency = self.graph.adjacency.to_dense().sum(dim=-1)
        x = (1 + eps) * self.input + adjacency.t() @ self.input
        truth = conv.activation(conv.mlp(x))
        self.assertTrue(torch.allclose(result, truth, rtol=1e-2, atol=1e-2), "Incorrect graph isomorphism convolution")

        conv = layers.RelationalGraphConv(self.input_dim, self.output_dim, self.num_relation).cuda()
        result = conv(self.graph, self.input)
        adjacency = self.graph.adjacency.to_dense()
        adjacency /= adjacency.sum(dim=0, keepdim=True)
        x = torch.einsum("htr, hd -> trd", adjacency, self.input)
        x = conv.linear(x.flatten(1)) + conv.self_loop(self.input)
        truth = conv.activation(x)
        self.assertTrue(torch.allclose(result, truth, rtol=1e-2, atol=1e-3), "Incorrect relational graph convolution")

        conv = layers.ChebyshevConv(self.input_dim, self.output_dim, k=2).cuda()
        result = conv(self.graph, self.input)
        adjacency = self.graph.adjacency.to_dense().sum(dim=-1)
        adjacency /= adjacency.sum(dim=0, keepdim=True).sqrt() * adjacency.sum(dim=1, keepdim=True).sqrt()
        identity = torch.eye(self.num_node, device=adjacency.device)
        laplacian = identity - adjacency
        bases = [self.input, laplacian.t() @ self.input, (2 * laplacian.t() @ laplacian.t() - identity) @ self.input]
        x = conv.linear(torch.cat(bases, dim=-1))
        truth = conv.activation(x)
        self.assertTrue(torch.allclose(result, truth, rtol=1e-2, atol=1e-3), "Incorrect chebyshev graph convolution")


if __name__ == "__main__":
    unittest.main()