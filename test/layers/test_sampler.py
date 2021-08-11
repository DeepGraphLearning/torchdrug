import unittest

import torch

from torchdrug import data, layers


class GraphSamplerTest(unittest.TestCase):

    def setUp(self):
        self.num_node = 10
        self.input_dim = 5
        self.output_dim = 7
        adjacency = torch.rand(self.num_node, self.num_node)
        threshold = adjacency.flatten().kthvalue((self.num_node - 3) * self.num_node)[0]
        adjacency = adjacency * (adjacency > threshold)
        self.graph = data.Graph.from_dense(adjacency).cuda()
        self.input = torch.rand(self.num_node, self.input_dim).cuda()

    def test_sampler(self):
        conv = layers.GraphConv(self.input_dim, self.output_dim, activation=None).cuda()
        readout = layers.SumReadout().cuda()

        sampler = layers.NodeSampler(ratio=0.8).cuda()
        results = []
        for i in range(2000):
            graph = sampler(self.graph)
            node_feature = conv(graph, self.input)
            result = readout(graph, node_feature)
            results.append(result)
        result = torch.stack(results).mean(dim=0)
        node_feature = conv(self.graph, self.input)
        truth = readout(self.graph, node_feature)
        self.assertTrue(torch.allclose(result, truth, rtol=5e-2, atol=5e-2), "Found bias in node sampler")

        sampler = layers.EdgeSampler(ratio=0.8).cuda()
        results = []
        for i in range(2000):
            graph = sampler(self.graph)
            node_feature = conv(graph, self.input)
            result = readout(graph, node_feature)
            results.append(result)
        result = torch.stack(results).mean(dim=0)
        node_feature = conv(self.graph, self.input)
        truth = readout(self.graph, node_feature)
        self.assertTrue(torch.allclose(result, truth, rtol=5e-2, atol=5e-2), "Found bias in edge sampler")


if __name__ == "__main__":
    unittest.main()