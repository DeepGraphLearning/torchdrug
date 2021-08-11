import unittest

import torch
from torch.nn import functional as F

from torchdrug import data, layers


class GraphReadoutTest(unittest.TestCase):

    def setUp(self):
        self.num_node = 10
        self.num_graph = 5
        self.feature_dim = 5
        self.graphs = []
        for i in range(self.num_graph):
            adjacency = torch.rand(self.num_node, self.num_node)
            threshold = adjacency.flatten().kthvalue((self.num_node - 3) * self.num_node)[0]
            adjacency = adjacency * (adjacency > threshold)
            node_feature = torch.rand(self.num_node, self.feature_dim)
            graph = data.Graph.from_dense(adjacency, node_feature).cuda()
            self.graphs.append(graph)
        self.graph = data.Graph.pack(self.graphs)

    def test_readout(self):
        readout = layers.SumReadout().cuda()
        result = readout(self.graph, self.graph.node_feature)
        truth = [graph.node_feature.sum(0) for graph in self.graphs]
        truth = torch.stack(truth)
        self.assertTrue(torch.allclose(result, truth), "Incorrect sum readout")

        readout = layers.MeanReadout().cuda()
        result = readout(self.graph, self.graph.node_feature)
        truth = [graph.node_feature.mean(0) for graph in self.graphs]
        truth = torch.stack(truth)
        self.assertTrue(torch.allclose(result, truth), "Incorrect mean readout")

        readout = layers.MaxReadout().cuda()
        result = readout(self.graph, self.graph.node_feature)
        truth = [graph.node_feature.max(0)[0] for graph in self.graphs]
        truth = torch.stack(truth)
        self.assertTrue(torch.allclose(result, truth), "Incorrect max readout")

        softmax = layers.Softmax().cuda()
        result = softmax(self.graph, self.graph.node_feature)
        truth = [F.softmax(graph.node_feature, dim=0) for graph in self.graphs]
        truth = torch.cat(truth)
        self.assertTrue(torch.allclose(result, truth), "Incorrect softmax")


if __name__ == "__main__":
    unittest.main()