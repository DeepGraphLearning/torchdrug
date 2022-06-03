import unittest

import torch
from torch.nn import functional as F

from torchdrug import data, layers


class GraphPoolTest(unittest.TestCase):

    def setUp(self):
        self.num_node = 10
        self.num_graph = 5
        self.input_dim = 5
        self.output_dim = 8
        self.output_node = 6
        self.graphs = []
        for i in range(self.num_graph):
            adjacency = torch.rand(self.num_node, self.num_node)
            threshold = adjacency.flatten().kthvalue((self.num_node - 3) * self.num_node)[0]
            adjacency = adjacency * (adjacency > threshold)
            node_feature = torch.rand(self.num_node, self.input_dim)
            graph = data.Graph.from_dense(adjacency, node_feature).cuda()
            self.graphs.append(graph)
        self.graph = data.Graph.pack(self.graphs)
        self.feature_layer = layers.GraphConv(self.input_dim, self.output_dim).cuda()
        self.pool_layer = layers.GraphConv(self.input_dim, self.output_dim).cuda()

    def test_pool(self):
        for sparse in [False, True]:
            pool = layers.DiffPool(self.input_dim, self.output_node, self.feature_layer, self.pool_layer,
                                   zero_diagonal=True, sparse=sparse).cuda()
            pooled, result, assignment = pool(self.graph, self.graph.node_feature)
            result = result.view(self.num_graph, self.output_node, -1)
            result_adj = torch.stack([g.adjacency.to_dense() for g in pooled])
            feature = pool.feature_layer(self.graph, self.graph.node_feature).view(self.num_graph, self.num_node, -1)
            x = pool.linear(pool.pool_layer(self.graph, self.graph.node_feature))
            if not sparse:
                assignment = F.softmax(x, dim=-1)
            assignment = assignment.view(self.num_graph, self.num_node, -1)
            adjacency = torch.stack([g.adjacency.to_dense() for g in self.graph])
            truth = torch.einsum("bna, bnd -> bad", assignment, feature)
            truth_adj = torch.einsum("bna, bnm, bmc -> bac", assignment, adjacency, assignment)
            index = torch.arange(self.output_node, device=truth.device)
            truth_adj[:, index, index] = 0
            self.assertTrue(torch.allclose(result, truth, rtol=1e-3, atol=1e-4), "Incorrect diffpool node feature")
            self.assertTrue(torch.allclose(result_adj, truth_adj, rtol=1e-3, atol=1e-4), "Incorrect diffpool adjacency")

            graph = self.graph[0]
            rng_state = torch.get_rng_state()
            pooled, result, assignment = pool(graph, graph.node_feature)
            result_adj = pooled.adjacency.to_dense()
            torch.set_rng_state(rng_state)
            feature = pool.feature_layer(graph, graph.node_feature)
            x = pool.linear(pool.pool_layer(graph, graph.node_feature))
            if not sparse:
                assignment = F.softmax(x, dim=-1)
            adjacency = graph.adjacency.to_dense()
            truth = torch.einsum("na, nd -> ad", assignment, feature)
            truth_adj = torch.einsum("na, nm, mc -> ac", assignment, adjacency, assignment)
            index = torch.arange(self.output_node, device=truth.device)
            truth_adj[index, index] = 0
            self.assertTrue(torch.allclose(result, truth, rtol=1e-3, atol=1e-4), "Incorrect diffpool node feature")
            self.assertTrue(torch.allclose(result_adj, truth_adj, rtol=1e-3, atol=1e-4), "Incorrect diffpool adjacency")

        pool = layers.MinCutPool(self.input_dim, self.output_node, self.feature_layer, self.pool_layer).cuda()
        all_loss = torch.tensor(0, dtype=torch.float32, device="cuda")
        result_metric = {}
        pooled, result, assignment = pool(self.graph, self.graph.node_feature, all_loss, result_metric)
        result = result.view(self.num_graph, self.output_node, -1)
        result_adj = torch.stack([g.adjacency.to_dense() for g in pooled])
        feature = pool.feature_layer(self.graph, self.graph.node_feature).view(self.num_graph, self.num_node, -1)
        x = pool.linear(pool.pool_layer(self.graph, self.graph.node_feature))
        assignment = F.softmax(x, dim=-1)
        assignment = assignment.view(self.num_graph, self.num_node, -1)
        adjacency = torch.stack([g.adjacency.to_dense() for g in self.graph])
        truth = torch.einsum("bna, bnd -> bad", assignment, feature)
        adjacency = torch.einsum("bna, bnm, bmc -> bac", assignment, adjacency, assignment)
        truth_adj = adjacency.clone()
        index = torch.arange(self.output_node, device=truth.device)
        truth_adj[:, index, index] = 0
        num_intra = torch.einsum("baa -> b", adjacency)
        degree = self.graph.degree_in.view(self.num_graph, self.num_node)
        degree = torch.einsum("bna, bn, bnc -> bac", assignment, degree, assignment)
        num_all = torch.einsum("baa -> b", degree)
        cut_loss = (1 - num_intra / num_all).mean()
        x = torch.einsum("bna, bnc -> bac", assignment, assignment)
        x = x / x.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1)
        x = x - torch.eye(self.output_node, device=x.device) / (self.output_node ** 0.5)
        regularization = x.flatten(-2).norm(dim=-1).mean()
        truth_metric = {"normalized cut loss": cut_loss, "orthogonal regularization": regularization}
        self.assertTrue(torch.allclose(result, truth, rtol=1e-3, atol=1e-4), "Incorrect min cut pool feature")
        self.assertTrue(torch.allclose(result_adj, truth_adj, rtol=1e-3, atol=1e-4), "Incorrect min cut pool adjcency")
        for key in result_metric:
            self.assertTrue(torch.allclose(result_metric[key], truth_metric[key], rtol=1e-3, atol=1e-4),
                            "Incorrect min cut pool metric")


if __name__ == "__main__":
    unittest.main()