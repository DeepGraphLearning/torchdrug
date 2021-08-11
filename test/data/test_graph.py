import unittest

import torch

from torchdrug import data


class GraphTest(unittest.TestCase):

    def setUp(self):
        self.num_node = 10
        self.num_feature = 3
        adjacency = torch.rand(self.num_node, self.num_node)
        threshold = adjacency.flatten().kthvalue((self.num_node - 3) * self.num_node)[0]
        self.adjacency = adjacency * (adjacency > threshold)
        self.edge_list = self.adjacency.nonzero()
        self.edge_weight = self.adjacency[self.adjacency > 0]
        self.node_feature = torch.rand(self.num_node, self.num_feature)
        self.edge_feature = torch.rand(len(self.edge_list), self.num_feature)
        self.graph_feature = torch.rand(self.num_feature)

    def block_diag(self, tensors):
        total_row = 0
        total_col = 0
        for tensor in tensors:
            num_row, num_col = tensor.shape
            total_row += num_row
            total_col += num_col
        result = torch.zeros(total_row, total_col)
        x = 0
        y = 0
        for tensor in tensors:
            num_row, num_col = tensor.shape
            result[x: x + num_row, y: y + num_col] = tensor
            x += num_row
            y += num_col
        return result

    def test_type_cast(self):
        dense_edge_feature = torch.zeros(self.num_node, self.num_node, self.num_feature)
        dense_edge_feature[tuple(self.edge_list.t())] = self.edge_feature
        graph = data.Graph.from_dense(self.adjacency, self.node_feature, dense_edge_feature)
        graph1 = data.Graph(self.edge_list.tolist(), self.edge_weight.tolist(), self.num_node,
                            node_feature=self.node_feature.tolist(), edge_feature=self.edge_feature.tolist())
        graph2 = data.Graph(self.edge_list.numpy(), self.edge_weight.numpy(), self.num_node,
                            node_feature=self.node_feature.numpy(), edge_feature=self.edge_feature.numpy())
        self.assertTrue(torch.equal(graph.edge_list, graph1.edge_list), "Incorrect type cast")
        self.assertTrue(torch.equal(graph.edge_feature, graph1.edge_feature), "Incorrect type cast")
        self.assertTrue(torch.equal(graph1.edge_list, graph2.edge_list), "Incorrect type cast")
        self.assertTrue(torch.equal(graph1.edge_weight, graph2.edge_weight), "Incorrect type cast")
        self.assertTrue(torch.equal(graph1.node_feature, graph2.node_feature), "Incorrect type cast")
        self.assertTrue(torch.equal(graph1.edge_feature, graph2.edge_feature), "Incorrect type cast")

    def test_index(self):
        graph = data.Graph(self.edge_list, self.edge_weight, self.num_node,
                           node_feature=self.node_feature, edge_feature=self.edge_feature)

        index = tuple(torch.randint(self.num_node, (2,)).tolist())
        result = graph[index]
        truth = self.adjacency[index]
        self.assertTrue(torch.equal(result, truth), "Incorrect index in single item")

        h_index = torch.randperm(self.num_node)[:self.num_node // 2]
        t_index = torch.randperm(self.num_node)[:self.num_node // 2]
        new_graph = graph[h_index, t_index]
        adj_result = new_graph.adjacency.to_dense()
        feat_result = new_graph.node_feature
        not_h_index = list(set(range(self.num_node)) - set(h_index.tolist()))
        not_t_index = list(set(range(self.num_node)) - set(t_index.tolist()))
        adj_truth = self.adjacency.clone()
        adj_truth[not_h_index, :] = 0
        adj_truth[:, not_t_index] = 0
        feat_truth = self.node_feature
        self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect index in node mask")
        self.assertTrue(torch.equal(feat_result, feat_truth), "Incorrect feature in node mask")

        new_graph = graph[:, 1: -1]
        adj_result = new_graph.adjacency.to_dense()
        feat_result = new_graph.node_feature
        adj_truth = torch.zeros_like(self.adjacency)
        adj_truth[:, 1: -1] = self.adjacency[:, 1: -1]
        feat_truth = self.node_feature
        self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect index in slice")
        self.assertTrue(torch.equal(feat_result, feat_truth), "Incorrect feature in slice")

        index = torch.randperm(self.num_node)[:self.num_node // 2]
        new_graph = graph.subgraph(index)
        adj_result = new_graph.adjacency.to_dense()
        feat_result = new_graph.node_feature
        adj_truth = self.adjacency[index][:, index]
        feat_truth = self.node_feature[index]
        self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect index in subgraph")
        self.assertTrue(torch.equal(feat_result, feat_truth), "Incorrect feature in subgraph")

    def test_device(self):
        graph = data.Graph(self.edge_list, self.edge_weight, self.num_node,
                           node_feature=self.node_feature, edge_feature=self.edge_feature,
                           graph_feature=self.graph_feature)
        graph1 = graph.cuda()
        self.assertEqual(graph1.adjacency.device.type, "cuda", "Incorrect device")
        graph2 = graph1.cpu()
        self.assertEqual(graph2.adjacency.device.type, "cpu", "Incorrect device")
        self.assertTrue(torch.equal(graph.adjacency.to_dense(), graph2.adjacency.to_dense()),
                        "Incorrect feature when changing device")
        self.assertTrue(torch.equal(graph.node_feature, graph2.node_feature), "Incorrect feature when changing device")
        self.assertTrue(torch.equal(graph.edge_feature, graph2.edge_feature), "Incorrect feature when changing device")
        self.assertTrue(torch.equal(graph.graph_feature, graph2.graph_feature), "Incorrect feature when changing device")

    def test_pack(self):
        graph = data.Graph(self.edge_list, self.edge_weight, self.num_node,
                           node_feature=self.node_feature, edge_feature=self.edge_feature,
                           graph_feature=self.graph_feature)
        # special case: graphs with no edges
        graphs = [graph.edge_mask([]), graph.edge_mask([])]
        for start in range(4):
            index = torch.arange(start, self.num_node)
            graphs.append(graph.subgraph(index))
        for graph in graphs:
            with graph.graph():
                graph.graph_feature = torch.rand_like(self.graph_feature)

        packed_graph = data.Graph.pack(graphs)
        adj_result = packed_graph.adjacency.to_dense()
        adj_truth = self.block_diag([graph.adjacency.to_dense() for graph in graphs])
        node_feat_result = packed_graph.node_feature
        node_feat_truth = torch.cat([graph.node_feature for graph in graphs])
        edge_feat_result = packed_graph.edge_feature
        edge_feat_truth = torch.cat([graph.edge_feature for graph in graphs])
        graph_feat_result = packed_graph.graph_feature
        graph_feat_truth = torch.stack([graph.graph_feature for graph in graphs])
        self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect index in pack")
        self.assertTrue(torch.equal(node_feat_result, node_feat_truth), "Incorrect feature in pack")
        self.assertTrue(torch.equal(edge_feat_result, edge_feat_truth), "Incorrect feature in pack")
        self.assertTrue(torch.equal(graph_feat_result, graph_feat_truth), "Incorrect feature in pack")

        new_graphs = packed_graph.unpack()
        self.assertEqual(len(graphs), len(new_graphs), "Incorrect length in unpack")
        for graph, new_graph in zip(graphs, new_graphs):
            adj_truth = graph.adjacency.to_dense()
            adj_result = new_graph.adjacency.to_dense()
            node_feat_truth = graph.node_feature
            node_feat_result = new_graph.node_feature
            edge_feat_truth = graph.edge_feature
            edge_feat_result = new_graph.edge_feature
            graph_feat_truth = graph.graph_feature
            graph_feat_result = new_graph.graph_feature
            self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect index in unpack")
            self.assertTrue(torch.equal(node_feat_result, node_feat_truth), "Incorrect feature in unpack")
            self.assertTrue(torch.equal(edge_feat_result, edge_feat_truth), "Incorrect feature in unpack")
            self.assertTrue(torch.equal(graph_feat_result, graph_feat_truth), "Incorrect feature in unpack")

        graph = data.Graph(self.edge_list, self.edge_weight, self.num_node,
                           node_feature=self.node_feature, edge_feature=self.edge_feature)
        graphs = graphs[2:]
        packed_graph = data.Graph.pack(graphs)
        packed_graph2 = data.Graph.pack([graph] * len(graphs))
        mask = torch.zeros(self.num_node * len(graphs), dtype=torch.bool)
        for start in range(4):
            mask[start * self.num_node + start: (start + 1) * self.num_node] = 1
        packed_graph2 = packed_graph2.subgraph(mask)
        adj_result = packed_graph2.adjacency.to_dense()
        adj_truth = packed_graph.adjacency.to_dense()
        node_feat_result = packed_graph2.node_feature
        node_feat_truth = packed_graph.node_feature
        edge_feat_result = packed_graph2.edge_feature
        edge_feat_truth = packed_graph.edge_feature
        self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect index in subgraph")
        self.assertTrue(torch.equal(node_feat_result, node_feat_truth), "Incorrect feature in subgraph")
        self.assertTrue(torch.equal(edge_feat_result, edge_feat_truth), "Incorrect feature in subgraph")

        packed_graph = data.Graph.pack(graphs[::2])
        packed_graph2 = data.Graph.pack(graphs)[::2]
        adj_result = packed_graph2.adjacency.to_dense()
        adj_truth = packed_graph.adjacency.to_dense()
        node_feat_result = packed_graph2.node_feature
        node_feat_truth = packed_graph.node_feature
        edge_feat_result = packed_graph2.edge_feature
        edge_feat_truth = packed_graph.edge_feature
        self.assertEqual(len(packed_graph), len(packed_graph2), "Incorrect batch size in graph mask")
        self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect index in graph mask")
        self.assertTrue(torch.equal(node_feat_result, node_feat_truth), "Incorrect feature in graph mask")
        self.assertTrue(torch.equal(edge_feat_result, edge_feat_truth), "Incorrect feature in graph mask")

    def test_reorder(self):
        graph = data.Graph(self.edge_list, self.edge_weight, self.num_node,
                           node_feature=self.node_feature, edge_feature=self.edge_feature)
        order = torch.randperm(graph.num_node)
        new_graph = graph.subgraph(order)
        adj_result = new_graph.adjacency.to_dense()
        adj_truth = graph.adjacency.to_dense().index_select(0, order).index_select(1, order)
        node_feat_result = new_graph.node_feature
        node_feat_truth = graph.node_feature[order]
        edge_feat_result = new_graph.edge_feature
        edge_feat_truth = graph.edge_feature
        self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect node reorder")
        self.assertTrue(torch.equal(node_feat_result, node_feat_truth), "Incorrect node reorder")
        self.assertTrue(torch.equal(edge_feat_result, edge_feat_truth), "Incorrect node reorder")

        order = torch.randperm(graph.num_edge)
        new_graph = graph.edge_mask(order)
        edge_result = new_graph.edge_list
        edge_truth = graph.edge_list[order]
        node_feat_result = new_graph.node_feature
        node_feat_truth = graph.node_feature
        edge_feat_result = new_graph.edge_feature
        edge_feat_truth = graph.edge_feature[order]
        self.assertTrue(torch.equal(edge_result, edge_truth), "Incorrect edge reorder")
        self.assertTrue(torch.equal(node_feat_result, node_feat_truth), "Incorrect edge reorder")
        self.assertTrue(torch.equal(edge_feat_result, edge_feat_truth), "Incorrect edge reorder")

        graphs = []
        for start in range(4):
            index = torch.arange(start, self.num_node)
            graphs.append(graph.subgraph(index))
        packed_graph = data.Graph.pack(graphs)
        order = torch.randperm(4)
        packed_graph = packed_graph.subbatch(order)
        packed_graph2 = data.Graph.pack([graphs[i] for i in order])
        adj_result = packed_graph.adjacency.to_dense()
        adj_truth = packed_graph2.adjacency.to_dense()
        node_feat_result = packed_graph.node_feature
        node_feat_truth = packed_graph2.node_feature
        edge_feat_result = packed_graph.edge_feature
        edge_feat_truth = packed_graph2.edge_feature
        self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect graph reorder")
        self.assertTrue(torch.equal(node_feat_result, node_feat_truth), "Incorrect graph reorder")
        self.assertTrue(torch.equal(edge_feat_result, edge_feat_truth), "Incorrect graph reorder")

    def test_repeat(self):
        graph = data.Graph(self.edge_list, self.edge_weight, self.num_node,
                           node_feature=self.node_feature, edge_feature=self.edge_feature)
        repeat_graph = graph.repeat(5)
        true_graph = data.Graph.pack([graph] * 5)
        adj_result = repeat_graph.adjacency.to_dense()
        adj_truth = true_graph.adjacency.to_dense()
        node_feat_result = repeat_graph.node_feature
        node_feat_truth = true_graph.node_feature
        edge_feat_result = repeat_graph.edge_feature
        edge_feat_truth = true_graph.edge_feature
        self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect index in repeat")
        self.assertTrue(torch.equal(node_feat_result, node_feat_truth), "Incorrect feature in repeat")
        self.assertTrue(torch.equal(edge_feat_result, edge_feat_truth), "Incorrect feature in repeat")

        # special case: graphs with no edges
        graphs = [graph.edge_mask([]), graph.edge_mask([])]
        for start in range(4):
            index = torch.arange(start, self.num_node)
            graphs.append(graph.subgraph(index))
        packed_graph = data.Graph.pack(graphs)
        repeat_graph = packed_graph.repeat(5)
        true_graph = data.Graph.pack(graphs * 5)
        adj_result = repeat_graph.adjacency.to_dense()
        adj_truth = true_graph.adjacency.to_dense()
        node_feat_result = repeat_graph.node_feature
        node_feat_truth = true_graph.node_feature
        edge_feat_result = repeat_graph.edge_feature
        edge_feat_truth = true_graph.edge_feature
        self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect index in repeat")
        self.assertTrue(torch.equal(node_feat_result, node_feat_truth), "Incorrect feature in repeat")
        self.assertTrue(torch.equal(edge_feat_result, edge_feat_truth), "Incorrect feature in repeat")

    def test_repeat_interleave(self):
        graph = data.Graph(self.edge_list, self.edge_weight, self.num_node,
                           node_feature=self.node_feature, edge_feature=self.edge_feature)
        # special case: graphs with no edges
        graphs = [graph.edge_mask([]), graph.edge_mask([])]
        for start in range(4):
            index = torch.arange(start, self.num_node)
            graphs.append(graph.subgraph(index))
        packed_graph = data.Graph.pack(graphs)
        # special case: 0 repetition
        repeats = [2, 0, 0, 2, 3, 0]
        repeat_graph = packed_graph.repeat_interleave(repeats)
        true_graphs = []
        for i, graph in zip(repeats, graphs):
            true_graphs += [graph] * i
        true_graph = data.Graph.pack(true_graphs)
        adj_result = repeat_graph.adjacency.to_dense()
        adj_truth = true_graph.adjacency.to_dense()
        node_feat_result = repeat_graph.node_feature
        node_feat_truth = true_graph.node_feature
        edge_feat_result = repeat_graph.edge_feature
        edge_feat_truth = true_graph.edge_feature
        self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect index in repeat_interleave")
        self.assertTrue(torch.equal(node_feat_result, node_feat_truth), "Incorrect feature in repeat_interleave")
        self.assertTrue(torch.equal(edge_feat_result, edge_feat_truth), "Incorrect feature in repeat_interleave")

    def test_split(self):
        graph = data.Graph(self.edge_list, self.edge_weight, self.num_node)
        node2graph = torch.randint(3, (10,))
        graphs = []
        for i in range(3):
            subgraph = graph.subgraph(node2graph == i)
            if subgraph.num_node > 0:
                graphs.append(subgraph)
        new_graphs = graph.split(node2graph).unpack()
        self.assertEqual(len(graphs), len(new_graphs), "Incorrect length in split")
        for graph, new_graph in zip(graphs, new_graphs):
            adj_truth = graph.adjacency.to_dense()
            adj_result = new_graph.adjacency.to_dense()
            self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect split")

        graphs = []
        for i in range(5, 10):
            # ensure connected graph
            edge_list = torch.stack([torch.arange(i - 1), torch.arange(1, i)], dim=-1)
            graph = data.Graph(edge_list, num_node=i)
            graphs.append(graph)
        packed_graph = data.Graph.pack(graphs)
        packed_graph2, num_cc_result = packed_graph.connected_components()
        num_cc_truth = torch.ones_like(num_cc_result)
        self.assertTrue(torch.equal(num_cc_result, num_cc_truth), "Incorrect connected components")
        stat_truth = sorted((graph.num_node, graph.num_edge) for graph in packed_graph)
        stat_result = sorted((graph.num_node, graph.num_edge) for graph in packed_graph2)
        self.assertEqual(stat_result, stat_truth, "Incorrect connected components")

        # shuffle node order
        perm = torch.randperm(packed_graph.num_node)
        adjacency = packed_graph.adjacency.to_dense()
        adjacency = adjacency.index_select(0, perm).index_select(1, perm)
        packed_graph2, num_cc_result = data.Graph.from_dense(adjacency).connected_components()
        num_cc_truth = torch.tensor([len(graphs)])
        self.assertTrue(torch.equal(num_cc_result, num_cc_truth), "Incorrect connected components")
        stat_truth = sorted((graph.num_node, graph.num_edge) for graph in packed_graph)
        stat_result = sorted((graph.num_node, graph.num_edge) for graph in packed_graph2)
        self.assertEqual(stat_result, stat_truth, "Incorrect connected components")

    def test_merge(self):
        graph = data.Graph(self.edge_list, self.edge_weight, self.num_node)
        graph2graph = torch.randint(2, (6,))
        graph2graph[0] = 0
        graph2graph[-1] = 1
        graphs = []
        for start in range(6):
            index = torch.arange(start, self.num_node)
            graphs.append(graph.subgraph(index))
        packed_graph = data.Graph.pack(graphs)
        merged_graph = packed_graph.merge(graph2graph)
        truth_graphs = []
        for i in range(2):
            index = (graph2graph == i).nonzero().flatten().tolist()
            truth_graph = data.Graph.pack([graphs[j] for j in index])
            truth_graphs.append(truth_graph)
        self.assertEqual(len(merged_graph), len(truth_graphs), "Incorrect length in merge")
        for graph, truth in zip(merged_graph, truth_graphs):
            adj_result = graph.adjacency.to_dense()
            adj_truth = truth.adjacency.to_dense()
            self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect merge")

    def test_directed(self):
        digraph = data.Graph(self.edge_list, self.edge_weight, self.num_node)
        graph = digraph.undirected()
        adj_result = graph.adjacency.to_dense()
        adj_truth = (digraph.adjacency + digraph.adjacency.t()).to_dense()
        self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect undirected graph")
        digraph2 = graph.directed()
        adj_result = digraph2.adjacency.to_dense()
        adj_truth = adj_truth.triu()
        self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect directed graph")


if __name__ == "__main__":
    unittest.main()