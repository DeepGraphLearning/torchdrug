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

    def assert_equal(self, graph1, graph2, prompt):
        self.assertTrue(torch.equal(graph1.adjacency.to_dense(), graph2.adjacency.to_dense()),
                        "Incorrect edge list in %s" % prompt)
        if hasattr(graph1, "node_feature") and hasattr(graph2, "node_feature"):
            self.assertTrue(torch.equal(graph1.node_feature, graph2.node_feature), "Incorrect feature in %s" % prompt)
        if hasattr(graph1, "edge_feature") and hasattr(graph2, "edge_feature"):
            self.assertTrue(torch.equal(graph1.edge_feature, graph2.edge_feature), "Incorrect feature in %s" % prompt)
        if hasattr(graph1, "graph_feature") and hasattr(graph2, "graph_feature"):
            self.assertTrue(torch.equal(graph1.graph_feature, graph2.graph_feature), "Incorrect feature in %s" % prompt)

    def test_type_cast(self):
        dense_edge_feature = torch.zeros(self.num_node, self.num_node, self.num_feature)
        dense_edge_feature[tuple(self.edge_list.t())] = self.edge_feature
        graph = data.Graph.from_dense(self.adjacency, self.node_feature, dense_edge_feature)
        graph1 = data.Graph(self.edge_list.tolist(), self.edge_weight.tolist(), self.num_node,
                            node_feature=self.node_feature.tolist(), edge_feature=self.edge_feature.tolist())
        graph2 = data.Graph(self.edge_list.numpy(), self.edge_weight.numpy(), self.num_node,
                            node_feature=self.node_feature.numpy(), edge_feature=self.edge_feature.numpy())
        self.assert_equal(graph, graph1, "type cast")
        self.assert_equal(graph, graph2, "type cast")

    def test_index(self):
        graph = data.Graph(self.edge_list, self.edge_weight, self.num_node,
                           node_feature=self.node_feature, edge_feature=self.edge_feature)

        index = tuple(torch.randint(self.num_node, (2,)).tolist())
        result = graph[index]
        truth = self.adjacency[index]
        self.assertTrue(torch.equal(result, truth), "Incorrect edge in single item")

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
        self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect edge list in node mask")
        self.assertTrue(torch.equal(feat_result, feat_truth), "Incorrect feature in node mask")

        new_graph = graph[:, 1: -1]
        adj_result = new_graph.adjacency.to_dense()
        feat_result = new_graph.node_feature
        adj_truth = torch.zeros_like(self.adjacency)
        adj_truth[:, 1: -1] = self.adjacency[:, 1: -1]
        feat_truth = self.node_feature
        self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect edge list in slice")
        self.assertTrue(torch.equal(feat_result, feat_truth), "Incorrect feature in slice")

        index = torch.randperm(self.num_node)[:self.num_node // 2]
        new_graph = graph.subgraph(index)
        adj_result = new_graph.adjacency.to_dense()
        feat_result = new_graph.node_feature
        adj_truth = self.adjacency[index][:, index]
        feat_truth = self.node_feature[index]
        self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect edge list in subgraph")
        self.assertTrue(torch.equal(feat_result, feat_truth), "Incorrect feature in subgraph")

    def test_device(self):
        graph = data.Graph(self.edge_list, self.edge_weight, self.num_node,
                           node_feature=self.node_feature, edge_feature=self.edge_feature,
                           graph_feature=self.graph_feature)
        graph1 = graph.cuda()
        self.assertEqual(graph1.adjacency.device.type, "cuda", "Incorrect device")
        graph2 = graph1.cpu()
        self.assertEqual(graph2.adjacency.device.type, "cpu", "Incorrect device")
        self.assert_equal(graph, graph2, "device")

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
        self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect edge list in pack")
        self.assertTrue(torch.equal(node_feat_result, node_feat_truth), "Incorrect feature in pack")
        self.assertTrue(torch.equal(edge_feat_result, edge_feat_truth), "Incorrect feature in pack")
        self.assertTrue(torch.equal(graph_feat_result, graph_feat_truth), "Incorrect feature in pack")

        new_graphs = packed_graph.unpack()
        self.assertEqual(len(graphs), len(new_graphs), "Incorrect length in unpack")
        for graph, new_graph in zip(graphs, new_graphs):
            self.assert_equal(graph, new_graph, "unpack")

        graph = data.Graph(self.edge_list, self.edge_weight, self.num_node,
                           node_feature=self.node_feature, edge_feature=self.edge_feature)
        graphs = graphs[2:]
        packed_graph = data.Graph.pack(graphs)
        packed_graph2 = data.Graph.pack([graph] * len(graphs))
        mask = torch.zeros(self.num_node * len(graphs), dtype=torch.bool)
        for start in range(4):
            mask[start * self.num_node + start: (start + 1) * self.num_node] = 1
        packed_graph2 = packed_graph2.subgraph(mask)
        self.assert_equal(packed_graph, packed_graph2, "subgraph")

        packed_graph = data.Graph.pack(graphs[::2])
        packed_graph2 = data.Graph.pack(graphs)[::2]
        self.assertEqual(len(packed_graph), len(packed_graph2), "Incorrect batch size in graph mask")
        self.assert_equal(packed_graph, packed_graph2, "graph mask")

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
        self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect edge list in node reorder")
        self.assertTrue(torch.equal(node_feat_result, node_feat_truth), "Incorrect feature in node reorder")
        self.assertTrue(torch.equal(edge_feat_result, edge_feat_truth), "Incorrect feature in node reorder")

        order = torch.randperm(graph.num_edge)
        new_graph = graph.edge_mask(order)
        edge_result = new_graph.edge_list
        edge_truth = graph.edge_list[order]
        node_feat_result = new_graph.node_feature
        node_feat_truth = graph.node_feature
        edge_feat_result = new_graph.edge_feature
        edge_feat_truth = graph.edge_feature[order]
        self.assertTrue(torch.equal(edge_result, edge_truth), "Incorrect edge list in edge reorder")
        self.assertTrue(torch.equal(node_feat_result, node_feat_truth), "Incorrect feature in edge reorder")
        self.assertTrue(torch.equal(edge_feat_result, edge_feat_truth), "Incorrect feature in edge reorder")

        graphs = []
        for start in range(4):
            index = torch.arange(start, self.num_node)
            graphs.append(graph.subgraph(index))
        packed_graph = data.Graph.pack(graphs)
        order = torch.randperm(4)
        packed_graph = packed_graph.subbatch(order)
        packed_graph2 = data.Graph.pack([graphs[i] for i in order])
        self.assert_equal(packed_graph, packed_graph2, "graph reorder")

    def test_repeat(self):
        graph = data.Graph(self.edge_list, self.edge_weight, self.num_node,
                           node_feature=self.node_feature, edge_feature=self.edge_feature)
        repeat_graph = graph.repeat(5)
        true_graph = data.Graph.pack([graph] * 5)
        self.assert_equal(repeat_graph, true_graph, "repeat")

        # special case: graphs with no edges
        graphs = [graph.edge_mask([]), graph.edge_mask([])]
        for start in range(4):
            index = torch.arange(start, self.num_node)
            graphs.append(graph.subgraph(index))
        packed_graph = data.Graph.pack(graphs)
        repeat_graph = packed_graph.repeat(5)
        true_graph = data.Graph.pack(graphs * 5)
        self.assert_equal(repeat_graph, true_graph, "repeat")

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
        self.assert_equal(repeat_graph, true_graph, "repeat interleave")

    def test_repeated_index(self):
        graph = data.Graph(self.edge_list, self.edge_weight, self.num_node)
        graphs = []
        for start in range(4):
            index = torch.arange(start, self.num_node)
            graphs.append(graph.subgraph(index))
        packed_graph = data.Graph.pack(graphs)
        # special case: some indexes missing, not sorted
        index = [1, 0, 2, 1, 0]
        packed_graph = packed_graph[index]
        packed_graph2 = data.Graph.pack([graphs[i] for i in index])
        self.assert_equal(packed_graph, packed_graph2, "repeated index")

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
        self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect conversion from directed to undirected")
        digraph2 = graph.directed()
        adj_result = digraph2.adjacency.to_dense()
        adj_truth = adj_truth.triu()
        self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect conversion from undirected to directed")

    def test_match(self):
        graph = data.Graph(self.edge_list, self.edge_weight, self.num_node)
        index = torch.randperm(graph.num_edge)[:self.num_node]
        edge = graph.edge_list[index]
        mask = torch.randint(2, (len(edge), 1))
        edge.scatter_(1, mask, -1)
        random = torch.randint_like(edge, self.num_node)
        edge = torch.cat([edge, random])
        index_result, num_match_result = graph.match(edge)
        index_results = index_result.split(num_match_result.tolist())
        match = ((graph.edge_list.unsqueeze(0) == edge.unsqueeze(1)) | (edge.unsqueeze(1) == -1)).all(dim=-1)
        query_index, index_truth = match.nonzero().t()
        num_match_truth = query_index.bincount(minlength=len(edge))
        index_truths = index_truth.split(num_match_truth.tolist())
        self.assertTrue(torch.equal(num_match_result, num_match_truth), "Incorrect edge match")
        for index_result, index_truth in zip(index_results, index_truths):
            self.assertTrue(torch.equal(index_result.sort()[0], index_truth.sort()[0]), "Incorrect edge match")

    def test_reference(self):
        node_out = torch.arange(1, self.num_node)
        node_in = torch.div(node_out - 1, 2, rounding_mode="floor")
        edge_list = torch.stack([node_in, node_out], dim=-1)
        tree = data.Graph(edge_list, num_node=self.num_node)
        with tree.node(), tree.node_reference():
            tree.dad = torch.div(torch.arange(self.num_node) - 1, 2, rounding_mode="floor")

        mask = torch.arange(1, self.num_node)
        graph = tree.subgraph(mask)
        degree_in_result = graph.dad[graph.dad != -1].bincount(minlength=graph.num_node)
        is_root_result = graph.dad == -1
        node_in, node_out = graph.edge_list.t()
        degree_in_truth = node_in.bincount(minlength=graph.num_node)
        is_root_truth = node_out.bincount(minlength=graph.num_node) == 0
        self.assertTrue(torch.equal(degree_in_result, degree_in_truth), "Incorrect node reference")
        self.assertTrue(torch.equal(is_root_result, is_root_truth), "Incorrect node reference")

        packed_graph = tree.repeat(4)
        packed_graph2 = data.Graph.pack([tree] * 4)
        self.assert_equal(packed_graph, packed_graph2, "node reference")

        # special case: 0 repetition
        repeats = [2, 0, 1, 2]
        trees = []
        for start in range(4):
            index = torch.arange(start, self.num_node)
            trees.append(tree.subgraph(index))
        packed_graph = data.Graph.pack(trees)
        repeat_graph = packed_graph.repeat_interleave(repeats)
        true_graphs = []
        for i, tree in zip(repeats, trees):
            true_graphs += [tree] * i
        true_graph = data.Graph.pack(true_graphs)
        self.assert_equal(repeat_graph, true_graph, "node reference")

    def test_line_graph(self):
        graph = data.Graph(self.edge_list, self.edge_weight, self.num_node, edge_feature=self.edge_feature)
        line_graph = graph.line_graph()
        adj_result = line_graph.adjacency.to_dense()
        feat_result = line_graph.node_feature
        edge_index = torch.arange(graph.num_edge)
        node_in, node_out = graph.edge_list.t()
        edge2node_out = torch.zeros(graph.num_edge, graph.num_node)
        node_in2edge = torch.zeros(graph.num_node, graph.num_edge)
        edge2node_out[edge_index, node_out] = 1
        node_in2edge[node_in, edge_index] = 1
        adj_truth = edge2node_out @ node_in2edge
        feat_truth = graph.edge_feature
        self.assertTrue(torch.equal(adj_result, adj_truth), "Incorrect line graph")
        self.assertTrue(torch.equal(feat_result, feat_truth), "Incorrect line graph")


if __name__ == "__main__":
    unittest.main()