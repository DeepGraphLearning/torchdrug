import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add, scatter_mean

from torchdrug import data


class DiffPool(nn.Module):
    """
    Differentiable pooling operator from `Hierarchical Graph Representation Learning with Differentiable Pooling`_

    .. _Hierarchical Graph Representation Learning with Differentiable Pooling:
        https://papers.nips.cc/paper/7729-hierarchical-graph-representation-learning-with-differentiable-pooling.pdf

    Parameter
        input_dim (int): input dimension
        output_node (int): number of nodes after pooling
        feature_layer (Module, optional): graph convolution layer for embedding
        pool_layer (Module, optional): graph convolution layer for pooling assignment
        loss_weight (float, optional): weight of entropy regularization
        zero_diagonal (bool, optional): remove self loops in the pooled graph or not
        sparse (bool, optional): use sparse assignment or not
    """

    tau = 1
    eps = 1e-10

    def __init__(self, input_dim, output_node, feature_layer=None, pool_layer=None, loss_weight=1, zero_diagonal=False,
                 sparse=False):
        super(DiffPool, self).__init__()
        self.input_dim = input_dim
        self.output_dim = feature_layer.output_dim
        self.output_node = output_node
        self.feature_layer = feature_layer
        self.pool_layer = pool_layer
        self.loss_weight = loss_weight
        self.zero_diagonal = zero_diagonal
        self.sparse = sparse

        if pool_layer is not None:
            self.linear = nn.Linear(pool_layer.output_dim, output_node)
        else:
            self.linear = nn.Linear(input_dim, output_node)

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the node cluster assignment and pool the nodes.

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            (PackedGraph, Tensor, Tensor):
                pooled graph, output node representations, node-to-cluster assignment
        """
        feature = input
        if self.feature_layer:
            feature = self.feature_layer(graph, feature)

        x = input
        if self.pool_layer:
            x = self.pool_layer(graph, x)
        x = self.linear(x)
        if self.sparse:
            assignment = F.gumbel_softmax(x, hard=True, tau=self.tau, dim=-1)
            new_graph, output = self.sparse_pool(graph, feature, assignment)
        else:
            assignment = F.softmax(x, dim=-1)
            new_graph, output = self.dense_pool(graph, feature, assignment)

        if all_loss is not None:
            prob = scatter_mean(assignment, graph.node2graph, dim=0, dim_size=graph.batch_size)
            entropy = -(prob * (prob + self.eps).log()).sum(dim=-1)
            entropy = entropy.mean()
            metric["assignment entropy"] = entropy
            if self.loss_weight > 0:
                all_loss -= entropy * self.loss_weight

        if self.zero_diagonal:
            edge_list = new_graph.edge_list[:, :2]
            is_diagonal = edge_list[:, 0] == edge_list[:, 1]
            new_graph = new_graph.edge_mask(~is_diagonal)

        return new_graph, output, assignment

    def dense_pool(self, graph, input, assignment):
        node_in, node_out = graph.edge_list.t()[:2]
        # S^T A S, O(|V|k^2 + |E|k)
        x = graph.edge_weight.unsqueeze(-1) * assignment[node_out]
        x = scatter_add(x, node_in, dim=0, dim_size=graph.num_node)
        x = torch.einsum("np, nq -> npq", assignment, x)
        adjacency = scatter_add(x, graph.node2graph, dim=0, dim_size=graph.batch_size)
        # S^T X
        x = torch.einsum("na, nd -> nad", assignment, input)
        output = scatter_add(x, graph.node2graph, dim=0, dim_size=graph.batch_size).flatten(0, 1)

        index = torch.arange(self.output_node, device=graph.device).expand(len(graph), self.output_node, -1)
        edge_list = torch.stack([index.transpose(-1, -2), index], dim=-1).flatten(0, -2)
        edge_weight = adjacency.flatten()
        if isinstance(graph, data.PackedGraph):
            num_nodes = torch.ones(len(graph), dtype=torch.long, device=input.device) * self.output_node
            num_edges = torch.ones(len(graph), dtype=torch.long, device=input.device) * self.output_node ** 2
            graph = data.PackedGraph(edge_list, edge_weight=edge_weight, num_nodes=num_nodes, num_edges=num_edges)
        else:
            graph = data.Graph(edge_list, edge_weight=edge_weight, num_node=self.output_node)
        return graph, output

    def sparse_pool(self, graph, input, assignment):
        assignment = assignment.argmax(dim=-1)
        edge_list = graph.edge_list[:, :2]
        edge_list = assignment[edge_list]
        pooled_node = graph.node2graph * self.output_node + assignment
        output = scatter_add(input, pooled_node, dim=0, dim_size=graph.batch_size * self.output_node)

        edge_weight = graph.edge_weight
        if isinstance(graph, data.PackedGraph):
            num_nodes = torch.ones(len(graph), dtype=torch.long, device=input.device) * self.output_node
            num_edges = graph.num_edges
            graph = data.PackedGraph(edge_list, edge_weight=edge_weight, num_nodes=num_nodes, num_edges=num_edges)
        else:
            graph = data.Graph(edge_list, edge_weight=edge_weight, num_node=self.output_node)
        return graph, output


class MinCutPool(DiffPool):
    """
    Min cut pooling operator from `Spectral Clustering with Graph Neural Networks for Graph Pooling`_

    .. _Spectral Clustering with Graph Neural Networks for Graph Pooling:
        http://proceedings.mlr.press/v119/bianchi20a/bianchi20a.pdf

    Parameters:
        input_dim (int): input dimension
        output_node (int): number of nodes after pooling
        feature_layer (Module, optional): graph convolution layer for embedding
        pool_layer (Module, optional): graph convolution layer for pooling assignment
        loss_weight (float, optional): weight of entropy regularization
        zero_diagonal (bool, optional): remove self loops in the pooled graph or not
        sparse (bool, optional): use sparse assignment or not
    """

    eps = 1e-10

    def __init__(self, input_dim, output_node, feature_layer=None, pool_layer=None, loss_weight=1, zero_diagonal=True,
                 sparse=False):
        super(MinCutPool, self).__init__(input_dim, output_node, feature_layer, pool_layer, loss_weight, zero_diagonal,
                                         sparse)

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the node cluster assignment and pool the nodes.

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            (PackedGraph, Tensor, Tensor):
                pooled graph, output node representations, node-to-cluster assignment
        """
        feature = input
        if self.feature_layer:
            feature = self.feature_layer(graph, feature)

        x = input
        if self.pool_layer:
            x = self.pool_layer(graph, x)
        x = self.linear(x)
        if self.sparse:
            assignment = F.gumbel_softmax(x, hard=True, tau=self.tau, dim=-1)
            new_graph, output = self.sparse_pool(graph, feature, assignment)
        else:
            assignment = F.softmax(x, dim=-1)
            new_graph, output = self.dense_pool(graph, feature, assignment)

        if all_loss is not None:
            edge_list = new_graph.edge_list
            is_diagonal = edge_list[:, 0] == edge_list[:, 1]
            num_intra = scatter_add(new_graph.edge_weight[is_diagonal], new_graph.edge2graph[is_diagonal],
                                    dim=0, dim_size=new_graph.batch_size)
            x = torch.einsum("na, n, nc -> nac", assignment, graph.degree_in, assignment)
            x = scatter_add(x, graph.node2graph, dim=0, dim_size=graph.batch_size)
            num_all = torch.einsum("baa -> b", x)
            cut_loss = (1 - num_intra / (num_all + self.eps)).mean()
            metric["normalized cut loss"] = cut_loss

            x = torch.einsum("na, nc -> nac", assignment, assignment)
            x = scatter_add(x, graph.node2graph, dim=0, dim_size=graph.batch_size)
            x = x / x.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1)
            x = x - torch.eye(self.output_node, device=x.device) / (self.output_node ** 0.5)
            regularization = x.flatten(-2).norm(dim=-1).mean()
            metric["orthogonal regularization"] = regularization
            if self.loss_weight > 0:
                all_loss += (cut_loss + regularization) * self.loss_weight

        if self.zero_diagonal:
            edge_list = new_graph.edge_list[:, :2]
            is_diagonal = edge_list[:, 0] == edge_list[:, 1]
            new_graph = new_graph.edge_mask(~is_diagonal)

        return new_graph, output, assignment