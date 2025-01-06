from torch import nn
from torch_scatter import scatter_add

from torchdrug.layers import functional


class NodeSampler(nn.Module):
    """
    Node sampler from `GraphSAINT: Graph Sampling Based Inductive Learning Method`_.

    .. _GraphSAINT\: Graph Sampling Based Inductive Learning Method:
        https://arxiv.org/pdf/1907.04931.pdf

    Parameters:
        budget (int, optional): number of node to keep
        ratio (int, optional): ratio of node to keep
    """

    def __init__(self, budget=None, ratio=None):
        super(NodeSampler, self).__init__()
        if budget is None and ratio is None:
            raise ValueError("At least one of `budget` and `ratio` should be provided")
        self.budget = budget
        self.ratio = ratio

    def forward(self, graph):
        """
        Sample a subgraph from the graph.

        Parameters:
            graph (Graph): graph(s)
        """
        # this is exact for a single graph
        # but approximate for packed graphs
        num_sample = graph.num_node
        if self.budget:
            num_sample = min(num_sample, self.budget)
        if self.ratio:
            num_sample = min(num_sample, int(self.ratio * graph.num_node))

        prob = scatter_add(graph.edge_weight ** 2, graph.edge_list[:, 1], dim_size=graph.num_node)
        prob /= prob.mean()
        index = functional.multinomial(prob, num_sample)
        new_graph = graph.node_mask(index)
        node_out = new_graph.edge_list[:, 1]
        new_graph._edge_weight /= num_sample * prob[node_out] / graph.num_node

        return new_graph


class EdgeSampler(nn.Module):
    """
    Edge sampler from `GraphSAINT: Graph Sampling Based Inductive Learning Method`_.

    .. _GraphSAINT\: Graph Sampling Based Inductive Learning Method:
        https://arxiv.org/pdf/1907.04931.pdf

    Parameters:
        budget (int, optional): number of node to keep
        ratio (int, optional): ratio of node to keep
    """

    def __init__(self, budget=None, ratio=None):
        super(EdgeSampler, self).__init__()
        if budget is None and ratio is None:
            raise ValueError("At least one of `budget` and `ratio` should be provided")
        self.budget = budget
        self.ratio = ratio

    def forward(self, graph):
        """
        Sample a subgraph from the graph.

        Parameters:
            graph (Graph): graph(s)
        """
        # this is exact for a single graph
        # but approximate for packed graphs
        node_in, node_out = graph.edge_list.t()[:2]
        num_sample = graph.num_edge
        if self.budget:
            num_sample = min(num_sample, self.budget)
        if self.ratio:
            num_sample = min(num_sample, int(self.ratio * graph.num_edge))

        prob = 1 / graph.degree_out[node_out] + 1 / graph.degree_in[node_in]
        prob = prob / prob.mean()
        index = functional.multinomial(prob, num_sample)
        new_graph = graph.edge_mask(index)
        new_graph._edge_weight /= num_sample * prob[index] / graph.num_edge

        return new_graph
