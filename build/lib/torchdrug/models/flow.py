import torch
from torch import nn

from torchdrug import core, layers
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("models.GraphAF")
class GraphAutoregressiveFlow(nn.Module, core.Configurable):
    """
    Graph autoregressive flow proposed in `GraphAF: a Flow-based Autoregressive Model for Molecular Graph Generation`_.

    .. _GraphAF\: a Flow-based Autoregressive Model for Molecular Graph Generation:
        https://arxiv.org/pdf/2001.09382.pdf

    Parameters:
        model (nn.Module): graph representation model
        prior (nn.Module): prior distribution
        use_edge (bool, optional): use edge or not
        num_flow_layer (int, optional): number of conditional flow layers
        num_mlp_layer (int, optional): number of MLP layers in each conditional flow
        dequantization_noise (float, optional): scale of dequantization noise
    """

    def __init__(self, model, prior, use_edge=False, num_layer=6, num_mlp_layer=2, dequantization_noise=0.9):
        super(GraphAutoregressiveFlow, self).__init__()
        self.model = model
        self.prior = prior
        self.use_edge = use_edge
        self.input_dim = self.output_dim = prior.dim
        self.dequantization_noise = dequantization_noise
        assert dequantization_noise < 1

        self.layers = nn.ModuleList()
        for i in range(num_layer):
            condition_dim = model.output_dim * (3 if use_edge else 1)
            self.layers.append(layers.ConditionalFlow(self.input_dim, condition_dim,
                                                      [model.output_dim] * (num_mlp_layer - 1)))

    def _standarize_edge(self, graph, edge):
        if edge is not None:
            edge = edge.clone()
            if (edge[:, :2] >= graph.num_nodes.unsqueeze(-1)).any():
                raise ValueError("Edge index exceeds the number of nodes in the graph")
            edge[:, :2] += (graph.num_cum_nodes - graph.num_nodes).unsqueeze(-1)
        return edge

    def forward(self, graph, input, edge=None, all_loss=None, metric=None):
        """
        Compute the log-likelihood for the input given the graph(s).

        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): discrete data of shape :math:`(n,)`
            edge (Tensor, optional): edge list of shape :math:`(n, 2)`.
                If specified, additionally condition on the edge for each input.
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        """
        if self.use_edge and edge is None:
            raise ValueError("`use_edge` is true, but no edge is provided")

        edge = self._standarize_edge(graph, edge)

        node_feature = functional.one_hot(graph.atom_type, self.model.input_dim)
        feature = self.model(graph, node_feature, all_loss, metric)
        node_feature = feature["node_feature"]
        graph_feature = feature["graph_feature"]
        if self.use_edge:
            condition = torch.cat([node_feature[edge], graph_feature.unsqueeze(1)], dim=1).flatten(1)
        else:
            condition = graph_feature

        x = functional.one_hot(input, self.input_dim)
        x = x + self.dequantization_noise * torch.rand_like(x)

        log_dets = []
        for layer in self.layers:
            x, log_det = layer(x, condition)
            log_dets.append(log_det)

        log_prior = self.prior(x)
        log_det = torch.stack(log_dets).sum(dim=0)
        log_likelihood = log_prior + log_det
        log_likelihood = log_likelihood.sum(dim=-1)

        return log_likelihood # (batch_size,)

    def sample(self, graph, edge=None, all_loss=None, metric=None):
        """
        Sample discrete data based on the given graph(s).

        Parameters:
            graph (Graph): :math:`n` graph(s)
            edge (Tensor, optional): edge list of shape :math:`(n, 2)`.
                If specified, additionally condition on the edge for each input.
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        """
        if self.use_edge and edge is None:
            raise ValueError("`use_edge` is true, but no edge is provided")

        edge = self._standarize_edge(graph, edge)

        node_feature = functional.one_hot(graph.atom_type, self.model.input_dim)
        feature = self.model(graph, node_feature, all_loss, metric)
        node_feature = feature["node_feature"]
        graph_feature = feature["graph_feature"]
        if self.use_edge:
            condition = torch.cat([node_feature[edge], graph_feature.unsqueeze(1)], dim=1).flatten(1)
        else:
            condition = graph_feature

        x = self.prior.sample(len(graph))
        for layer in self.layers[::-1]:
            x, log_det = layer.reverse(x, condition)

        output = x.argmax(dim=-1)

        return output # (batch_size,)