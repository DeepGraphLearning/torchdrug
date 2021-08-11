import copy

import torch
from torch import nn

from torchdrug import core, layers
from torchdrug.core import Registry as R


@R.register("models.InfoGraph")
class InfoGraph(nn.Module, core.Configurable):
    """
    InfoGraph proposed in
    `InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information
    Maximization`_.

    .. _InfoGraph\:
        Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization:
        https://arxiv.org/pdf/1908.01000.pdf

    Parameters:
        model (nn.Module): node & graph representation model
        num_mlp_layer (int, optional): number of MLP layers in mutual information estimators
        activation (str or function, optional): activation function
        loss_weight (float, optional): weight of both unsupervised & transfer losses
        separate_model (bool, optional): separate supervised and unsupervised encoders.
            If true, the unsupervised loss will be applied on a separate encoder,
            and a transfer loss is applied between the two encoders.
    """

    def __init__(self, model, num_mlp_layer=2, activation="relu", loss_weight=1, separate_model=False):
        super(InfoGraph, self).__init__()
        self.model = model
        self.separate_model = separate_model
        self.loss_weight = loss_weight
        self.output_dim = self.model.output_dim

        if separate_model:
            self.unsupervised_model = copy.deepcopy(model)
            self.transfer_mi = layers.MutualInformation(model.output_dim, num_mlp_layer, activation)
        else:
            self.unsupervised_model = model
        self.unsupervised_mi = layers.MutualInformation(model.output_dim, num_mlp_layer, activation)

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).
        Add the mutual information between graph and nodes to the loss.

        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        output = self.model(graph, input)

        if all_loss is not None:
            if self.separate_model:
                unsupervised_output = self.unsupervised_model(graph, input)
                mutual_info = self.transfer_mi(output["graph_feature"], unsupervised_output["graph_feature"])

                metric["distillation mutual information"] = mutual_info
                if self.loss_weight > 0:
                    all_loss -= mutual_info * self.loss_weight
            else:
                unsupervised_output = output

            graph_index = graph.node2graph
            node_index = torch.arange(graph.num_node, device=graph.device)
            pair_index = torch.stack([graph_index, node_index], dim=-1)

            mutual_info = self.unsupervised_mi(unsupervised_output["graph_feature"],
                                               unsupervised_output["node_feature"], pair_index)

            metric["graph-node mutual information"] = mutual_info
            if self.loss_weight > 0:
                all_loss -= mutual_info * self.loss_weight

        return output