import torch
from torch import nn

from torchdrug import core, models
from torchdrug.layers import functional
from torchdrug.core import Registry as R
from torchdrug.utils import doc


@R.register("models.KBGAT")
@doc.copy_args(models.GraphAttentionNetwork)
class KnowledgeBaseGraphAttentionNetwork(models.GraphAttentionNetwork, core.Configurable):
    """
    Knowledge Base Graph Attention Network proposed in
    `Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs`_.

    .. _Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs:
        https://arxiv.org/pdf/1906.01195.pdf

    Parameters:
        num_entity (int): number of entities
        num_relation (int): number of relations
        embedding_dim (int): dimension of embeddings
        hidden_dims (list of int): hidden dimensions
        max_score (float, optional): maximal score for triplets
        **kwargs
    """

    def __init__(self, num_entity, num_relation, embedding_dim, hidden_dims, max_score=12, **kwargs):
        super(KnowledgeBaseGraphAttentionNetwork, self).__init__(embedding_dim, hidden_dims, embedding_dim, **kwargs)
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.max_score = max_score

        self.linear = nn.Linear(self.output_dim, embedding_dim)
        self.output_dim = embedding_dim

        self.entity = nn.Parameter(torch.zeros(num_entity, embedding_dim))
        self.relation = nn.Parameter(torch.zeros(num_relation, embedding_dim))

        nn.init.uniform_(self.entity, -max_score / embedding_dim, max_score / embedding_dim)
        nn.init.uniform_(self.relation, -max_score / embedding_dim, max_score / embedding_dim)

    def forward(self, graph, h_index, t_index, r_index, all_loss=None, metric=None):
        """
        Compute the score for triplets.

        Parameters:
            graph (Graph): fact graph
            h_index (Tensor): indexes of head entities
            t_index (Tensor): indexes of tail entities
            r_index (Tensor): indexes of relations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        """
        with graph.edge():
            graph.edge_feature = self.relation[graph.edge_list[:, 2]].detach()

        output = super(KnowledgeBaseGraphAttentionNetwork, self).forward(graph, self.entity, all_loss, metric)
        entity = self.linear(output["node_feature"])
        score = functional.transe_score(entity, self.relation, h_index, t_index, r_index)
        return self.max_score - score

