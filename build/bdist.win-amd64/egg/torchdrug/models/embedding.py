import torch
from torch import nn

from torchdrug import core
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("models.TransE")
class TransE(nn.Module, core.Configurable):
    """
    TransE embedding proposed in `Translating Embeddings for Modeling Multi-relational Data`_.

    .. _Translating Embeddings for Modeling Multi-relational Data:
        https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf

    Parameters:
        num_entity (int): number of entities
        num_relation (int): number of relations
        embedding_dim (int): dimension of embeddings
        max_score (float, optional): maximal score for triplets
    """

    def __init__(self, num_entity, num_relation, embedding_dim, max_score=12):
        super(TransE, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.max_score = max_score

        self.entity = nn.Parameter(torch.empty(num_entity, embedding_dim))
        self.relation = nn.Parameter(torch.empty(num_relation, embedding_dim))

        nn.init.uniform_(self.entity, -self.max_score / embedding_dim, self.max_score / embedding_dim)
        nn.init.uniform_(self.relation, -self.max_score / embedding_dim, self.max_score / embedding_dim)

    def forward(self, graph, h_index, t_index, r_index, all_loss=None, metric=None):
        """
        Compute the score for each triplet.

        Parameters:
            graph (Graph): fact graph
            h_index (Tensor): indexes of head entities
            t_index (Tensor): indexes of tail entities
            r_index (Tensor): indexes of relations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        """
        score = functional.transe_score(self.entity, self.relation, h_index, t_index, r_index)
        return self.max_score - score


@R.register("models.DistMult")
class DistMult(nn.Module, core.Configurable):
    """
    DistMult embedding proposed in `Embedding Entities and Relations for Learning and Inference in Knowledge Bases`_.

    .. _Embedding Entities and Relations for Learning and Inference in Knowledge Bases:
        https://arxiv.org/pdf/1412.6575.pdf

    Parameters:
        num_entity (int): number of entities
        num_relation (int): number of relations
        embedding_dim (int): dimension of embeddings
        l3_regularization (float, optional): weight for l3 regularization
    """

    def __init__(self, num_entity, num_relation, embedding_dim, l3_regularization=0):
        super(DistMult, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.l3_regularization = l3_regularization

        self.entity = nn.Parameter(torch.empty(num_entity, embedding_dim))
        self.relation = nn.Parameter(torch.empty(num_relation, embedding_dim))

        nn.init.uniform_(self.entity, -0.5, 0.5)
        nn.init.uniform_(self.relation, -0.5, 0.5)

    def forward(self, graph, h_index, t_index, r_index, all_loss=None, metric=None):
        """
        Compute the score for each triplet.

        Parameters:
            graph (Graph): fact graph
            h_index (Tensor): indexes of head entities
            t_index (Tensor): indexes of tail entities
            r_index (Tensor): indexes of relations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        """
        score = functional.distmult_score(self.entity, self.relation, h_index, t_index, r_index)

        if all_loss is not None and self.l3_regularization > 0:
            loss = (self.entity.abs() ** 3).sum() + (self.relation.abs() ** 3).sum()
            all_loss += loss * self.l3_regularization
            metric["l3 regularization"] = loss / (self.num_entity + self.num_relation)

        return score


@R.register("models.ComplEx")
class ComplEx(nn.Module, core.Configurable):
    """
    ComplEx embedding proposed in `Complex Embeddings for Simple Link Prediction`_.

    .. _Complex Embeddings for Simple Link Prediction:
        http://proceedings.mlr.press/v48/trouillon16.pdf

    Parameters:
        num_entity (int): number of entities
        num_relation (int): number of relations
        embedding_dim (int): dimension of embeddings
        l3_regularization (float, optional): weight for l3 regularization
    """

    def __init__(self, num_entity, num_relation, embedding_dim, l3_regularization=0):
        super(ComplEx, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.l3_regularization = l3_regularization

        self.entity = nn.Parameter(torch.empty(num_entity, embedding_dim))
        self.relation = nn.Parameter(torch.empty(num_relation, embedding_dim))

        nn.init.uniform_(self.entity, -0.5, 0.5)
        nn.init.uniform_(self.relation, -0.5, 0.5)

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
        score = functional.complex_score(self.entity, self.relation, h_index, t_index, r_index)

        if all_loss is not None and self.l3_regularization > 0:
            loss = (self.entity.abs() ** 3).sum() + (self.relation.abs() ** 3).sum()
            all_loss += loss * self.l3_regularization
            metric["l3 regularization"] = loss / (self.num_entity + self.num_relation)

        return score


@R.register("models.RotatE")
class RotatE(nn.Module, core.Configurable):
    """
    RotatE embedding proposed in `RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space`_.

    .. _RotatE\: Knowledge Graph Embedding by Relational Rotation in Complex Space:
        https://arxiv.org/pdf/1902.10197.pdf

    Parameters:
        num_entity (int): number of entities
        num_relation (int): number of relations
        embedding_dim (int): dimension of embeddings
        max_score (float, optional): maximal score for triplets
    """

    def __init__(self, num_entity, num_relation, embedding_dim, max_score=12):
        super(RotatE, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.max_score = max_score

        self.entity = nn.Parameter(torch.empty(num_entity, embedding_dim))
        self.relation = nn.Parameter(torch.empty(num_relation, embedding_dim // 2))

        nn.init.uniform_(self.entity, -max_score * 2 / embedding_dim, max_score * 2 / embedding_dim)
        nn.init.uniform_(self.relation, -max_score * 2 / embedding_dim, max_score * 2 / embedding_dim)
        pi = torch.acos(torch.zeros(1)).item() * 2
        self.relation_scale = pi * embedding_dim / max_score / 2

    def forward(self, graph, h_index, t_index, r_index, all_loss=None, metric=None):
        """
        Compute the score for each triplet.

        Parameters:
            graph (Graph): fact graph
            h_index (Tensor): indexes of head entities
            t_index (Tensor): indexes of tail entities
            r_index (Tensor): indexes of relations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        """
        score = functional.rotate_score(self.entity, self.relation * self.relation_scale,
                                        h_index, t_index, r_index)
        return self.max_score - score


@R.register("models.SimplE")
class SimplE(nn.Module, core.Configurable):
    """
    SimplE embedding proposed in `SimplE Embedding for Link Prediction in Knowledge Graphs`_.

    .. _SimplE Embedding for Link Prediction in Knowledge Graphs:
        https://papers.nips.cc/paper/2018/file/b2ab001909a8a6f04b51920306046ce5-Paper.pdf

    Parameters:
        num_entity (int): number of entities
        num_relation (int): number of relations
        embedding_dim (int): dimension of embeddings
        l3_regularization (float, optional): maximal score for triplets
    """

    def __init__(self, num_entity, num_relation, embedding_dim, l3_regularization=0):
        super(SimplE, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.l3_regularization = l3_regularization

        self.entity = nn.Parameter(torch.empty(num_entity, embedding_dim))
        self.relation = nn.Parameter(torch.empty(num_relation, embedding_dim))

        nn.init.uniform_(self.entity, -0.5, 0.5)
        nn.init.uniform_(self.relation, -0.5, 0.5)

    def forward(self, graph, h_index, t_index, r_index, all_loss=None, metric=None):
        """
        Compute the score for each triplet.

        Parameters:
            graph (Graph): fact graph
            h_index (Tensor): indexes of head entities
            t_index (Tensor): indexes of tail entities
            r_index (Tensor): indexes of relations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        """
        score = functional.simple_score(self.entity, self.relation, h_index, t_index, r_index)

        if all_loss is not None and self.l3_regularization > 0:
            loss = (self.entity.abs() ** 3).sum() + (self.relation.abs() ** 3).sum()
            all_loss += loss * self.l3_regularization
            metric["l3 regularization"] = loss / (self.num_entity + self.num_relation)

        return score