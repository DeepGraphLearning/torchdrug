import torch
from torch import nn

from torch_scatter import scatter_add

from torchdrug import core, layers, data
from torchdrug.core import Registry as R


@R.register("models.Statistic")
class Statistic(nn.Module, core.Configurable):
    """
    The statistic feature engineering for protein sequence proposed in
    `Harnessing Computational Biology for Exact Linear B-cell Epitope Prediction`_.

    .. _Harnessing Computational Biology for Exact Linear B-cell Epitope Prediction:
        https://www.liebertpub.com/doi/abs/10.1089/omi.2015.0095

    Parameters:
        type (str, optional): statistic feature. Available feature is ``DDE``.
        hidden_dims (list of int, optional): hidden dimensions
    """

    num_residue_type = len(data.Protein.id2residue_symbol)
    input_dim = num_residue_type ** 2
    _codons = {"A": 4, "C": 2, "D": 2, "E": 2, "F": 2, "G": 4, "H": 2, "I": 3, "K": 2, "L": 6,
               "M": 1, "N": 2, "P": 4, "Q": 2, "R": 6, "S": 6, "T": 4, "V": 4, "W": 1, "Y": 2}

    def __init__(self, type="DDE", hidden_dims=(512,)):
        super(Statistic, self).__init__()
        self.type = type
        self.output_dim = hidden_dims[-1]
        
        codons = self.calculate_codons()
        self.register_buffer("codons", codons)
        self.mlp = layers.Sequential(
            layers.MultiLayerPerceptron(self.input_dim, hidden_dims),
            nn.ReLU()
        )

    def calculate_codons(self):
        codons = [0] * self.num_residue_type
        for i, token in data.Protein.id2residue_symbol.items():
            codons[i] = self._codons[token]
        codons = torch.tensor(codons)
        return codons

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the residue representations and the graph representation(s).

        Parameters:
            graph (Protein): :math:`n` protein(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``graph_feature`` field: graph representations of shape :math:`(n, d)`
        """
        input = graph.residue_type
        
        index = input[:-1] * self.num_residue_type + input[1:]
        index = graph.residue2graph[:-1] * self.input_dim + index
        value = torch.ones(graph.num_residue - 1, dtype=torch.float, device=graph.device)
        mask = graph.residue2graph[:-1] == graph.residue2graph[1:]
        feature = scatter_add(value * mask.float(), index, dim=0, dim_size=graph.batch_size * self.input_dim)
        feature = feature.view(graph.batch_size, self.input_dim)
        feature = feature / (feature.sum(dim=-1, keepdim=True) + 1e-10)
        if self.type == "DDE":
            TM = self.codons.unsqueeze(0) * self.codons.unsqueeze(1) / 61 ** 2
            TM = TM.flatten()
            TV = (TM * (1 - TM)).unsqueeze(0) / (graph.num_residues - 1 + 1e-10).unsqueeze(1)
            feature = (feature - TM.unsqueeze(0)) / (TV.sqrt() + 1e-10)
        else:
            raise ValueError("Unknown statistic feature type `%s`" % self.type)

        graph_feature = self.mlp(feature)

        return {
            "graph_feature": graph_feature,
        }