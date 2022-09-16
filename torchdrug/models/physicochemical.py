import os

import torch
from torch import nn

from torch_scatter import scatter_mean, scatter_add

from torchdrug import core, layers, utils, data
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("models.Physicochemical")
class Physicochemical(nn.Module, core.Configurable):
    """
    The physicochemical feature engineering for protein sequence proposed in
    `Prediction of Membrane Protein Types based on the Hydrophobic Index of Amino Acids`_.

    .. _Prediction of Membrane Protein Types based on the Hydrophobic Index of Amino Acids:
        https://link.springer.com/article/10.1023/A:1007091128394

    Parameters:
        path (str): path to store feature file
        type (str, optional): physicochemical feature. Available features are ``moran``, ``geary`` and ``nmbroto``.
        nlag (int, optional): maximum position interval to compute features
        hidden_dims (list of int, optional): hidden dimensions
    """

    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/documents/AAidx.txt"
    md5 = "ec612f4df41b93ae03c31ae376c23ce0"

    property_key = ["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102",
                    "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"]
    num_residue_type = len(data.Protein.id2residue_symbol)

    def __init__(self, path, type="moran", nlag=30, hidden_dims=(512,)):
        super(Physicochemical, self).__init__()
        self.type = type
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        index_file = utils.download(self.url, path, md5=self.md5)
        property = self.read_property(index_file)
        self.register_buffer("property", property)

        self.nlag = nlag
        self.input_dim = len(self.property_key) * nlag
        self.output_dim = hidden_dims[-1]
        
        self.mlp = layers.Sequential(
            layers.MultiLayerPerceptron(self.input_dim, hidden_dims),
            nn.ReLU()
        )

    def read_property(self, file):
        with open(file, "r") as fin:
            lines = fin.readlines()
        vocab = lines[0].strip().split("\t")[1:]

        property_dict = {}
        for line in lines[1:]:
            line = line.strip().split("\t")
            property_dict[line[0]] = [float(x) if x != "NA" else 0 for x in line[1:]]

        _property = []
        for key in self.property_key:
            _property.append(property_dict[key])
        _property = torch.tensor(_property)
        mapping = [data.Protein.residue_symbol2id[residue] for residue in vocab]
        property = torch.zeros((len(self.property_key), self.num_residue_type), dtype=torch.float)
        property[:, mapping] = _property

        property = (property - property.mean(dim=1, keepdim=True)) / \
                        (property.std(dim=1, keepdim=True) + 1e-10)
        return property.transpose(0, 1)

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
        
        x = self.property[input]    # num_residue * 8
        x_mean = scatter_mean(x, graph.residue2graph, dim=0, dim_size=graph.batch_size)    # batch_size * 8

        size = graph.num_residues
        starts = size.cumsum(0) - size  # batch_size * nlag
        starts = starts.unsqueeze(-1).expand(-1, self.nlag)
        steps = torch.arange(self.nlag, dtype=torch.long, device=graph.device) + 1
        steps = (graph.num_residues.unsqueeze(-1) - steps.unsqueeze(0)).clamp(min=0)
        ends = starts + steps
        mask_0 = functional.multi_slice_mask(starts, ends, graph.num_residue) # num_residue * nlag

        ends = size.cumsum(0)  # batch_size * nlag
        ends = ends.unsqueeze(-1).expand(-1, self.nlag)
        starts = ends - steps
        mask_1 = functional.multi_slice_mask(starts, ends, graph.num_residue) # num_residue * nlag

        index2sample = functional._size_to_index(size)
        numerator = torch.zeros((graph.num_residue, self.nlag, x.shape[-1]), dtype=torch.float, device=graph.device)
        if self.type == "moran":
            _numerator = (x - x_mean[index2sample]).unsqueeze(1).expand(-1, self.nlag, -1)[mask_0] * \
                         (x - x_mean[index2sample]).unsqueeze(1).expand(-1, self.nlag, -1)[mask_1]
            numerator[mask_0] = _numerator
            numerator = numerator / (steps[index2sample].unsqueeze(-1) + 1e-10)
            numerator = scatter_add(numerator, graph.residue2graph, dim=0, dim_size=graph.batch_size) # batch_size * nlag * 8
            demonimator = scatter_add((x - x_mean[index2sample]) ** 2, graph.residue2graph, dim=0, dim_size=graph.batch_size)
            demonimator = demonimator / graph.num_residues.unsqueeze(-1)
            demonimator = demonimator.unsqueeze(1)  # batch_size * 1 * 8
        elif self.type == "geary":
            _numerator = x.unsqueeze(1).expand(-1, self.nlag, -1)[mask_0] - \
                         x.unsqueeze(1).expand(-1, self.nlag, -1)[mask_1]
            _numerator = _numerator ** 2
            numerator[mask_0] = _numerator
            numerator = numerator / (steps[index2sample].unsqueeze(-1) * 2 + 1e-10)
            numerator = scatter_add(numerator, graph.residue2graph, dim=0, dim_size=graph.batch_size) # batch_size * nlag * 8
            demonimator = scatter_add((x - x_mean[index2sample]) ** 2, graph.residue2graph, dim=0, dim_size=graph.batch_size)
            demonimator = demonimator / (graph.num_residues - 1 + 1e-10).unsqueeze(-1)
            demonimator = demonimator.unsqueeze(1)  # batch_size * 1 * 8
        elif self.type == "nmbroto":
            _numerator = x.unsqueeze(1).expand(-1, self.nlag, -1)[mask_0] * \
                         x.unsqueeze(1).expand(-1, self.nlag, -1)[mask_1]
            numerator[mask_0] = _numerator
            numerator = scatter_add(numerator, graph.residue2graph, dim=0, dim_size=graph.batch_size) # batch_size * nlag * 8
            demonimator = steps.unsqueeze(-1)   # batch_size * nlag * 1
        else:
            raise ValueError("Unknown physicochemical feature type `%s`" % self.type)
        feature = numerator / (demonimator + 1e-10)
        feature = feature.flatten(1, 2)

        graph_feature = self.mlp(feature)

        return {
            "graph_feature": graph_feature,
        }