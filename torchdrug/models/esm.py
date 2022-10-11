import os
import warnings

import torch
from torch import nn
import esm

from torchdrug import core, layers, utils, data
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("models.ESM")
class EvolutionaryScaleModeling(nn.Module, core.Configurable):
    """
    The protein language model, Evolutionary Scale Modeling (ESM) proposed in
    `Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences`_.

    .. _Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences:
        https://www.biorxiv.org/content/10.1101/622803v1.full.pdf

    Parameters:
        path (str): path to store ESM model weights
        model (str, optional): model name. Available model names are ``ESM-1b``, ``ESM-1v`` and ``ESM-1b-regression``.
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    url = {
        "ESM-1b": "https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt",
        "ESM-1v": "https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt",
        "ESM-1b-regression":
            "https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt",
    }

    md5 = {
        "ESM-1b": "ba8914bc3358cae2254ebc8874ee67f6",
        "ESM-1v": "1f04c2d2636b02b544ecb5fbbef8fefd",
        "ESM-1b-regression": "e7fe626dfd516fb6824bd1d30192bdb1",
    }

    output_dim = {
        "ESM-1b": 1280,
        "ESM-1v": 1280,
    }

    max_input_length = 1024 - 2

    def __init__(self, path, model="ESM-1b", readout="mean"):
        super(EvolutionaryScaleModeling, self).__init__()
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        _model, alphabet = self.load_weight(path, model)
        mapping = self.construct_mapping(alphabet)
        self.output_dim = self.output_dim[model]
        self.model = _model
        self.alphabet = alphabet
        self.register_buffer("mapping", mapping)

        if readout == "sum":
            self.readout = layers.SumReadout("residue")
        elif readout == "mean":
            self.readout = layers.MeanReadout("residue")
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def load_weight(self, path, model):
        if model not in self.url:
            raise ValueError("Unknown model `%s`" % model)
        model_file = utils.download(self.url[model], path, md5=self.md5[model])
        model_data = torch.load(model_file, map_location="cpu")
        if model != "ESM-1v":
            regression_model = "%s-regression" % model
            regression_file = utils.download(self.url[regression_model], path, md5=self.md5[regression_model])
            regression_data = torch.load(regression_file, map_location="cpu")
        else:
            regression_data = None
        model_name = os.path.basename(self.url[model])
        return esm.pretrained.load_model_and_alphabet_core(model_name, model_data, regression_data)

    def construct_mapping(self, alphabet):
        mapping = [0] * len(data.Protein.id2residue_symbol)
        for i, token in data.Protein.id2residue_symbol.items():
            mapping[i] = alphabet.get_idx(token)
        mapping = torch.tensor(mapping)
        return mapping

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the residue representations and the graph representation(s).

        Parameters:
            graph (Protein): :math:`n` protein(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``residue_feature`` and ``graph_feature`` fields:
                residue representations of shape :math:`(|V_{res}|, d)`, graph representations of shape :math:`(n, d)`
        """
        input = graph.residue_type
        input = self.mapping[input]
        size = graph.num_residues
        if (size > self.max_input_length).any():
            warnings.warn("ESM can only encode proteins within %d residues. Truncate the input to fit into ESM."
                          % self.max_input_length)
            starts = size.cumsum(0) - size
            size = size.clamp(max=self.max_input_length)
            ends = starts + size
            mask = functional.multi_slice_mask(starts, ends, graph.num_residue)
            input = input[mask]
            graph = graph.subresidue(mask)
        size_ext = size
        if self.alphabet.prepend_bos:
            bos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.alphabet.cls_idx
            input, size_ext = functional._extend(bos, torch.ones_like(size_ext), input, size_ext)
        if self.alphabet.append_eos:
            eos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.alphabet.eos_idx
            input, size_ext = functional._extend(input, size_ext, eos, torch.ones_like(size_ext))
        input = functional.variadic_to_padded(input, size_ext, value=self.alphabet.padding_idx)[0]

        output = self.model(input, repr_layers=[33])
        residue_feature = output["representations"][33]

        residue_feature = functional.padded_to_variadic(residue_feature, size_ext)
        starts = size_ext.cumsum(0) - size_ext
        if self.alphabet.prepend_bos:
            starts = starts + 1
        ends = starts + size
        mask = functional.multi_slice_mask(starts, ends, len(residue_feature))
        residue_feature = residue_feature[mask]
        graph_feature = self.readout(graph, residue_feature)

        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature
        }