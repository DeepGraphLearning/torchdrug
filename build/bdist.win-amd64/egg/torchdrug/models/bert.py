import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core, layers
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("models.ProteinBERT")
class ProteinBERT(nn.Module, core.Configurable):
    """
    Protein BERT proposed in `Evaluating Protein Transfer Learning with TAPE`_.

    .. _Evaluating Protein Transfer Learning with TAPE:
        https://arxiv.org/pdf/1906.08230.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dim (int, optional): hidden dimension
        num_layers (int, optional): number of Transformer blocks
        num_heads (int, optional): number of attention heads
        intermediate_dim (int, optional): intermediate hidden dimension of Transformer block
        activation (str or function, optional): activation function
        hidden_dropout (float, optional): dropout ratio of hidden features
        attention_dropout (float, optional): dropout ratio of attention maps
        max_position (int, optional): maximum number of positions
    """

    def __init__(self, input_dim, hidden_dim=768, num_layers=12, num_heads=12, intermediate_dim=3072,
                 activation="gelu", hidden_dropout=0.1, attention_dropout=0.1, max_position=8192):
        super(ProteinBERT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.max_position = max_position

        self.num_residue_type = input_dim
        self.embedding = nn.Embedding(input_dim + 3, hidden_dim)
        self.position_embedding = nn.Embedding(max_position, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(hidden_dropout)

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(layers.ProteinBERTBlock(hidden_dim, intermediate_dim, num_heads,
                                                       attention_dropout, hidden_dropout, activation))
        
        self.linear = nn.Linear(hidden_dim, hidden_dim)

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
        size_ext = graph.num_residues
        # Prepend BOS
        bos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.num_residue_type
        input, size_ext = functional._extend(bos, torch.ones_like(size_ext), input, size_ext)
        # Append EOS
        eos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * (self.num_residue_type + 1)
        input, size_ext = functional._extend(input, size_ext, eos, torch.ones_like(size_ext))
        # Padding
        input, mask = functional.variadic_to_padded(input, size_ext, value=self.num_residue_type + 2)
        mask = mask.long().unsqueeze(-1)

        input = self.embedding(input)
        position_indices = torch.arange(input.shape[1], device=input.device)
        input = input + self.position_embedding(position_indices).unsqueeze(0)
        input = self.layer_norm(input)
        input = self.dropout(input)

        for layer in self.layers:
            input = layer(input, mask)

        residue_feature = functional.padded_to_variadic(input, graph.num_residues)

        graph_feature = input[:, 0]
        graph_feature = self.linear(graph_feature)
        graph_feature = F.tanh(graph_feature)

        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature
        }
