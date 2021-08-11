import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add

from torchdrug import core, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("models.NeuralLP")
class NeuralLogicProgramming(nn.Module, core.Configurable):
    """
    Neural Logic Programming proposed in `Differentiable Learning of Logical Rules for Knowledge Base Reasoning`_.

    .. _Differentiable Learning of Logical Rules for Knowledge Base Reasoning:
        https://papers.nips.cc/paper/2017/file/0e55666a4ad822e0e34299df3591d979-Paper.pdf

    Parameters:
        num_entity (int): number of entities
        num_relation (int): number of relations
        hidden_dim (int): dimension of hidden units in LSTM
        num_step (int): number of recurrent steps
        num_lstm_layer (int, optional): number of LSTM layers
    """

    eps = 1e-10

    def __init__(self, num_entity, num_relation, hidden_dim, num_step, num_lstm_layer=1):
        super(NeuralLogicProgramming, self).__init__()

        num_relation = int(num_relation)
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.num_step = num_step

        self.query = nn.Embedding(num_relation * 2 + 1, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_lstm_layer)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.weight_linear = nn.Linear(hidden_dim, num_relation * 2)
        self.linear = nn.Linear(1, 1)

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        return new_h_index, new_t_index, new_r_index

    @utils.cached
    def get_t_output(self, graph, h_index, r_index):
        end_index = torch.ones_like(r_index) * graph.num_relation
        q_index = torch.stack([r_index] * (self.num_step - 1) + [end_index], dim=0)
        query = self.query(q_index)

        hidden, hx = self.lstm(query)
        memory = functional.one_hot(h_index, self.num_entity).unsqueeze(0)

        for i in range(self.num_step):
            key = hidden[i]
            value = hidden[:i + 1]
            x = torch.einsum("bd, tbd -> bt", key, value)
            attention = F.softmax(x, dim=-1)
            input = torch.einsum("bt, tbn -> nb", attention, memory)
            weight = F.softmax(self.weight_linear(key), dim=-1).t()

            node_in, node_out, relation = graph.edge_list.t()
            if graph.num_node * graph.num_relation < graph.num_edge:
                # O(|V|d) memory
                node_out = node_out * graph.num_relation + relation
                adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), graph.edge_weight,
                                                    (graph.num_node, graph.num_node * graph.num_relation))
                output = adjacency.t() @ input
                output = output.view(graph.num_node, graph.num_relation, -1)
                output = (output * weight).sum(dim=1)
            else:
                # O(|E|) memory
                message = input[node_in]
                message = message * weight[relation]
                edge_weight = graph.edge_weight.unsqueeze(-1)
                output = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            output = output / output.sum(dim=0, keepdim=True).clamp(self.eps)

            memory = torch.cat([memory, output.t().unsqueeze(0)])

        return output

    def forward(self, graph, h_index, t_index, r_index, all_loss=None, metric=None):
        """
        Compute the score for triplets.

        Parameters:
            graph (Tensor): fact graph
            h_index (Tensor): indexes of head entities
            t_index (Tensor): indexes of tail entities
            r_index (Tensor): indexes of relations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        """
        assert graph.num_relation == self.num_relation
        graph = graph.undirected(add_inverse=True)

        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        hr_index = h_index * graph.num_relation + r_index
        hr_index_set, hr_inverse = torch.unique(hr_index, return_inverse=True)
        h_index_set = hr_index_set // graph.num_relation
        r_index_set = hr_index_set % graph.num_relation

        output = self.get_t_output(graph, h_index_set, r_index_set)

        score = output[t_index, hr_inverse]
        score = self.linear(score.unsqueeze(-1)).squeeze(-1)
        return score