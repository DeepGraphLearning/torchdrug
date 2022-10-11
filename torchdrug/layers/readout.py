import torch
from torch import nn
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter_softmax


class Readout(nn.Module):

    def __init__(self, type="node"):
        super(Readout, self).__init__()
        self.type = type

    def get_index2graph(self, graph):
        if self.type == "node":
            input2graph = graph.node2graph
        elif self.type == "edge":
            input2graph = graph.edge2graph
        elif self.type == "residue":
            input2graph = graph.residue2graph
        else:
            raise ValueError("Unknown input type `%s` for readout functions" % self.type)
        return input2graph


class MeanReadout(Readout):
    """Mean readout operator over graphs with variadic sizes."""

    def forward(self, graph, input):
        """
        Perform readout over the graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations

        Returns:
            Tensor: graph representations
        """
        input2graph = self.get_index2graph(graph)
        output = scatter_mean(input, input2graph, dim=0, dim_size=graph.batch_size)
        return output


class SumReadout(Readout):
    """Sum readout operator over graphs with variadic sizes."""

    def forward(self, graph, input):
        """
        Perform readout over the graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations

        Returns:
            Tensor: graph representations
        """
        input2graph = self.get_index2graph(graph)
        output = scatter_add(input, input2graph, dim=0, dim_size=graph.batch_size)
        return output


class MaxReadout(Readout):
    """Max readout operator over graphs with variadic sizes."""

    def forward(self, graph, input):
        """
        Perform readout over the graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations

        Returns:
            Tensor: graph representations
        """
        input2graph = self.get_index2graph(graph)
        output = scatter_max(input, input2graph, dim=0, dim_size=graph.batch_size)[0]
        return output


class AttentionReadout(Readout):
    """Attention readout operator over graphs with variadic sizes."""

    def __init__(self, input_dim, type="node"):
        super(AttentionReadout, self).__init__(type)
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, graph, input):
        index2graph = self.get_index2graph(graph)
        weight = self.linear(input)
        attention = scatter_softmax(weight, index2graph, dim=0)
        output = scatter_add(attention * input, index2graph, dim=0, dim_size=graph.batch_size)
        return output


class Softmax(Readout):
    """Softmax operator over graphs with variadic sizes."""

    eps = 1e-10

    def forward(self, graph, input):
        """
        Perform softmax over the graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node logits

        Returns:
            Tensor: node probabilities
        """
        input2graph = self.get_index2graph(graph)
        x = input - scatter_max(input, input2graph, dim=0, dim_size=graph.batch_size)[0][input2graph]
        x = x.exp()
        normalizer = scatter_add(x, input2graph, dim=0, dim_size=graph.batch_size)[input2graph]
        return x / (normalizer + self.eps)


class Sort(Readout):
    """
    Sort operator over graphs with variadic sizes.

    Parameters:
        descending (bool, optional): use descending sort order or not
    """

    def __init__(self, type="node", descending=False):
        super(Sort, self).__init__(type)
        self.descending = descending

    def forward(self, graph, input):
        """
        Perform sort over graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node values

        Returns:
            (Tensor, LongTensor): sorted values, sorted indices
        """
        input2graph = self.get_index2graph(graph)
        step = input.max(dim=0) - input.min(dim=0) + 1
        if self.descending:
            step = -step
        x = input + input2graph * step
        sorted, index = x.sort(dim=0, descending=self.descending)
        sorted = sorted - input2graph * step
        return sorted, index


class Set2Set(Readout):
    """
    Set2Set operator from `Order Matters: Sequence to sequence for sets`_.

    .. _Order Matters\: Sequence to sequence for sets:
        https://arxiv.org/pdf/1511.06391.pdf

    Parameters:
        input_dim (int): input dimension
        num_step (int, optional): number of process steps
        num_lstm_layer (int, optional): number of LSTM layers
    """

    def __init__(self, input_dim, type="node", num_step=3, num_lstm_layer=1):
        super(Set2Set, self).__init__(type)
        self.input_dim = input_dim
        self.output_dim = self.input_dim * 2
        self.num_step = num_step
        self.lstm = nn.LSTM(input_dim * 2, input_dim, num_lstm_layer)
        self.softmax = Softmax(type)

    def forward(self, graph, input):
        """
        Perform Set2Set readout over graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations

        Returns:
            Tensor: graph representations
        """
        input2graph = self.get_index2graph(graph)
        hx = (torch.zeros(self.lstm.num_layers, graph.batch_size, self.lstm.hidden_size, device=input.device),) * 2
        query_star = torch.zeros(graph.batch_size, self.output_dim, device=input.device)

        for i in range(self.num_step):
            query, hx = self.lstm(query_star.unsqueeze(0), hx)
            query = query.squeeze(0)
            product = torch.einsum("bd, bd -> b", query[input2graph], input)
            attention = self.softmax(graph, product)
            output = scatter_add(attention.unsqueeze(-1) * input, input2graph, dim=0, dim_size=graph.batch_size)
            query_star = torch.cat([query, output], dim=-1)

        return query_star