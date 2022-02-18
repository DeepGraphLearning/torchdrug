import torch
from torch import nn
from torch_scatter import scatter_mean, scatter_add, scatter_max


class MeanReadout(nn.Module):
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
        output = scatter_mean(input, graph.node2graph, dim=0, dim_size=graph.batch_size)
        return output


class SumReadout(nn.Module):
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
        output = scatter_add(input, graph.node2graph, dim=0, dim_size=graph.batch_size)
        return output


class MaxReadout(nn.Module):
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
        output = scatter_max(input, graph.node2graph, dim=0, dim_size=graph.batch_size)[0]
        return output


class Softmax(nn.Module):
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
        x = input - scatter_max(input, graph.node2graph, dim=0, dim_size=graph.batch_size)[0][graph.node2graph]
        x = x.exp()
        normalizer = scatter_add(x, graph.node2graph, dim=0, dim_size=graph.batch_size)[graph.node2graph]
        return x / (normalizer + self.eps)


class Sort(nn.Module):
    """
    Sort operator over graphs with variadic sizes.

    Parameters:
        descending (bool, optional): use descending sort order or not
    """

    def __init__(self, descending=False):
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
        step = input.max(dim=0) - input.min(dim=0) + 1
        if self.descending:
            step = -step
        x = input + graph.node2graph * step
        sorted, index = x.sort(dim=0, descending=self.descending)
        sorted = sorted - graph.node2graph * step
        return sorted, index


class Set2Set(nn.Module):
    """
    Set2Set operator from `Order Matters: Sequence to sequence for sets`_.

    .. _Order Matters\: Sequence to sequence for sets:
        https://arxiv.org/pdf/1511.06391.pdf

    Parameters:
        input_dim (int): input dimension
        num_step (int, optional): number of process steps
        num_lstm_layer (int, optional): number of LSTM layers
    """

    def __init__(self, input_dim, num_step=3, num_lstm_layer=1):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.output_dim = self.input_dim * 2
        self.num_step = num_step
        self.lstm = nn.LSTM(input_dim * 2, input_dim, num_lstm_layer)
        self.softmax = Softmax()

    def forward(self, graph, input):
        """
        Perform Set2Set readout over graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations

        Returns:
            Tensor: graph representations
        """
        hx = (torch.zeros(self.lstm.num_layers, graph.batch_size, self.lstm.hidden_size, device=input.device),) * 2
        query_star = torch.zeros(graph.batch_size, self.output_dim, device=input.device)

        for i in range(self.num_step):
            query, hx = self.lstm(query_star.unsqueeze(0), hx)
            query = query.squeeze(0)
            product = torch.einsum("bd, bd -> b", query[graph.node2graph], input)
            attention = self.softmax(graph, product)
            output = scatter_add(attention.unsqueeze(-1) * input, graph.node2graph, dim=0, dim_size=graph.batch_size)
            query_star = torch.cat([query, output], dim=-1)

        return query_star