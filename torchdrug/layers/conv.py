import functools

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint
from torch_scatter import scatter_mean, scatter_add, scatter_max

from torchdrug import data, layers, utils
from torchdrug.layers import functional


class MessagePassingBase(nn.Module):
    """
    Base module for message passing.

    Any custom message passing module should be derived from this class.
    """
    gradient_checkpoint = False

    def message(self, graph, input):
        """
        Compute edge messages for the graph.

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations of shape :math:`(|V|, ...)`

        Returns:
            Tensor: edge messages of shape :math:`(|E|, ...)`
        """
        raise NotImplementedError

    def aggregate(self, graph, message):
        """
        Aggregate edge messages to nodes.

        Parameters:
            graph (Graph): graph(s)
            message (Tensor): edge messages of shape :math:`(|E|, ...)`

        Returns:
            Tensor: node updates of shape :math:`(|V|, ...)`
        """
        raise NotImplementedError

    def message_and_aggregate(self, graph, input):
        """
        Fused computation of message and aggregation over the graph.
        This may provide better time or memory complexity than separate calls of
        :meth:`message <MessagePassingBase.message>` and :meth:`aggregate <MessagePassingBase.aggregate>`.

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations of shape :math:`(|V|, ...)`

        Returns:
            Tensor: node updates of shape :math:`(|V|, ...)`
        """
        message = self.message(graph, input)
        update = self.aggregate(graph, message)
        return update

    def _message_and_aggregate(self, *tensors):
        graph = data.Graph.from_tensors(tensors[:-1])
        input = tensors[-1]
        update = self.message_and_aggregate(graph, input)
        return update

    def combine(self, input, update):
        """
        Combine node input and node update.

        Parameters:
            input (Tensor): node representations of shape :math:`(|V|, ...)`
            update (Tensor): node updates of shape :math:`(|V|, ...)`
        """
        raise NotImplementedError

    def forward(self, graph, input):
        """
        Perform message passing over the graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations of shape :math:`(|V|, ...)`
        """
        if self.gradient_checkpoint:
            update = checkpoint.checkpoint(self._message_and_aggregate, *graph.to_tensors(), input)
        else:
            update = self.message_and_aggregate(graph, input)
        output = self.combine(input, update)
        return output


class GraphConv(MessagePassingBase):
    """
    Graph convolution operator from `Semi-Supervised Classification with Graph Convolutional Networks`_.

    .. _Semi-Supervised Classification with Graph Convolutional Networks:
        https://arxiv.org/pdf/1609.02907.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, output_dim, edge_input_dim=None, batch_norm=False, activation="relu"):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.linear = nn.Linear(input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def message(self, graph, input):
        # add self loop
        node_in = torch.cat([graph.edge_list[:, 0], torch.arange(graph.num_node, device=graph.device)])
        degree_in = graph.degree_in.unsqueeze(-1) + 1
        message = input[node_in]
        if self.edge_linear:
            edge_input = self.edge_linear(graph.edge_feature.float())
            edge_input = torch.cat([edge_input, torch.zeros(graph.num_node, self.input_dim, device=graph.device)])
            message += edge_input
        message /= (degree_in[node_in].sqrt() + 1e-10)
        return message

    def aggregate(self, graph, message):
        # add self loop
        node_out = torch.cat([graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        edge_weight = edge_weight.unsqueeze(-1)
        degree_out = graph.degree_out.unsqueeze(-1) + 1
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        update = update / (degree_out.sqrt() + 1e-10)
        return update

    def message_and_aggregate(self, graph, input):
        node_in, node_out = graph.edge_list.t()[:2]
        node_in = torch.cat([node_in, torch.arange(graph.num_node, device=graph.device)])
        node_out = torch.cat([node_out, torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        degree_in = graph.degree_in + 1
        degree_out = graph.degree_out + 1
        edge_weight = edge_weight / ((degree_in[node_in] * degree_out[node_out]).sqrt() + 1e-10)
        adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), edge_weight,
                                            (graph.num_node, graph.num_node))
        update = torch.sparse.mm(adjacency.t(), input)
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_input = torch.cat([self.edge_linear(edge_input), torch.zeros(graph.num_node, self.input_dim, device=graph.device)])
            edge_weight = edge_weight.unsqueeze(-1)
            node_out = torch.cat([graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)])
            edge_update = scatter_add(edge_input * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            update += edge_update

        return update

    def combine(self, input, update):
        output = self.linear(update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class GraphAttentionConv(MessagePassingBase):
    """
    Graph attentional convolution operator from `Graph Attention Networks`_.

    .. _Graph Attention Networks:
        https://arxiv.org/pdf/1710.10903.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        num_head (int, optional): number of attention heads
        negative_slope (float, optional): negative slope of leaky relu activation
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    eps = 1e-10

    def __init__(self, input_dim, output_dim, edge_input_dim=None, num_head=1, negative_slope=0.2, concat=True,
                 batch_norm=False, activation="relu"):
        super(GraphAttentionConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        self.num_head = num_head
        self.concat = concat
        self.leaky_relu = functools.partial(F.leaky_relu, negative_slope=negative_slope)

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if output_dim % num_head != 0:
            raise ValueError("Expect output_dim to be a multiplier of num_head, but found `%d` and `%d`"
                             % (output_dim, num_head))

        self.linear = nn.Linear(input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, output_dim)
        else:
            self.edge_linear = None
        self.query = nn.Parameter(torch.zeros(num_head, output_dim * 2 // num_head))
        nn.init.kaiming_uniform_(self.query, negative_slope, mode="fan_in")

    def message(self, graph, input):
        # add self loop
        node_in = torch.cat([graph.edge_list[:, 0], torch.arange(graph.num_node, device=graph.device)])
        node_out = torch.cat([graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        edge_weight = edge_weight.unsqueeze(-1)
        hidden = self.linear(input)

        key = torch.stack([hidden[node_in], hidden[node_out]], dim=-1)
        if self.edge_linear:
            edge_input = self.edge_linear(graph.edge_feature.float())
            edge_input = torch.cat([edge_input, torch.zeros(graph.num_node, self.output_dim, device=graph.device)])
            key += edge_input.unsqueeze(-1)
        key = key.view(-1, *self.query.shape)
        weight = torch.einsum("hd, nhd -> nh", self.query, key)
        weight = self.leaky_relu(weight)

        weight = weight - scatter_max(weight, node_out, dim=0, dim_size=graph.num_node)[0][node_out]
        attention = weight.exp() * edge_weight
        # why mean? because with mean we have normalized message scale across different node degrees
        normalizer = scatter_mean(attention, node_out, dim=0, dim_size=graph.num_node)[node_out]
        attention = attention / (normalizer + self.eps)

        value = hidden[node_in].view(-1, self.num_head, self.query.shape[-1] // 2)
        attention = attention.unsqueeze(-1).expand_as(value)
        message = (attention * value).flatten(1)
        return message

    def aggregate(self, graph, message):
        # add self loop
        node_out = torch.cat([graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)])
        update = scatter_mean(message, node_out, dim=0, dim_size=graph.num_node)
        return update

    def combine(self, input, update):
        output = update
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class GraphIsomorphismConv(MessagePassingBase):
    """
    Graph isomorphism convolution operator from `How Powerful are Graph Neural Networks?`_

    .. _How Powerful are Graph Neural Networks?:
        https://arxiv.org/pdf/1810.00826.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        hidden_dims (list of int, optional): hidden dimensions
        eps (float, optional): initial epsilon
        learn_eps (bool, optional): learn epsilon or not
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, output_dim, edge_input_dim=None, hidden_dims=None, eps=0, learn_eps=False,
                 batch_norm=False, activation="relu"):
        super(GraphIsomorphismConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim

        eps = torch.tensor([eps], dtype=torch.float32)
        if learn_eps:
            self.eps = nn.Parameter(eps)
        else:
            self.register_buffer("eps", eps)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if hidden_dims is None:
            hidden_dims = []
        self.mlp = layers.MLP(input_dim, list(hidden_dims) + [output_dim], activation)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def message(self, graph, input):
        node_in = graph.edge_list[:, 0]
        message = input[node_in]
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feature.float())
        return message

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        return update

    def message_and_aggregate(self, graph, input):
        adjacency = utils.sparse_coo_tensor(graph.edge_list.t()[:2], graph.edge_weight,
                                            (graph.num_node, graph.num_node))
        update = torch.sparse.mm(adjacency.t(), input)
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_weight = graph.edge_weight.unsqueeze(-1)
            edge_input = self.edge_linear(edge_input)
            edge_update = scatter_add(edge_input * edge_weight, graph.edge_list[:, 1], dim=0,
                                      dim_size=graph.num_node)
            update += edge_update

        return update

    def combine(self, input, update):
        output = self.mlp((1 + self.eps) * input + update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class RelationalGraphConv(MessagePassingBase):
    """
    Relational graph convolution operator from `Modeling Relational Data with Graph Convolutional Networks`_.

    .. _Modeling Relational Data with Graph Convolutional Networks:
        https://arxiv.org/pdf/1703.06103.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """
    eps = 1e-10

    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None, batch_norm=False, activation="relu"):
        super(RelationalGraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.edge_input_dim = edge_input_dim

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.self_loop = nn.Linear(input_dim, output_dim)
        self.linear = nn.Linear(num_relation * input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def message(self, graph, input):
        node_in = graph.edge_list[:, 0]
        message = input[node_in]
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feature.float())
        return message
    
    def aggregate(self, graph, message):
        assert graph.num_relation == self.num_relation

        node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node * self.num_relation) / \
                 (scatter_add(edge_weight, node_out, dim=0, dim_size=graph.num_node * self.num_relation) + self.eps)
        return update.view(graph.num_node, self.num_relation * self.input_dim)

    def message_and_aggregate(self, graph, input):
        assert graph.num_relation == self.num_relation

        node_in, node_out, relation = graph.edge_list.t()
        node_out = node_out * self.num_relation + relation
        degree_out = scatter_add(graph.edge_weight, node_out, dim_size=graph.num_node * graph.num_relation)
        edge_weight = graph.edge_weight / degree_out[node_out]
        adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), edge_weight,
                                            (graph.num_node, graph.num_node * graph.num_relation))
        update = torch.sparse.mm(adjacency.t(), input)
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_input = self.edge_linear(edge_input)
            edge_weight = edge_weight.unsqueeze(-1)
            edge_update = scatter_add(edge_input * edge_weight, node_out, dim=0,
                                      dim_size=graph.num_node * graph.num_relation)
            update += edge_update

        return update.view(graph.num_node, self.num_relation * self.input_dim)

    def combine(self, input, update):
        output = self.linear(update) + self.self_loop(input)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class NeuralFingerprintConv(MessagePassingBase):
    """
    Graph neural network operator from `Convolutional Networks on Graphs for Learning Molecular Fingerprints`_.

    Note this operator doesn't include the sparsifying step of the original paper.

    .. _Convolutional Networks on Graphs for Learning Molecular Fingerprints:
        https://arxiv.org/pdf/1509.09292.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, output_dim, edge_input_dim=None, batch_norm=False, activation="relu"):
        super(NeuralFingerprintConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.linear = nn.Linear(input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def message(self, graph, input):
        node_in = graph.edge_list[:, 0]
        message = input[node_in]
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feature.float())
        return message

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        return update

    def message_and_aggregate(self, graph, input):
        adjacency = utils.sparse_coo_tensor(graph.edge_list.t()[:2], graph.edge_weight,
                                            (graph.num_node, graph.num_node))
        update = torch.sparse.mm(adjacency.t(), input)
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_weight = graph.edge_weight.unsqueeze(-1)
            edge_input = self.edge_linear(edge_input)
            edge_update = scatter_add(edge_input * edge_weight, graph.edge_list[:, 1], dim=0,
                                      dim_size=graph.num_node)
            update += edge_update

        return update

    def combine(self, input, update):
        output = self.linear(input + update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class ContinuousFilterConv(MessagePassingBase):
    """
    Continuous filter operator from
    `SchNet: A continuous-filter convolutional neural network for modeling quantum interactions`_.

    .. _SchNet\: A continuous-filter convolutional neural network for modeling quantum interactions:
        https://arxiv.org/pdf/1706.08566.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        hidden_dim (int, optional): hidden dimension. By default, same as :attr:`output_dim`
        cutoff (float, optional): maximal scale for RBF kernels
        num_gaussian (int, optional): number of RBF kernels
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, output_dim, edge_input_dim=None, hidden_dim=None, cutoff=5, num_gaussian=100,
                 batch_norm=False, activation="shifted_softplus"):
        super(ContinuousFilterConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        if hidden_dim is None:
            hidden_dim = output_dim
        self.hidden_dim = hidden_dim
        self.rbf = layers.RBF(stop=cutoff, num_kernel=num_gaussian)

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if activation == "shifted_softplus":
            self.activation = functional.shifted_softplus
        elif isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.rbf_layer = nn.Linear(num_gaussian, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, hidden_dim)
        else:
            self.edge_linear = None

    def message(self, graph, input):
        node_in, node_out = graph.edge_list.t()[:2]
        position = graph.node_position
        message = self.input_layer(input)[node_in]
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feature.float())
        weight = self.rbf_layer(self.rbf(position[node_in], position[node_out]))
        message *= weight
        return message

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        return update

    def message_and_aggregate(self, graph, input):
        node_in, node_out = graph.edge_list.t()[:2]
        position = graph.node_position
        rbf_weight = self.rbf_layer(self.rbf(position[node_in], position[node_out]))
        indices = torch.stack([node_out, node_in, torch.arange(graph.num_edge, device=graph.device)])
        adjacency = utils.sparse_coo_tensor(indices, graph.edge_weight, (graph.num_node, graph.num_node, graph.num_edge))
        update = functional.generalized_rspmm(adjacency, rbf_weight, self.input_layer(input))
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_input = self.edge_linear(edge_input)
            edge_weight = graph.edge_weight.unsqueeze(-1) * rbf_weight
            edge_update = scatter_add(edge_input * edge_weight, graph.edge_list[:, 1], dim=0,
                                      dim_size=graph.num_node)
            update += edge_update

        return update

    def combine(self, input, update):
        output = self.output_layer(update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class MessagePassing(MessagePassingBase):
    """
    Message passing operator from `Neural Message Passing for Quantum Chemistry`_.

    This implements the edge network variant in the original paper.

    .. _Neural Message Passing for Quantum Chemistry:
        https://arxiv.org/pdf/1704.01212.pdf

    Parameters:
        input_dim (int): input dimension
        edge_input_dim (int): dimension of edge features
        hidden_dims (list of int, optional): hidden dims of edge network
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, edge_input_dim, hidden_dims=None, batch_norm=False, activation="relu"):
        super(MessagePassing, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.edge_input_dim = edge_input_dim
        if hidden_dims is None:
            hidden_dims = []

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(input_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.edge_mlp = layers.MLP(edge_input_dim, list(hidden_dims) + [input_dim * input_dim], activation)

    def message(self, graph, input):
        node_in = graph.edge_list[:, 0]
        transform = self.edge_mlp(graph.edge_feature.float()).view(-1, self.input_dim, self.input_dim)
        if graph.num_edge:
            message = torch.einsum("bed, bd -> be", transform, input[node_in])
        else:
            message = torch.zeros(0, self.input_dim, device=graph.device)
        return message

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        return update

    def combine(self, input, update):
        output = update
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class ChebyshevConv(MessagePassingBase):
    """
    Chebyshev spectral graph convolution operator from
    `Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering`_.

    .. _Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering:
        https://arxiv.org/pdf/1606.09375.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        k (int, optional): number of Chebyshev polynomials.
            This also corresponds to the radius of the receptive field.
        hidden_dims (list of int, optional): hidden dims of edge network
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, output_dim, edge_input_dim=None, k=1, batch_norm=False, activation="relu"):
        super(ChebyshevConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k
        self.edge_input_dim = edge_input_dim

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.linear = nn.Linear((k + 1) * input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def message(self, graph, input):
        node_in = graph.edge_list[:, 0]
        degree_in = graph.degree_in.unsqueeze(-1)
        # because self-loop messages have a different scale, they are processed in combine()
        message = input[node_in]
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feature.float())
        message /= (degree_in[node_in].sqrt() + 1e-10)
        return message

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        degree_out = graph.degree_out.unsqueeze(-1)
        # because self-loop messages have a different scale, they are processed in combine()
        update = -scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        update = update / (degree_out.sqrt() + 1e-10)
        return update

    def message_and_aggregate(self, graph, input):
        node_in, node_out = graph.edge_list.t()[:2]
        edge_weight = -graph.edge_weight / ((graph.degree_in[node_in] * graph.degree_out[node_out]).sqrt() + 1e-10)
        adjacency = utils.sparse_coo_tensor(graph.edge_list.t()[:2], edge_weight, (graph.num_node, graph.num_node))
        update = torch.sparse.mm(adjacency.t(), input)
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_input = self.edge_linear(edge_input)
            edge_weight = edge_weight.unsqueeze(-1)
            edge_update = scatter_add(edge_input * edge_weight, graph.edge_list[:, 1], dim=0,
                                      dim_size=graph.num_node)
            update += edge_update

        return update

    def forward(self, graph, input):
        # Chebyshev polynomial bases
        bases = [input]
        for i in range(self.k):
            x = super(ChebyshevConv, self).forward(graph, bases[-1])
            if i > 0:
                x = 2 * x - bases[-2]
            bases.append(x)
        bases = torch.cat(bases, dim=-1)

        output = self.linear(bases)
        if self.batch_norm:
            x = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

    def combine(self, input, update):
        output = input + update
        return output


class GeometricRelationalGraphConv(RelationalGraphConv):
    """
    Geometry-aware relational graph convolution operator from
    `Protein Representation Learning by Geometric Structure Pretraining`_.

    .. _Protein Representation Learning by Geometric Structure Pretraining:
        https://arxiv.org/pdf/2203.06125.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None, batch_norm=False, activation="relu"):
        super(GeometricRelationalGraphConv, self).__init__(input_dim, output_dim, num_relation, edge_input_dim,
                                                           batch_norm, activation)

    def aggregate(self, graph, message):
        assert graph.num_relation == self.num_relation

        node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node * self.num_relation)
        update = update.view(graph.num_node, self.num_relation * self.input_dim)

        return update

    def message_and_aggregate(self, graph, input):
        assert graph.num_relation == self.num_relation

        node_in, node_out, relation = graph.edge_list.t()
        node_out = node_out * self.num_relation + relation
        adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out]), graph.edge_weight,
                                            (graph.num_node, graph.num_node * graph.num_relation))
        update = torch.sparse.mm(adjacency.t(), input)
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_input = self.edge_linear(edge_input)
            edge_weight = graph.edge_weight.unsqueeze(-1)
            edge_update = scatter_add(edge_input * edge_weight, node_out, dim=0,
                                      dim_size=graph.num_node * graph.num_relation)
            update += edge_update

        return update.view(graph.num_node, self.num_relation * self.input_dim)