import math
import warnings
from functools import reduce
from collections import defaultdict

import networkx as nx

from matplotlib import pyplot as plt
import torch
from torch_scatter import scatter_add, scatter_min

from torchdrug import core, utils
from torchdrug.data import Dictionary
from torchdrug.utils import pretty

plt.switch_backend("agg")


class Graph(core._MetaContainer):
    r"""
    Basic container for sparse graphs.

    To batch graphs with variadic sizes, use :meth:`data.Graph.pack <torchdrug.data.Graph.pack>`.
    This will return a PackedGraph object with the following block diagonal adjacency matrix.

    .. math::

        \begin{bmatrix}
            A_1    & \cdots & 0      \\
            \vdots & \ddots & \vdots \\
            0      & \cdots & A_n
        \end{bmatrix}

    where :math:`A_i` is the adjacency of :math:`i`-th graph.

    You may register dynamic attributes for each graph.
    The registered attributes will be automatically processed during packing.

    .. warning::

        This class doesn't enforce any order on the edges.

    Example::

        >>> graph = data.Graph(torch.randint(10, (30, 2)))
        >>> with graph.node():
        >>>     graph.my_node_attr = torch.rand(10, 5, 5)

    Parameters:
        edge_list (array_like, optional): list of edges of shape :math:`(|E|, 2)` or :math:`(|E|, 3)`.
            Each tuple is (node_in, node_out) or (node_in, node_out, relation).
        edge_weight (array_like, optional): edge weights of shape :math:`(|E|,)`
        num_node (int, optional): number of nodes.
            By default, it will be inferred from the largest id in `edge_list`
        num_relation (int, optional): number of relations
        node_feature (array_like, optional): node features of shape :math:`(|V|, ...)`
        edge_feature (array_like, optional): edge features of shape :math:`(|E|, ...)`
        graph_feature (array_like, optional): graph feature of any shape
    """

    _meta_types = {"node", "edge", "graph", "node reference", "edge reference", "graph reference"}

    def __init__(self, edge_list=None, edge_weight=None, num_node=None, num_relation=None,
                 node_feature=None, edge_feature=None, graph_feature=None, **kwargs):
        super(Graph, self).__init__(**kwargs)
        # edge_list: N * [h, t] or N * [h, t, r]
        edge_list, num_edge = self._standarize_edge_list(edge_list, num_relation)
        edge_weight = self._standarize_edge_weight(edge_weight, edge_list)

        num_node = self._standarize_num_node(num_node, edge_list)
        num_relation = self._standarize_num_relation(num_relation, edge_list)

        self._edge_list = edge_list
        self._edge_weight = edge_weight
        self.num_node = num_node
        self.num_edge = num_edge
        self.num_relation = num_relation

        if node_feature is not None:
            with self.node():
                self.node_feature = torch.as_tensor(node_feature, device=self.device)
        if edge_feature is not None:
            with self.edge():
                self.edge_feature = torch.as_tensor(edge_feature, device=self.device)
        if graph_feature is not None:
            with self.graph():
                self.graph_feature = torch.as_tensor(graph_feature, device=self.device)

    def node(self):
        """
        Context manager for node attributes.
        """
        return self.context("node")

    def edge(self):
        """
        Context manager for edge attributes.
        """
        return self.context("edge")

    def graph(self):
        """
        Context manager for graph attributes.
        """
        return self.context("graph")

    def node_reference(self):
        """
        Context manager for node references.
        """
        return self.context("node reference")

    def edge_reference(self):
        """
        Context manager for edge references.
        """
        return self.context("edge reference")

    def graph_reference(self):
        """
        Context manager for graph references.
        """
        return self.context("graph reference")

    def _check_attribute(self, key, value):
        for type in self._meta_contexts:
            if "reference" in type:
                if value.dtype != torch.long:
                    raise TypeError("Tensors used as reference must be long tensors")
            if type == "node":
                if len(value) != self.num_node:
                    raise ValueError("Expect node attribute `%s` to have shape (%d, *), but found %s" %
                                     (key, self.num_node, value.shape))
            elif type == "edge":
                if len(value) != self.num_edge:
                    raise ValueError("Expect edge attribute `%s` to have shape (%d, *), but found %s" %
                                     (key, self.num_edge, value.shape))
            elif type == "node reference":
                is_valid = (value >= -1) & (value < self.num_node)
                if not is_valid.all():
                    error_value = value[~is_valid]
                    raise ValueError("Expect node reference in [-1, %d), but found %d" %
                                     (self.num_node, error_value[0]))
            elif type == "edge reference":
                is_valid = (value >= -1) & (value < self.num_edge)
                if not is_valid.all():
                    error_value = value[~is_valid]
                    raise ValueError("Expect edge reference in [-1, %d), but found %d" %
                                     (self.num_edge, error_value[0]))
            elif type == "graph reference":
                is_valid = (value >= -1) & (value < self.batch_size)
                if not is_valid.all():
                    error_value = value[~is_valid]
                    raise ValueError("Expect graph reference in [-1, %d), but found %d" %
                                     (self.batch_size, error_value[0]))

    def __setattr__(self, key, value):
        if hasattr(self, "meta_dict"):
            self._check_attribute(key, value)
        super(Graph, self).__setattr__(key, value)

    def _standarize_edge_list(self, edge_list, num_relation):
        if edge_list is not None and len(edge_list):
            if isinstance(edge_list, torch.Tensor) and edge_list.dtype != torch.long:
                try:
                    edge_list = torch.LongTensor(edge_list)
                except TypeError:
                    raise TypeError("Can't convert `edge_list` to torch.long")
            else:
                edge_list = torch.as_tensor(edge_list, dtype=torch.long)
        else:
            num_element = 2 if num_relation is None else 3
            if isinstance(edge_list, torch.Tensor):
                device = edge_list.device
            else:
                device = "cpu"
            edge_list = torch.zeros(0, num_element, dtype=torch.long, device=device)
        if (edge_list < 0).any():
            raise ValueError("`edge_list` should only contain non-negative indexes")
        num_edge = torch.tensor(len(edge_list), device=edge_list.device)
        return edge_list, num_edge

    def _standarize_edge_weight(self, edge_weight, edge_list):
        if edge_weight is not None:
            edge_weight = torch.as_tensor(edge_weight, dtype=torch.float, device=edge_list.device)
            if len(edge_list) != len(edge_weight):
                raise ValueError("`edge_list` and `edge_weight` should be the same size, but found %d and %d"
                                 % (len(edge_list), len(edge_weight)))
        else:
            edge_weight = torch.ones(len(edge_list), device=edge_list.device)
        return edge_weight

    def _standarize_num_node(self, num_node, edge_list):
        if num_node is None:
            num_node = self._maybe_num_node(edge_list)
        num_node = torch.as_tensor(num_node, device=edge_list.device)
        if (edge_list[:, :2] >= num_node).any():
            raise ValueError("`num_node` is %d, but found node %d in `edge_list`" % (num_node, edge_list[:, :2].max()))
        return num_node

    def _standarize_num_relation(self, num_relation, edge_list):
        if num_relation is None and edge_list.shape[1] > 2:
            num_relation = self._maybe_num_relation(edge_list)
        if num_relation is not None:
            num_relation = torch.as_tensor(num_relation, device=edge_list.device)
            if edge_list.shape[1] <= 2:
                raise ValueError("`num_relation` is provided, but the number of dims of `edge_list` is less than 3.")
            elif (edge_list[:, 2] >= num_relation).any():
                raise ValueError("`num_relation` is %d, but found relation %d in `edge_list`" % (num_relation, edge_list[:, 2].max()))
        return num_relation

    def _maybe_num_node(self, edge_list):
        warnings.warn("_maybe_num_node() is used to determine the number of nodes. "
                      "This may underestimate the count if there are isolated nodes.")
        if len(edge_list):
            return edge_list[:, :2].max().item() + 1
        else:
            return 0

    def _maybe_num_relation(self, edge_list):
        warnings.warn("_maybe_num_relation() is used to determine the number of relations. "
                      "This may underestimate the count if there are unseen relations.")
        return edge_list[:, 2].max().item() + 1

    def _standarize_index(self, index, count):
        if isinstance(index, slice):
            start = index.start or 0
            if start < 0:
                start += count
            stop = index.stop or count
            if stop < 0:
                stop += count
            step = index.step or 1
            index = torch.arange(start, stop, step, device=self.device)
        else:
            index = torch.as_tensor(index, device=self.device)
            if index.ndim == 0:
                index = index.unsqueeze(0)
            if index.dtype == torch.bool:
                if index.shape != (count,):
                    raise IndexError("Invalid mask. Expect mask to have shape %s, but found %s" %
                                     ((int(count),), tuple(index.shape)))
                index = index.nonzero().squeeze(-1)
            else:
                index = index.long()
                max_index = -1 if len(index) == 0 else index.max().item()
                if max_index >= count:
                    raise IndexError("Invalid index. Expect index smaller than %d, but found %d" % (count, max_index))
        return index

    def _get_mapping(self, index, count):
        index = self._standarize_index(index, count)
        if (index.bincount() > 1).any():
            raise ValueError("Can't create mapping for duplicate index")
        mapping = -torch.ones(count + 1, dtype=torch.long, device=self.device)
        mapping[index] = torch.arange(len(index), device=self.device)
        return mapping

    def _get_repeat_pack_offsets(self, num_xs, repeats):
        new_num_xs = num_xs.repeat_interleave(repeats)
        cum_repeats_shifted = repeats.cumsum(0) - repeats
        new_num_xs[cum_repeats_shifted] -= num_xs
        offsets = new_num_xs.cumsum(0)
        return offsets

    @classmethod
    def from_dense(cls, adjacency, node_feature=None, edge_feature=None):
        """
        Create a sparse graph from a dense adjacency matrix.
        For zero entries in the adjacency matrix, their edge features will be ignored.

        Parameters:
            adjacency (array_like): adjacency matrix of shape :math:`(|V|, |V|)` or :math:`(|V|, |V|, |R|)`
            node_feature (array_like): node features of shape :math:`(|V|, ...)`
            edge_feature (array_like): edge features of shape :math:`(|V|, |V|, ...)` or :math:`(|V|, |V|, |R|, ...)`
        """
        adjacency = torch.as_tensor(adjacency)
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("`adjacency` should be a square matrix, but found %d and %d" % adjacency.shape[:2])

        edge_list = adjacency.nonzero()
        edge_weight = adjacency[tuple(edge_list.t())]
        num_node = adjacency.shape[0]
        num_relation = adjacency.shape[2] if adjacency.ndim > 2 else None
        if edge_feature is not None:
            edge_feature = torch.as_tensor(edge_feature)
            edge_feature = edge_feature[tuple(edge_list.t())]

        return cls(edge_list, edge_weight, num_node, num_relation, node_feature, edge_feature)

    def connected_components(self):
        """
        Split this graph into connected components.

        Returns:
            (PackedGraph, LongTensor): connected components, number of connected components per graph
        """
        node_in, node_out = self.edge_list.t()[:2]
        range = torch.arange(self.num_node, device=self.device)
        node_in, node_out = torch.cat([node_in, node_out, range]), torch.cat([node_out, node_in, range])

        # find connected component
        # O(|E|d), d is the diameter of the graph
        min_neighbor = torch.arange(self.num_node, device=self.device)
        last = torch.zeros_like(min_neighbor)
        while not torch.equal(min_neighbor, last):
            last = min_neighbor
            min_neighbor = scatter_min(min_neighbor[node_out], node_in, dim_size=self.num_node)[0]
        anchor = torch.unique(min_neighbor)
        num_cc = self.node2graph[anchor].bincount(minlength=self.batch_size)
        return self.split(min_neighbor), num_cc

    def split(self, node2graph):
        """
        Split a graph into multiple disconnected graphs.

        Parameters:
            node2graph (array_like): ID of the graph each node belongs to

        Returns:
            PackedGraph
        """
        node2graph = torch.as_tensor(node2graph, dtype=torch.long, device=self.device)
        # coalesce arbitrary graph IDs to [0, n)
        _, node2graph = torch.unique(node2graph, return_inverse=True)
        num_graph = node2graph.max() + 1
        index = node2graph.argsort()
        mapping = torch.zeros_like(index)
        mapping[index] = torch.arange(len(index), device=self.device)

        node_in, node_out = self.edge_list.t()[:2]
        edge_mask = node2graph[node_in] == node2graph[node_out]
        edge2graph = node2graph[node_in]
        edge_index = edge2graph.argsort()
        edge_index = edge_index[edge_mask[edge_index]]

        prepend = torch.tensor([-1], device=self.device)
        is_first_node = torch.diff(node2graph[index], prepend=prepend) > 0
        graph_index = self.node2graph[index[is_first_node]]

        edge_list = self.edge_list.clone()
        edge_list[:, :2] = mapping[edge_list[:, :2]]

        num_nodes = node2graph.bincount(minlength=num_graph)
        num_edges = edge2graph[edge_index].bincount(minlength=num_graph)

        num_cum_nodes = num_nodes.cumsum(0)
        offsets = (num_cum_nodes - num_nodes)[edge2graph[edge_index]]

        data_dict, meta_dict = self.data_mask(index, edge_index, graph_index=graph_index, exclude="graph reference")

        return self.packed_type(edge_list[edge_index], edge_weight=self.edge_weight[edge_index], num_nodes=num_nodes,
                                num_edges=num_edges, num_relation=self.num_relation, offsets=offsets,
                                meta_dict=meta_dict, **data_dict)

    @classmethod
    def pack(cls, graphs):
        """
        Pack a list of graphs into a PackedGraph object.

        Parameters:
            graphs (list of Graph): list of graphs

        Returns:
            PackedGraph
        """
        edge_list = []
        edge_weight = []
        num_nodes = []
        num_edges = []
        num_relation = -1
        num_cum_node = 0
        num_cum_edge = 0
        num_graph = 0
        data_dict = defaultdict(list)
        meta_dict = graphs[0].meta_dict
        for graph in graphs:
            edge_list.append(graph.edge_list)
            edge_weight.append(graph.edge_weight)
            num_nodes.append(graph.num_node)
            num_edges.append(graph.num_edge)
            for k, v in graph.data_dict.items():
                for type in meta_dict[k]:
                    if type == "graph":
                        v = v.unsqueeze(0)
                    elif type == "node reference":
                        v = v + num_cum_node
                    elif type == "edge reference":
                        v = v + num_cum_edge
                    elif type == "graph reference":
                        v = v + num_graph
                data_dict[k].append(v)
            if num_relation == -1:
                num_relation = graph.num_relation
            elif num_relation != graph.num_relation:
                raise ValueError("Inconsistent `num_relation` in graphs. Expect %d but got %d."
                                 % (num_relation, graph.num_relation))
            num_cum_node += graph.num_node
            num_cum_edge += graph.num_edge
            num_graph += 1

        edge_list = torch.cat(edge_list)
        edge_weight = torch.cat(edge_weight)
        data_dict = {k: torch.cat(v) for k, v in data_dict.items()}

        return cls.packed_type(edge_list, edge_weight=edge_weight, num_nodes=num_nodes, num_edges=num_edges,
                               num_relation=num_relation, meta_dict=meta_dict, **data_dict)

    def repeat(self, count):
        """
        Repeat this graph.

        Parameters:
            count (int): number of repetitions

        Returns:
            PackedGraph
        """
        edge_list = self.edge_list.repeat(count, 1)
        edge_weight = self.edge_weight.repeat(count)
        num_nodes = [self.num_node] * count
        num_edges = [self.num_edge] * count
        num_relation = self.num_relation

        data_dict = {}
        for k, v in self.data_dict.items():
            if "graph" in self.meta_dict[k]:
                v = v.unsqueeze(0)
            shape = [1] * v.ndim
            shape[0] = count
            length = len(v)
            v = v.repeat(shape)
            for type in self.meta_dict[k]:
                if type == "node reference":
                    offsets = torch.arange(count, device=self.device) * self.num_node
                    v = v + offsets.repeat_interleave(length)
                elif type == "edge reference":
                    offsets = torch.arange(count, device=self.device) * self.num_edge
                    v = v + offsets.repeat_interleave(length)
                elif type == "graph reference":
                    offsets = torch.arange(count, device=self.device)
                    v = v + offsets.repeat_interleave(length)
            data_dict[k] = v

        return self.packed_type(edge_list, edge_weight=edge_weight, num_nodes=num_nodes, num_edges=num_edges,
                                num_relation=num_relation, meta_dict=self.meta_dict, **data_dict)

    def get_edge(self, edge):
        """
        Get the weight of of an edge.

        Parameters:
            edge (array_like): index of shape :math:`(2,)` or :math:`(3,)`

        Returns:
            Tensor: weight of the edge
        """
        if len(edge) != self.edge_list.shape[1]:
            raise ValueError("Incorrect edge index. Expect %d axes but got %d axes"
                             % (self.edge_list.shape[1], len(edge)))

        edge_index, num_match = self.match(edge)
        return self.edge_weight[edge_index].sum()

    def match(self, pattern):
        """
        Return all matched indexes for each pattern. Support patterns with ``-1`` as the wildcard.

        Parameters:
            pattern (array_like): index of shape :math:`(N, 2)` or :math:`(N, 3)`

        Returns:
            (LongTensor, LongTensor): matched indexes, number of matches per edge

        Examples::

            >>> graph = data.Graph([[0, 1], [1, 0], [1, 2], [2, 1], [2, 0], [0, 2]])
            >>> index, num_match = graph.match([[0, -1], [1, 2]])
            >>> assert (index == torch.tensor([0, 5, 2])).all()
            >>> assert (num_match == torch.tensor([2, 1])).all()

        """
        if len(pattern) == 0:
            index = num_match = torch.zeros(0, dtype=torch.long, device=self.device)
            return index, num_match

        if not hasattr(self, "edge_inverted_index"):
            self.edge_inverted_index = {}
        pattern = torch.as_tensor(pattern, dtype=torch.long, device=self.device)
        if pattern.ndim == 1:
            pattern = pattern.unsqueeze(0)
        mask = pattern != -1
        scale = 2 ** torch.arange(pattern.shape[-1], device=self.device)
        query_type = (mask * scale).sum(dim=-1)
        query_index = query_type.argsort()
        num_query = query_type.unique(return_counts=True)[1]
        query_ends = num_query.cumsum(0)
        query_starts = query_ends - num_query
        mask_set = mask[query_index[query_starts]].tolist()

        type_ranges = []
        type_orders = []
        # get matched range for each query type
        for i, mask in enumerate(mask_set):
            query_type = tuple(mask)
            type_index = query_index[query_starts[i]: query_ends[i]]
            type_edge = pattern[type_index][:, mask]
            if query_type not in self.edge_inverted_index:
                self.edge_inverted_index[query_type] = self._build_edge_inverted_index(mask)
            inverted_range, order = self.edge_inverted_index[query_type]
            ranges = inverted_range.get(type_edge, default=0)
            type_ranges.append(ranges)
            type_orders.append(order)
        ranges = torch.cat(type_ranges)
        orders = torch.stack(type_orders)
        types = torch.arange(len(mask_set), device=self.device)
        types = types.repeat_interleave(num_query)

        # reorder matched ranges according to the query order
        ranges = scatter_add(ranges, query_index, dim=0, dim_size=len(pattern))
        types = scatter_add(types, query_index, dim_size=len(pattern))
        # convert range to indexes
        starts, ends = ranges.t()
        num_match = ends - starts
        offsets = num_match.cumsum(0) - num_match
        types = types.repeat_interleave(num_match)
        ranges = torch.arange(num_match.sum(), device=self.device)
        ranges = ranges + (starts - offsets).repeat_interleave(num_match)
        index = orders[types, ranges]

        return index, num_match

    def _build_edge_inverted_index(self, mask):
        keys = self.edge_list[:, mask]
        base = torch.tensor(self.shape, device=self.device)
        base = base[mask]
        max = reduce(int.__mul__, base.tolist())
        if max > torch.iinfo(torch.int64).max:
            raise ValueError("Fail to build an inverted index table based on sorting. "
                             "The graph is too large.")
        scale = base.cumprod(0)
        scale = scale[-1] // scale
        key = (keys * scale).sum(dim=-1)
        order = key.argsort()
        num_keys = key.unique(return_counts=True)[1]
        ends = num_keys.cumsum(0)
        starts = ends - num_keys
        ranges = torch.stack([starts, ends], dim=-1)
        keys_set = keys[order[starts]]
        inverted_range = Dictionary(keys_set, ranges)
        return inverted_range, order

    def __getitem__(self, index):
        # why do we check tuple?
        # case 1: x[0, 1] is parsed as (0, 1)
        # case 2: x[[0, 1]] is parsed as [0, 1]
        if not isinstance(index, tuple):
            index = (index,)
        index = list(index)

        while len(index) < 2:
            index.append(slice(None))
        if len(index) > 2:
            raise ValueError("Graph has only 2 axis, but %d axis is indexed" % len(index))

        if all([isinstance(axis_index, int) for axis_index in index]):
            return self.get_edge(index)

        edge_list = self.edge_list.clone()
        for i, axis_index in enumerate(index):
            axis_index = self._standarize_index(axis_index, self.num_node)
            mapping = -torch.ones(self.num_node, dtype=torch.long, device=self.device)
            mapping[axis_index] = axis_index
            edge_list[:, i] = mapping[edge_list[:, i]]
        edge_index = (edge_list >= 0).all(dim=-1)

        return self.edge_mask(edge_index)

    def __len__(self):
        return 1

    @property
    def batch_size(self):
        """Batch size."""
        return 1

    def subgraph(self, index):
        """
        Return a subgraph based on the specified nodes.
        Equivalent to :meth:`node_mask(index, compact=True) <node_mask>`.

        Parameters:
            index (array_like): node index

        Returns:
            Graph

        See also:
            :meth:`Graph.node_mask`
        """
        return self.node_mask(index, compact=True)

    def data_mask(self, node_index=None, edge_index=None, graph_index=None, include=None, exclude=None):
        data_dict, meta_dict = self.data_by_meta(include, exclude)
        node_mapping = None
        edge_mapping = None
        graph_mapping = None
        for k, v in data_dict.items():
            for type in meta_dict[k]:
                if type == "node" and node_index is not None:
                    v = v[node_index]
                elif type == "edge" and edge_index is not None:
                    v = v[edge_index]
                elif type == "graph" and graph_index is not None:
                    v = v.unsqueeze(0)[graph_index]
                elif type == "node reference" and node_index is not None:
                    if node_mapping is None:
                        node_mapping = self._get_mapping(node_index, self.num_node)
                    v = node_mapping[v]
                elif type == "edge reference" and edge_index is not None:
                    if edge_mapping is None:
                        edge_mapping = self._get_mapping(edge_index, self.num_edge)
                    v = edge_mapping[v]
                elif type == "graph reference" and graph_index is not None:
                    if graph_mapping is None:
                        graph_mapping = self._get_mapping(graph_index, self.batch_size)
                    v = graph_mapping[v]
            data_dict[k] = v

        return data_dict, meta_dict

    def node_mask(self, index, compact=False):
        """
        Return a masked graph based on the specified nodes.

        This function can also be used to re-order the nodes.

        Parameters:
            index (array_like): node index
            compact (bool, optional): compact node ids or not

        Returns:
            Graph

        Examples::

            >>> graph = data.Graph.from_dense(torch.eye(3))
            >>> assert graph.node_mask([1, 2]).adjacency.shape == (3, 3)
            >>> assert graph.node_mask([1, 2], compact=True).adjacency.shape == (2, 2)

        """
        index = self._standarize_index(index, self.num_node)
        mapping = -torch.ones(self.num_node, dtype=torch.long, device=self.device)
        if compact:
            mapping[index] = torch.arange(len(index), device=self.device)
            num_node = len(index)
        else:
            mapping[index] = index
            num_node = self.num_node

        edge_list = self.edge_list.clone()
        edge_list[:, :2] = mapping[edge_list[:, :2]]
        edge_index = (edge_list[:, :2] >= 0).all(dim=-1)

        if compact:
            data_dict, meta_dict = self.data_mask(index, edge_index)
        else:
            data_dict, meta_dict = self.data_mask(edge_index=edge_index)

        return type(self)(edge_list[edge_index], edge_weight=self.edge_weight[edge_index], num_node=num_node,
                          num_relation=self.num_relation, meta_dict=meta_dict, **data_dict)

    def compact(self):
        """
        Remove isolated nodes and compact node ids.

        Returns:
            Graph
        """
        index = self.degree_out + self.degree_in > 0
        return self.subgraph(index)

    def edge_mask(self, index):
        """
        Return a masked graph based on the specified edges.

        This function can also be used to re-order the edges.

        Parameters:
            index (array_like): edge index

        Returns:
            Graph
        """
        index = self._standarize_index(index, self.num_edge)
        data_dict, meta_dict = self.data_mask(edge_index=index)

        return type(self)(self.edge_list[index], edge_weight=self.edge_weight[index], num_node=self.num_node,
                          num_relation=self.num_relation, meta_dict=meta_dict, **data_dict)

    def line_graph(self):
        """
        Construct a line graph of this graph.
        The node feature of the line graph is inherited from the edge feature of the original graph.

        In the line graph, each node corresponds to an edge in the original graph.
        For a pair of edges (a, b) and (b, c) that share the same intermediate node in the original graph,
        there is a directed edge (a, b) -> (b, c) in the line graph.

        Returns:
            Graph
        """
        node_in, node_out = self.edge_list.t()[:2]
        edge_index = torch.arange(self.num_edge, device=self.device)
        edge_in = edge_index[node_out.argsort()]
        edge_out = edge_index[node_in.argsort()]

        degree_in = node_in.bincount(minlength=self.num_node)
        degree_out = node_out.bincount(minlength=self.num_node)
        size = degree_out * degree_in
        starts = (size.cumsum(0) - size).repeat_interleave(size)
        range = torch.arange(size.sum(), device=self.device)
        # each node u has degree_out[u] * degree_in[u] local edges
        local_index = range - starts
        local_inner_size = degree_in.repeat_interleave(size)
        edge_in_offset = (degree_out.cumsum(0) - degree_out).repeat_interleave(size)
        edge_out_offset = (degree_in.cumsum(0) - degree_in).repeat_interleave(size)
        edge_in_index = local_index // local_inner_size + edge_in_offset
        edge_out_index = local_index % local_inner_size + edge_out_offset

        edge_in = edge_in[edge_in_index]
        edge_out = edge_out[edge_out_index]
        edge_list = torch.stack([edge_in, edge_out], dim=-1)
        node_feature = getattr(self, "edge_feature", None)
        num_node = self.num_edge
        num_edge = size.sum()

        return Graph(edge_list, num_node=num_node, num_edge=num_edge, node_feature=node_feature)

    def full(self):
        """
        Return a fully connected graph over the nodes.

        Returns:
            Graph
        """
        index = torch.arange(self.num_node, device=self.device)
        if self.num_relation:
            edge_list = torch.meshgrid(index, index, torch.arange(self.num_relation, device=self.device))
        else:
            edge_list = torch.meshgrid(index, index)
        edge_list = torch.stack(edge_list).flatten(1)
        edge_weight = torch.ones(len(edge_list))

        data_dict, meta_dict = self.data_by_meta(exclude="edge")

        return type(self)(edge_list, edge_weight=edge_weight, num_node=self.num_node, num_relation=self.num_relation,
                          meta_dict=meta_dict, **data_dict)

    def directed(self, order=None):
        """
        Mask the edges to create a directed graph.
        Edges that go from a node index to a larger or equal node index will be kept.

        Parameters:
            order (Tensor, optional): topological order of the nodes
        """
        node_in, node_out = self.edge_list.t()[:2]
        if order is not None:
            edge_index = order[node_in] <= order[node_out]
        else:
            edge_index = node_in <= node_out

        return self.edge_mask(edge_index)

    def undirected(self, add_inverse=False):
        """
        Flip all the edges to create an undirected graph.

        For knowledge graphs, the flipped edges can either have the original relation or an inverse relation.
        The inverse relation for relation :math:`r` is defined as :math:`|R| + r`.

        Parameters:
            add_inverse (bool, optional): whether to use inverse relations for flipped edges
        """
        edge_list = self.edge_list.clone()
        edge_list[:, :2] = edge_list[:, :2].flip(1)
        num_relation = self.num_relation
        if num_relation and add_inverse:
            edge_list[:, 2] += num_relation
            num_relation = num_relation * 2
        edge_list = torch.stack([self.edge_list, edge_list], dim=1).flatten(0, 1)

        index = torch.arange(self.num_edge, device=self.device).unsqueeze(-1).expand(-1, 2).flatten()
        data_dict, meta_dict = self.data_mask(edge_index=index)

        return type(self)(edge_list, edge_weight=self.edge_weight[index], num_node=self.num_node,
                          num_relation=num_relation, meta_dict=meta_dict, **data_dict)

    @utils.cached_property
    def adjacency(self):
        """
        Adjacency matrix of this graph.

        If :attr:`num_relation` is specified, a sparse tensor of shape :math:`(|V|, |V|, num\_relation)` will be
        returned.
        Otherwise, a sparse tensor of shape :math:`(|V|, |V|)` will be returned.
        """
        return utils.sparse_coo_tensor(self.edge_list.t(), self.edge_weight, self.shape)

    _tensor_names = ["edge_list", "edge_weight", "num_node", "num_relation", "edge_feature"]

    def to_tensors(self):
        edge_feature = getattr(self, "edge_feature", torch.tensor(0, device=self.device))
        return self.edge_list, self.edge_weight, self.num_node, self.num_relation, edge_feature

    @classmethod
    def from_tensors(cls, tensors):
        edge_list, edge_weight, num_node, num_relation, edge_feature = tensors
        if edge_feature.ndim == 0:
            edge_feature = None
        return cls(edge_list, edge_weight, num_node, num_relation, edge_feature=edge_feature)

    @property
    def node2graph(self):
        """Node id to graph id mapping."""
        return torch.zeros(self.num_node, dtype=torch.long, device=self.device)

    @property
    def edge2graph(self):
        """Edge id to graph id mapping."""
        return torch.zeros(self.num_edge, dtype=torch.long, device=self.device)

    @utils.cached_property
    def degree_out(self):
        """
        Weighted number of edges containing each node as output.

        Note this is the **in-degree** in graph theory.
        """
        return scatter_add(self.edge_weight, self.edge_list[:, 1], dim_size=self.num_node)

    @utils.cached_property
    def degree_in(self):
        """
        Weighted number of edges containing each node as input.

        Note this is the **out-degree** in graph theory.
        """
        return scatter_add(self.edge_weight, self.edge_list[:, 0], dim_size=self.num_node)

    @property
    def edge_list(self):
        """List of edges."""
        return self._edge_list

    @property
    def edge_weight(self):
        """Edge weights."""
        return self._edge_weight

    @property
    def device(self):
        """Device."""
        return self.edge_list.device

    @property
    def requires_grad(self):
        return self.edge_weight.requires_grad

    @property
    def grad(self):
        return self.edge_weight.grad

    @property
    def data(self):
        return self

    def requires_grad_(self):
        self.edge_weight.requires_grad_()
        return self

    def size(self, dim=None):
        if self.num_relation:
            size = torch.Size((self.num_node, self.num_node, self.num_relation))
        else:
            size = torch.Size((self.num_node, self.num_node))
        if dim is None:
            return size
        return size[dim]

    @property
    def shape(self):
        return self.size()

    def copy_(self, src):
        """
        Copy data from ``src`` into ``self`` and return ``self``.

        The ``src`` graph must have the same set of attributes as ``self``.
        """
        self.edge_list.copy_(src.edge_list)
        self.edge_weight.copy_(src.edge_weight)
        self.num_node.copy_(src.num_node)
        self.num_edge.copy_(src.num_edge)
        if self.num_relation is not None:
            self.num_relation.copy_(src.num_relation)

        keys = set(self.data_dict.keys())
        src_keys = set(src.data_dict.keys())
        if keys != src_keys:
            raise RuntimeError("Attributes mismatch. Trying to assign attributes %s, "
                               "but current graph has attributes %s" % (src_keys, keys))
        for k, v in self.data_dict.items():
            v.copy_(src.data_dict[k])

        return self

    def detach(self):
        """
        Detach this graph.
        """
        return type(self)(self.edge_list.detach(), edge_weight=self.edge_weight.detach(),
                          num_node=self.num_node, num_relation=self.num_relation,
                          meta_dict=self.meta_dict, **utils.detach(self.data_dict))

    def clone(self):
        """
        Clone this graph.
        """
        return type(self)(self.edge_list.clone(), edge_weight=self.edge_weight.clone(),
                          num_node=self.num_node, num_relation=self.num_relation,
                          meta_dict=self.meta_dict, **utils.clone(self.data_dict))

    def cuda(self, *args, **kwargs):
        """
        Return a copy of this graph in CUDA memory.

        This is a non-op if the graph is already on the correct device.
        """
        edge_list = self.edge_list.cuda(*args, **kwargs)

        if edge_list is self.edge_list:
            return self
        else:
            return type(self)(edge_list, edge_weight=self.edge_weight,
                              num_node=self.num_node, num_relation=self.num_relation,
                              meta_dict=self.meta_dict, **utils.cuda(self.data_dict, *args, **kwargs))

    def cpu(self):
        """
        Return a copy of this graph in CPU memory.

        This is a non-op if the graph is already in CPU memory.
        """
        edge_list = self.edge_list.cpu()

        if edge_list is self.edge_list:
            return self
        else:
            return type(self)(edge_list, edge_weight=self.edge_weight, num_node=self.num_node,
                              num_relation=self.num_relation, meta_dict=self.meta_dict, **utils.cpu(self.data_dict))

    def to(self, device, *args, **kwargs):
        """
        Return a copy of this graph on the given device.
        """
        device = torch.device(device)
        if device.type == "cpu":
            return self.cpu(*args, **kwargs)
        else:
            return self.cuda(device, *args, **kwargs)

    def __repr__(self):
        fields = ["num_node=%d" % self.num_node, "num_edge=%d" % self.num_edge]
        if self.num_relation is not None:
            fields.append("num_relation=%d" % self.num_relation)
        if self.device.type != "cpu":
            fields.append("device='%s'" % self.device)
        return "%s(%s)" % (self.__class__.__name__, ", ".join(fields))

    def visualize(self, title=None, save_file=None, figure_size=(3, 3), ax=None, layout="spring"):
        """
        Visualize this graph with matplotlib.

        Parameters:
            title (str, optional): title for this graph
            save_file (str, optional): ``png`` or ``pdf`` file to save visualization.
                If not provided, show the figure in window.
            figure_size (tuple of int, optional): width and height of the figure
            ax (matplotlib.axes.Axes, optional): axis to plot the figure
            layout (str, optional): graph layout

        See also:
            `NetworkX graph layout`_

            .. _NetworkX graph layout:
                https://networkx.github.io/documentation/stable/reference/drawing.html#module-networkx.drawing.layout
        """
        is_root = ax is None
        if ax is None:
            fig = plt.figure(figsize=figure_size)
            if title is not None:
                ax = plt.gca()
            else:
                ax = fig.add_axes([0, 0, 1, 1])
        if title is not None:
            ax.set_title(title)

        edge_list = self.edge_list[:, :2].tolist()
        G = nx.DiGraph(edge_list)
        G.add_nodes_from(range(self.num_node))
        if hasattr(nx, "%s_layout" % layout):
            func = getattr(nx, "%s_layout" % layout)
        else:
            raise ValueError("Unknown networkx layout `%s`" % layout)
        if layout == "spring" or layout == "random":
            pos = func(G, seed=0)
        else:
            pos = func(G)
        nx.draw_networkx(G, pos, ax=ax)
        if self.num_relation:
            edge_labels = self.edge_list[:, 2].tolist()
            edge_labels = {tuple(e): l for e, l in zip(edge_list, edge_labels)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax)
        ax.set_frame_on(False)

        if is_root:
            if save_file:
                fig.savefig(save_file)
            else:
                fig.show()

    def __getstate__(self):
        state = {}
        cls = self.__class__
        for k, v in self.__dict__.items():
            # do not pickle property / cached property
            if hasattr(cls, k) and isinstance(getattr(cls, k), property):
                continue
            state[k] = v
        return state


class PackedGraph(Graph):
    """
    Container for sparse graphs with variadic sizes.

    To create a PackedGraph from Graph objects

        >>> batch = data.Graph.pack(graphs)

    To retrieve Graph objects from a PackedGraph

        >>> graphs = batch.unpack()

    .. warning::
        
        Edges of the same graph are guaranteed to be consecutive in the edge list.
        However, this class doesn't enforce any order on the edges.

    Parameters:
        edge_list (array_like, optional): list of edges of shape :math:`(|E|, 2)` or :math:`(|E|, 3)`.
            Each tuple is (node_in, node_out) or (node_in, node_out, relation).
        edge_weight (array_like, optional): edge weights of shape :math:`(|E|,)`
        num_nodes (array_like, optional): number of nodes in each graph
            By default, it will be inferred from the largest id in `edge_list`
        num_edges (array_like, optional): number of edges in each graph
        num_relation (int, optional): number of relations
        node_feature (array_like, optional): node features of shape :math:`(|V|, ...)`
        edge_feature (array_like, optional): edge features of shape :math:`(|E|, ...)`
        offsets (array_like, optional): node id offsets of shape :math:`(|E|,)`.
            If not provided, nodes in `edge_list` should be relative index, i.e., the index in each graph.
            If provided, nodes in `edge_list` should be absolute index, i.e., the index in the packed graph.
    """

    unpacked_type = Graph

    def __init__(self, edge_list=None, edge_weight=None, num_nodes=None, num_edges=None, num_relation=None,
                 offsets=None, **kwargs):
        edge_list, num_nodes, num_edges, num_cum_nodes, num_cum_edges, offsets = \
            self._get_cumulative(edge_list, num_nodes, num_edges, offsets)

        if offsets is None:
            offsets = self._get_offsets(num_nodes, num_edges, num_cum_nodes)
            edge_list = edge_list.clone()
            edge_list[:, :2] += offsets.unsqueeze(-1)

        num_node = num_nodes.sum()
        if (edge_list[:, :2] >= num_node).any():
            raise ValueError("Sum of `num_nodes` is %d, but found %d in `edge_list`" %
                             (num_node, edge_list[:, :2].max()))

        self._offsets = offsets
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_cum_nodes = num_cum_nodes
        self.num_cum_edges = num_cum_edges

        super(PackedGraph, self).__init__(edge_list, edge_weight=edge_weight, num_node=num_node,
                                          num_relation=num_relation, **kwargs)

    def _get_offsets(self, num_nodes=None, num_edges=None, num_cum_nodes=None, num_cum_edges=None):
        if num_nodes is None:
            prepend = torch.tensor([0], device=self.device)
            num_nodes = torch.diff(num_cum_nodes, prepend=prepend)
        if num_edges is None:
            prepend = torch.tensor([0], device=self.device)
            num_edges = torch.diff(num_cum_edges, prepend=prepend)
        if num_cum_nodes is None:
            num_cum_nodes = num_nodes.cumsum(0)
        return (num_cum_nodes - num_nodes).repeat_interleave(num_edges)

    def merge(self, graph2graph):
        """
        Merge multiple graphs into a single graph.

        Parameters:
            graph2graph (array_like): ID of the new graph each graph belongs to
        """
        graph2graph = torch.as_tensor(graph2graph, dtype=torch.long, device=self.device)
        # coalesce arbitrary graph IDs to [0, n)
        _, graph2graph = torch.unique(graph2graph, return_inverse=True)

        graph_key = graph2graph * self.batch_size + torch.arange(self.batch_size, device=self.device)
        graph_index = graph_key.argsort()
        graph = self.subbatch(graph_index)
        graph2graph = graph2graph[graph_index]

        num_graph = graph2graph[-1] + 1
        num_nodes = scatter_add(graph.num_nodes, graph2graph, dim_size=num_graph)
        num_edges = scatter_add(graph.num_edges, graph2graph, dim_size=num_graph)
        offsets = self._get_offsets(num_nodes, num_edges)

        data_dict, meta_dict = graph.data_mask(exclude="graph")

        return type(self)(graph.edge_list, edge_weight=graph.edge_weight, num_nodes=num_nodes,
                          num_edges=num_edges, num_relation=graph.num_relation, offsets=offsets,
                          meta_dict=meta_dict, **data_dict)

    def unpack(self):
        """
        Unpack this packed graph into a list of graphs.

        Returns:
            list of Graph
        """
        graphs = []
        for i in range(self.batch_size):
            graphs.append(self.get_item(i))
        return graphs

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index < self.batch_size:
            item = self[self._iter_index]
            self._iter_index += 1
            return item
        raise StopIteration

    def _check_attribute(self, key, value):
        for type in self._meta_contexts:
            if "reference" in type:
                if value.dtype != torch.long:
                    raise TypeError("Tensors used as reference must be long tensors")
            if type == "node":
                if len(value) != self.num_node:
                    raise ValueError("Expect node attribute `%s` to have shape (%d, *), but found %s" %
                                     (key, self.num_node, value.shape))
            elif type == "edge":
                if len(value) != self.num_edge:
                    raise ValueError("Expect edge attribute `%s` to have shape (%d, *), but found %s" %
                                     (key, self.num_edge, value.shape))
            elif type == "graph":
                if len(value) != self.batch_size:
                    raise ValueError("Expect graph attribute `%s` to have shape (%d, *), but found %s" %
                                     (key, self.batch_size, value.shape))
            elif type == "node reference":
                is_valid = (value >= -1) & (value < self.num_node)
                if not is_valid.all():
                    error_value = value[~is_valid]
                    raise ValueError("Expect node reference in [-1, %d), but found %d" %
                                     (self.num_node, error_value[0]))
            elif type == "edge reference":
                is_valid = (value >= -1) & (value < self.num_edge)
                if not is_valid.all():
                    error_value = value[~is_valid]
                    raise ValueError("Expect edge reference in [-1, %d), but found %d" %
                                     (self.num_edge, error_value[0]))
            elif type == "graph reference":
                is_valid = (value >= -1) & (value < self.batch_size)
                if not is_valid.all():
                    error_value = value[~is_valid]
                    raise ValueError("Expect graph reference in [-1, %d), but found %d" %
                                     (self.batch_size, error_value[0]))

    def unpack_data(self, data, type="auto"):
        """
        Unpack node or edge data according to the packed graph.

        Parameters:
            data (Tensor): data to unpack
            type (str, optional): data type. Can be ``auto``, ``node``, or ``edge``.

        Returns:
            list of Tensor
        """
        if type == "auto":
            if self.num_node == self.num_edge:
                raise ValueError("Ambiguous type. Please specify either `node` or `edge`")
            if len(data) == self.num_node:
                type = "node"
            elif len(data) == self.num_edge:
                type = "edge"
            else:
                raise ValueError("Graph has %d nodes and %d edges, but data has %d entries" %
                                 (self.num_node, self.num_edge, len(data)))
        data_list = []
        if type == "node":
            for i in range(self.batch_size):
                data_list.append(data[self.num_cum_nodes[i] - self.num_nodes[i]: self.num_cum_nodes[i]])
        elif type == "edge":
            for i in range(self.batch_size):
                data_list.append(data[self.num_cum_edges[i] - self.num_edges[i]: self.num_cum_edges[i]])

        return data_list

    def repeat(self, count):
        """
        Repeat this packed graph. This function behaves similarly to `torch.Tensor.repeat`_.

        .. _torch.Tensor.repeat:
            https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html

        Parameters:
            count (int): number of repetitions

        Returns:
            PackedGraph
        """
        num_nodes = self.num_nodes.repeat(count)
        num_edges = self.num_edges.repeat(count)
        offsets = self._get_offsets(num_nodes, num_edges)
        edge_list = self.edge_list.repeat(count, 1)
        edge_list[:, :2] += (offsets - self._offsets.repeat(count)).unsqueeze(-1)

        data_dict = {}
        for k, v in self.data_dict.items():
            shape = [1] * v.ndim
            shape[0] = count
            length = len(v)
            v = v.repeat(shape)
            for _type in self.meta_dict[k]:
                if _type == "node reference":
                    pack_offsets = torch.arange(count, device=self.device) * self.num_node
                    v = v + pack_offsets.repeat_interleave(length)
                elif _type == "edge reference":
                    pack_offsets = torch.arange(count, device=self.device) * self.num_edge
                    v = v + pack_offsets.repeat_interleave(length)
                elif _type == "graph reference":
                    pack_offsets = torch.arange(count, device=self.device) * self.batch_size
                    v = v + pack_offsets.repeat_interleave(length)
            data_dict[k] = v

        return type(self)(edge_list, edge_weight=self.edge_weight.repeat(count),
                          num_nodes=num_nodes, num_edges=num_edges, num_relation=self.num_relation,
                          offsets=offsets, meta_dict=self.meta_dict, **data_dict)

    def repeat_interleave(self, repeats):
        """
        Repeat this packed graph. This function behaves similarly to `torch.repeat_interleave`_.

        .. _torch.repeat_interleave:
            https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html

        Parameters:
            repeats (Tensor or int): number of repetitions for each graph

        Returns:
            PackedGraph
        """
        repeats = torch.as_tensor(repeats, dtype=torch.long, device=self.device)
        if repeats.numel() == 1:
            repeats = repeats * torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        num_nodes = self.num_nodes.repeat_interleave(repeats)
        num_edges = self.num_edges.repeat_interleave(repeats)
        num_cum_nodes = num_nodes.cumsum(0)
        num_cum_edges = num_edges.cumsum(0)
        num_node = num_nodes.sum()
        num_edge = num_edges.sum()
        batch_size = repeats.sum()
        num_graphs = torch.ones(batch_size, device=self.device)

        # special case 1: graphs[i] may have no node or no edge
        # special case 2: repeats[i] may be 0
        cum_repeats_shifted = repeats.cumsum(0) - repeats
        graph_mask = cum_repeats_shifted < batch_size
        cum_repeats_shifted = cum_repeats_shifted[graph_mask]

        index = num_cum_nodes - num_nodes
        index = torch.cat([index, index[cum_repeats_shifted]])
        value = torch.cat([-num_nodes, self.num_nodes[graph_mask]])
        mask = index < num_node
        node_index = scatter_add(value[mask], index[mask], dim_size=num_node)
        node_index = (node_index + 1).cumsum(0) - 1

        index = num_cum_edges - num_edges
        index = torch.cat([index, index[cum_repeats_shifted]])
        value = torch.cat([-num_edges, self.num_edges[graph_mask]])
        mask = index < num_edge
        edge_index = scatter_add(value[mask], index[mask], dim_size=num_edge)
        edge_index = (edge_index + 1).cumsum(0) - 1

        graph_index = torch.repeat_interleave(repeats)

        offsets = self._get_offsets(num_nodes, num_edges)
        edge_list = self.edge_list[edge_index]
        edge_list[:, :2] += (offsets - self._offsets[edge_index]).unsqueeze(-1)

        node_offsets = None
        edge_offsets = None
        graph_offsets = None
        data_dict = {}
        for k, v in self.data_dict.items():
            num_xs = None
            pack_offsets = None
            for _type in self.meta_dict[k]:
                if _type == "node":
                    v = v[node_index]
                    num_xs = num_nodes
                elif _type == "edge":
                    v = v[edge_index]
                    num_xs = num_edges
                elif _type == "graph":
                    v = v[graph_index]
                    num_xs = num_graphs
                elif _type == "node reference":
                    if node_offsets is None:
                        node_offsets = self._get_repeat_pack_offsets(self.num_nodes, repeats)
                    pack_offsets = node_offsets
                elif _type == "edge reference":
                    if edge_offsets is None:
                        edge_offsets = self._get_repeat_pack_offsets(self.num_edges, repeats)
                    pack_offsets = edge_offsets
                elif _type == "graph reference":
                    if graph_offsets is None:
                        graph_offsets = self._get_repeat_pack_offsets(num_graphs, repeats)
                    pack_offsets = graph_offsets
            # add offsets to make references point to indexes in their own graph
            if num_xs is not None and pack_offsets is not None:
                v = v + pack_offsets.repeat_interleave(num_xs)
            data_dict[k] = v

        return type(self)(edge_list, edge_weight=self.edge_weight[edge_index],
                          num_nodes=num_nodes, num_edges=num_edges, num_relation=self.num_relation,
                          offsets=offsets, meta_dict=self.meta_dict, **data_dict)

    def get_item(self, index):
        """
        Get the i-th graph from this packed graph.

        Parameters:
            index (int): graph index

        Returns:
            Graph
        """
        node_index = torch.arange(self.num_cum_nodes[index] - self.num_nodes[index], self.num_cum_nodes[index],
                                  device=self.device)
        edge_index = torch.arange(self.num_cum_edges[index] - self.num_edges[index], self.num_cum_edges[index],
                                  device=self.device)
        graph_index = index
        edge_list = self.edge_list[edge_index].clone()
        edge_list[:, :2] -= self._offsets[edge_index].unsqueeze(-1)
        data_dict, meta_dict = self.data_mask(node_index, edge_index, graph_index=graph_index)

        return self.unpacked_type(edge_list, edge_weight=self.edge_weight[edge_index], num_node=self.num_nodes[index],
                                  num_relation=self.num_relation, meta_dict=meta_dict, **data_dict)

    def _get_cumulative(self, edge_list, num_nodes, num_edges, offsets):
        if edge_list is None:
            raise ValueError("`edge_list` should be provided")
        if num_edges is None:
            raise ValueError("`num_edges` should be provided")

        edge_list = torch.as_tensor(edge_list)
        num_edges = torch.as_tensor(num_edges, device=edge_list.device)
        num_edge = num_edges.sum()
        if num_edge != len(edge_list):
            raise ValueError("Sum of `num_edges` is %d, but found %d edges in `edge_list`" % (num_edge, len(edge_list)))
        num_cum_edges = num_edges.cumsum(0)

        if offsets is None:
            _edge_list = edge_list
        else:
            offsets = torch.as_tensor(offsets, device=edge_list.device)
            _edge_list = edge_list.clone()
            _edge_list[:, :2] -= offsets.unsqueeze(-1)
        if num_nodes is None:
            num_nodes = []
            for num_edge, num_cum_edge in zip(num_edges, num_cum_edges):
                num_nodes.append(self._maybe_num_node(_edge_list[num_cum_edge - num_edge: num_cum_edge]))
        num_nodes = torch.as_tensor(num_nodes, device=edge_list.device)
        num_cum_nodes = num_nodes.cumsum(0)

        return edge_list, num_nodes, num_edges, num_cum_nodes, num_cum_edges, offsets

    def _get_num_xs(self, index, num_cum_xs):
        x = torch.zeros(num_cum_xs[-1], dtype=torch.long, device=self.device)
        x[index] = 1
        num_cum_indexes = x.cumsum(0)
        num_cum_indexes = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), num_cum_indexes])
        new_num_cum_xs = num_cum_indexes[num_cum_xs]
        prepend = torch.zeros(1, dtype=torch.long, device=self.device)
        new_num_xs = torch.diff(new_num_cum_xs, prepend=prepend)
        return new_num_xs

    def data_mask(self, node_index=None, edge_index=None, graph_index=None, include=None, exclude=None):
        data_dict, meta_dict = self.data_by_meta(include, exclude)
        node_mapping = None
        edge_mapping = None
        graph_mapping = None
        for k, v in data_dict.items():
            for type in meta_dict[k]:
                if type == "node" and node_index is not None:
                    v = v[node_index]
                elif type == "edge" and edge_index is not None:
                    v = v[edge_index]
                elif type == "graph" and graph_index is not None:
                    v = v[graph_index]
                elif type == "node reference" and node_index is not None:
                    if node_mapping is None:
                        node_mapping = self._get_mapping(node_index, self.num_node)
                    v = node_mapping[v]
                elif type == "edge reference" and edge_index is not None:
                    if edge_mapping is None:
                        edge_mapping = self._get_mapping(edge_index, self.num_edge)
                    v = edge_mapping[v]
                elif type == "graph reference" and graph_index is not None:
                    if graph_mapping is None:
                        graph_mapping = self._get_mapping(graph_index, self.batch_size)
                    v = graph_mapping[v]
            data_dict[k] = v

        return data_dict, meta_dict

    def __getitem__(self, index):
        # why do we check tuple?
        # case 1: x[0, 1] is parsed as (0, 1)
        # case 2: x[[0, 1]] is parsed as [0, 1]
        if not isinstance(index, tuple):
            index = (index,)

        if isinstance(index[0], int):
            item = self.get_item(index[0])
            if len(index) > 1:
                item = item[index[1:]]
            return item
        if len(index) > 1:
            raise ValueError("Complex indexing is not supported for PackedGraph")

        index = self._standarize_index(index[0], self.batch_size)
        count = index.bincount(minlength=self.batch_size)
        if self.batch_size > 0 and count.max() > 1:
            graph = self.repeat_interleave(count)
            index_order = index.argsort()
            order = torch.zeros_like(index)
            order[index_order] = torch.arange(len(index), dtype=torch.long, device=self.device)
            return graph.subbatch(order)

        return self.subbatch(index)

    def __len__(self):
        return len(self.num_nodes)

    def full(self):
        """
        Return a pack of fully connected graphs.

        This is useful for computing node-pair-wise features.
        The computation can be implemented as message passing over a fully connected graph.

        Returns:
            PackedGraph
        """
        # TODO: more efficient implementation?
        graphs = self.unpack()
        graphs = [graph.full() for graph in graphs]
        return graphs[0].pack(graphs)

    @utils.cached_property
    def node2graph(self):
        """Node id to graph id mapping."""
        range = torch.arange(self.batch_size, device=self.device)
        node2graph = range.repeat_interleave(self.num_nodes)
        return node2graph

    @utils.cached_property
    def edge2graph(self):
        """Edge id to graph id mapping."""
        range = torch.arange(self.batch_size, device=self.device)
        edge2graph = range.repeat_interleave(self.num_edges)
        return edge2graph

    @property
    def batch_size(self):
        """Batch size."""
        return len(self.num_nodes)

    def node_mask(self, index, compact=False):
        """
        Return a masked packed graph based on the specified nodes.

        Note the compact option is only applied to node ids but not graph ids.
        To generate compact graph ids, use :meth:`subbatch`.

        Parameters:
            index (array_like): node index
            compact (bool, optional): compact node ids or not

        Returns:
            PackedGraph
        """
        index = self._standarize_index(index, self.num_node)
        mapping = -torch.ones(self.num_node, dtype=torch.long, device=self.device)
        if compact:
            mapping[index] = torch.arange(len(index), device=self.device)
            num_nodes = self._get_num_xs(index, self.num_cum_nodes)
            offsets = self._get_offsets(num_nodes, self.num_edges)
        else:
            mapping[index] = index
            num_nodes = self.num_nodes
            offsets = self._offsets

        edge_list = self.edge_list.clone()
        edge_list[:, :2] = mapping[edge_list[:, :2]]
        edge_index = (edge_list[:, :2] >= 0).all(dim=-1)
        num_edges = self._get_num_xs(edge_index, self.num_cum_edges)

        if compact:
            data_dict, meta_dict = self.data_mask(index, edge_index)
        else:
            data_dict, meta_dict = self.data_mask(edge_index=edge_index)

        return type(self)(edge_list[edge_index], edge_weight=self.edge_weight[edge_index], num_nodes=num_nodes,
                          num_edges=num_edges, num_relation=self.num_relation, offsets=offsets[edge_index],
                          meta_dict=meta_dict, **data_dict)

    def edge_mask(self, index):
        """
        Return a masked packed graph based on the specified edges.

        Parameters:
            index (array_like): edge index

        Returns:
            PackedGraph
        """
        index = self._standarize_index(index, self.num_edge)
        data_dict, meta_dict = self.data_mask(edge_index=index)
        num_edges = self._get_num_xs(index, self.num_cum_edges)

        return type(self)(self.edge_list[index], edge_weight=self.edge_weight[index], num_nodes=self.num_nodes,
                          num_edges=num_edges, num_relation=self.num_relation, offsets=self._offsets[index],
                          meta_dict=meta_dict, **data_dict)

    def graph_mask(self, index, compact=False):
        """
        Return a masked packed graph based on the specified graphs.

        This function can also be used to re-order the graphs.

        Parameters:
            index (array_like): graph index
            compact (bool, optional): compact graph ids or not

        Returns:
            PackedGraph
        """
        index = self._standarize_index(index, self.batch_size)
        graph_mapping = -torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        graph_mapping[index] = torch.arange(len(index), device=self.device)

        node_index = graph_mapping[self.node2graph] >= 0
        node_index = self._standarize_index(node_index, self.num_node)
        mapping = -torch.ones(self.num_node, dtype=torch.long, device=self.device)
        if compact:
            key = graph_mapping[self.node2graph[node_index]] * self.num_node + node_index
            order = key.argsort()
            node_index = node_index[order]
            mapping[node_index] = torch.arange(len(node_index), device=self.device)
            num_nodes = self.num_nodes[index]
        else:
            mapping[node_index] = node_index
            num_nodes = torch.zeros_like(self.num_nodes)
            num_nodes[index] = self.num_nodes[index]

        edge_list = self.edge_list.clone()
        edge_list[:, :2] = mapping[edge_list[:, :2]]
        edge_index = (edge_list[:, :2] >= 0).all(dim=-1)
        edge_index = self._standarize_index(edge_index, self.num_edge)
        if compact:
            key = graph_mapping[self.edge2graph[edge_index]] * self.num_edge + edge_index
            order = key.argsort()
            edge_index = edge_index[order]
            num_edges = self.num_edges[index]
        else:
            num_edges = torch.zeros_like(self.num_edges)
            num_edges[index] = self.num_edges[index]
        offsets = self._get_offsets(num_nodes, num_edges)

        if compact:
            data_dict, meta_dict = self.data_mask(node_index, edge_index, graph_index=index)
        else:
            data_dict, meta_dict = self.data_mask(edge_index=edge_index)

        return type(self)(edge_list[edge_index], edge_weight=self.edge_weight[edge_index], num_nodes=num_nodes,
                          num_edges=num_edges, num_relation=self.num_relation, offsets=offsets,
                          meta_dict=meta_dict, **data_dict)

    def subbatch(self, index):
        """
        Return a subbatch based on the specified graphs.
        Equivalent to :meth:`graph_mask(index, compact=True) <graph_mask>`.

        Parameters:
            index (array_like): graph index

        Returns:
            PackedGraph

        See also:
            :meth:`PackedGraph.graph_mask`
        """
        return self.graph_mask(index, compact=True)

    def line_graph(self):
        """
        Construct a packed line graph of this packed graph.
        The node features of the line graphs are inherited from the edge features of the original graphs.

        In the line graph, each node corresponds to an edge in the original graph.
        For a pair of edges (a, b) and (b, c) that share the same intermediate node in the original graph,
        there is a directed edge (a, b) -> (b, c) in the line graph.

        Returns:
            PackedGraph
        """
        node_in, node_out = self.edge_list.t()[:2]
        edge_index = torch.arange(self.num_edge, device=self.device)
        edge_in = edge_index[node_out.argsort()]
        edge_out = edge_index[node_in.argsort()]

        degree_in = node_in.bincount(minlength=self.num_node)
        degree_out = node_out.bincount(minlength=self.num_node)
        size = degree_out * degree_in
        starts = (size.cumsum(0) - size).repeat_interleave(size)
        range = torch.arange(size.sum(), device=self.device)
        # each node u has degree_out[u] * degree_in[u] local edges
        local_index = range - starts
        local_inner_size = degree_in.repeat_interleave(size)
        edge_in_offset = (degree_out.cumsum(0) - degree_out).repeat_interleave(size)
        edge_out_offset = (degree_in.cumsum(0) - degree_in).repeat_interleave(size)
        edge_in_index = local_index // local_inner_size + edge_in_offset
        edge_out_index = local_index % local_inner_size + edge_out_offset

        edge_in = edge_in[edge_in_index]
        edge_out = edge_out[edge_out_index]
        edge_list = torch.stack([edge_in, edge_out], dim=-1)
        node_feature = getattr(self, "edge_feature", None)
        num_nodes = self.num_edges
        num_edges = scatter_add(size, self.node2graph, dim=0, dim_size=self.batch_size)
        offsets = self._get_offsets(num_nodes, num_edges)

        return PackedGraph(edge_list, num_nodes=num_nodes, num_edges=num_edges, offsets=offsets,
                           node_feature=node_feature)

    def undirected(self, add_inverse=False):
        """
        Flip all the edges to create undirected graphs.

        For knowledge graphs, the flipped edges can either have the original relation or an inverse relation.
        The inverse relation for relation :math:`r` is defined as :math:`|R| + r`.

        Parameters:
            add_inverse (bool, optional): whether to use inverse relations for flipped edges
        """
        edge_list = self.edge_list.clone()
        edge_list[:, :2] = edge_list[:, :2].flip(1)
        num_relation = self.num_relation
        if num_relation and add_inverse:
            edge_list[:, 2] += num_relation
            num_relation = num_relation * 2
        edge_list = torch.stack([self.edge_list, edge_list], dim=1).flatten(0, 1)
        offsets = self._offsets.unsqueeze(-1).expand(-1, 2).flatten()

        index = torch.arange(self.num_edge, device=self.device).unsqueeze(-1).expand(-1, 2).flatten()
        data_dict, meta_dict = self.data_mask(edge_index=index, exclude="edge reference")

        return type(self)(edge_list, edge_weight=self.edge_weight[index], num_nodes=self.num_nodes,
                          num_edges=self.num_edges * 2, num_relation=num_relation, offsets=offsets,
                          meta_dict=meta_dict, **data_dict)

    def detach(self):
        """
        Detach this packed graph.
        """
        return type(self)(self.edge_list.detach(), edge_weight=self.edge_weight.detach(),
                          num_nodes=self.num_nodes, num_edges=self.num_edges, num_relation=self.num_relation,
                          offsets=self._offsets, meta_dict=self.meta_dict, **utils.detach(self.data_dict))

    def clone(self):
        """
        Clone this packed graph.
        """
        return type(self)(self.edge_list.clone(), edge_weight=self.edge_weight.clone(),
                          num_nodes=self.num_nodes, num_edges=self.num_edges, num_relation=self.num_relation,
                          offsets=self._offsets, meta_dict=self.meta_dict, **utils.clone(self.data_dict))

    def cuda(self, *args, **kwargs):
        """
        Return a copy of this packed graph in CUDA memory.

        This is a non-op if the graph is already on the correct device.
        """
        edge_list = self.edge_list.cuda(*args, **kwargs)

        if edge_list is self.edge_list:
            return self
        else:
            return type(self)(edge_list, edge_weight=self.edge_weight,
                              num_nodes=self.num_nodes, num_edges=self.num_edges, num_relation=self.num_relation,
                              offsets=self._offsets, meta_dict=self.meta_dict,
                              **utils.cuda(self.data_dict, *args, **kwargs))

    def cpu(self):
        """
        Return a copy of this packed graph in CPU memory.

        This is a non-op if the graph is already in CPU memory.
        """
        edge_list = self.edge_list.cpu()

        if edge_list is self.edge_list:
            return self
        else:
            return type(self)(edge_list, edge_weight=self.edge_weight,
                              num_nodes=self.num_nodes, num_edges=self.num_edges, num_relation=self.num_relation,
                              offsets=self._offsets, meta_dict=self.meta_dict, **utils.cpu(self.data_dict))

    def __repr__(self):
        fields = ["batch_size=%d" % self.batch_size,
                  "num_nodes=%s" % pretty.long_array(self.num_nodes.tolist()),
                  "num_edges=%s" % pretty.long_array(self.num_edges.tolist())]
        if self.num_relation is not None:
            fields.append("num_relation=%d" % self.num_relation)
        if self.device.type != "cpu":
            fields.append("device='%s'" % self.device)
        return "%s(%s)" % (self.__class__.__name__, ", ".join(fields))

    def visualize(self, titles=None, save_file=None, figure_size=(3, 3), layout="spring", num_row=None, num_col=None):
        """
        Visualize the packed graphs with matplotlib.

        Parameters:
            titles (list of str, optional): title for each graph. Default is the ID of each graph.
            save_file (str, optional): ``png`` or ``pdf`` file to save visualization.
                If not provided, show the figure in window.
            figure_size (tuple of int, optional): width and height of the figure
            layout (str, optional): graph layout
            num_row (int, optional): number of rows in the figure
            num_col (int, optional): number of columns in the figure

        See also:
            `NetworkX graph layout`_

            .. _NetworkX graph layout:
                https://networkx.github.io/documentation/stable/reference/drawing.html#module-networkx.drawing.layout
        """
        if titles is None:
            graph = self.get_item(0)
            titles = ["%s %d" % (type(graph).__name__, i) for i in range(self.batch_size)]
        if num_col is None:
            if num_row is None:
                num_col = math.ceil(self.batch_size ** 0.5)
            else:
                num_col = math.ceil(self.batch_size / num_row)
        if num_row is None:
            num_row = math.ceil(self.batch_size / num_col)

        figure_size = (num_col * figure_size[0], num_row * figure_size[1])
        fig = plt.figure(figsize=figure_size)

        for i in range(self.batch_size):
            graph = self.get_item(i)
            ax = fig.add_subplot(num_row, num_col, i + 1)
            graph.visualize(title=titles[i], ax=ax, layout=layout)
        # remove the space of axis labels
        fig.tight_layout()

        if save_file:
            fig.savefig(save_file)
        else:
            fig.show()


Graph.packed_type = PackedGraph


def cat(graphs):
    for i, graph in enumerate(graphs):
        if not isinstance(graph, PackedGraph):
            graphs[i] = graph.pack([graph])

    edge_list = torch.cat([graph.edge_list for graph in graphs])
    pack_num_nodes = torch.stack([graph.num_node for graph in graphs])
    pack_num_edges = torch.stack([graph.num_edge for graph in graphs])
    pack_num_cum_edges = pack_num_edges.cumsum(0)
    graph_index = pack_num_cum_edges < len(edge_list)
    pack_offsets = scatter_add(pack_num_nodes[graph_index], pack_num_cum_edges[graph_index],
                               dim_size=len(edge_list))
    pack_offsets = pack_offsets.cumsum(0)

    edge_list[:, :2] += pack_offsets.unsqueeze(-1)
    offsets = torch.cat([graph._offsets for graph in graphs]) + pack_offsets

    edge_weight = torch.cat([graph.edge_weight for graph in graphs])
    num_nodes = torch.cat([graph.num_nodes for graph in graphs])
    num_edges = torch.cat([graph.num_edges for graph in graphs])
    num_relation = graphs[0].num_relation
    assert all(graph.num_relation == num_relation for graph in graphs)

    # only keep attributes that exist in all graphs
    # TODO: this interface is not safe. re-design the interface
    keys = set(graphs[0].meta_dict.keys())
    for graph in graphs:
        keys = keys.intersection(graph.meta_dict.keys())

    meta_dict = {k: graphs[0].meta_dict[k] for k in keys}
    data_dict = {}
    for k in keys:
        data_dict[k] = torch.cat([graph.data_dict[k] for graph in graphs])

    return type(graphs[0])(edge_list, edge_weight=edge_weight,
                           num_nodes=num_nodes, num_edges=num_edges, num_relation=num_relation, offsets=offsets,
                           meta_dict=meta_dict, **data_dict)