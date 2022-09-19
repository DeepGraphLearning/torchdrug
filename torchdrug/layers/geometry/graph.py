import math

import torch
from torch import nn

from torchdrug import core, data
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("layers.GraphConstruction")
class GraphConstruction(nn.Module, core.Configurable):
    """
    Construct a new graph from an existing graph.

    See `torchdrug.layers.geometry` for a full list of available node and edge layers.

    Parameters:
        node_layers (list of nn.Module, optional): modules to construct nodes of the new graph
        edge_layers (list of nn.Module, optional): modules to construct edges of the new graph
        edge_feature (str, optional): edge features in the new graph.
            Available features are ``residue_type``, ``gearnet``.

            1. For ``residue_type``, the feature of the edge :math:`e_{ij}` between residue :math:`i` and residue
                :math:`j` is the concatenation ``[residue_type(i), residue_type(j)]``.
            2. For ``gearnet``, the feature of the edge :math:`e_{ij}` between residue :math:`i` and residue :math:`j`
                is the concatenation ``[residue_type(i), residue_type(j), edge_type(e_ij),
                sequential_distance(i,j), spatial_distance(i,j)]``.
    """

    max_seq_dist = 10

    def __init__(self, node_layers=None, edge_layers=None, edge_feature="residue_type"):
        super(GraphConstruction, self).__init__()
        if node_layers is None:
            self.node_layers = nn.ModuleList()
        else:
            self.node_layers = nn.ModuleList(node_layers)
        if edge_layers is None:
            edge_layers = nn.ModuleList()
        else:
            edge_layers = nn.ModuleList(edge_layers)
        self.edge_layers = edge_layers
        self.edge_feature = edge_feature

    def edge_residue_type(self, graph, edge_list):
        node_in, node_out, _ = edge_list.t()
        residue_in, residue_out = graph.atom2residue[node_in], graph.atom2residue[node_out]
        in_residue_type = graph.residue_type[residue_in]
        out_residue_type = graph.residue_type[residue_out]

        return torch.cat([
            functional.one_hot(in_residue_type, len(data.Protein.residue2id)),
            functional.one_hot(out_residue_type, len(data.Protein.residue2id))
        ], dim=-1)

    def edge_gearnet(self, graph, edge_list, num_relation):
        node_in, node_out, r = edge_list.t()
        residue_in, residue_out = graph.atom2residue[node_in], graph.atom2residue[node_out]
        in_residue_type = graph.residue_type[residue_in]
        out_residue_type = graph.residue_type[residue_out]
        sequential_dist = torch.abs(residue_in - residue_out)
        spatial_dist = (graph.node_position[node_in] - graph.node_position[node_out]).norm(dim=-1)

        return torch.cat([
            functional.one_hot(in_residue_type, len(data.Protein.residue2id)),
            functional.one_hot(out_residue_type, len(data.Protein.residue2id)),
            functional.one_hot(r, num_relation),
            functional.one_hot(sequential_dist.clamp(max=self.max_seq_dist), self.max_seq_dist + 1),
            spatial_dist.unsqueeze(-1)
        ], dim=-1)

    def apply_node_layer(self, graph):
        for layer in self.node_layers:
            graph = layer(graph)
        return graph

    def apply_edge_layer(self, graph):
        if not self.edge_layers:
            return graph

        edge_list = []
        num_edges = []
        num_relations = []
        for layer in self.edge_layers:
            edges, num_relation = layer(graph)
            edge_list.append(edges)
            num_edges.append(len(edges))
            num_relations.append(num_relation)

        edge_list = torch.cat(edge_list)
        num_edges = torch.tensor(num_edges, device=graph.device)
        num_relations = torch.tensor(num_relations, device=graph.device)
        num_relation = num_relations.sum()
        offsets = (num_relations.cumsum(0) - num_relations).repeat_interleave(num_edges)
        edge_list[:, 2] += offsets

        # reorder edges into a valid PackedGraph
        node_in = edge_list[:, 0]
        edge2graph = graph.node2graph[node_in]
        order = edge2graph.argsort()
        edge_list = edge_list[order]
        num_edges = edge2graph.bincount(minlength=graph.batch_size)
        offsets = (graph.num_cum_nodes - graph.num_nodes).repeat_interleave(num_edges)

        if self.edge_feature == "residue_type":
            edge_feature = self.edge_residue_type(graph, edge_list)
        elif self.edge_feature == "gearnet":
            edge_feature = self.edge_gearnet(graph, edge_list, num_relation)
        else:
            raise ValueError("Unknown edge feature `%s`" % self.edge_feature)
        data_dict, meta_dict = graph.data_by_meta(include=("node", "residue", "node reference", "residue reference"))

        if isinstance(graph, data.PackedProtein):
            data_dict["num_residues"] = graph.num_residues
        if isinstance(graph, data.PackedMolecule):
            data_dict["bond_type"] = torch.zeros_like(edge_list[:, 2])
        return type(graph)(edge_list, num_nodes=graph.num_nodes, num_edges=num_edges, num_relation=num_relation,
                           view=graph.view, offsets=offsets, edge_feature=edge_feature,
                           meta_dict=meta_dict, **data_dict)

    def forward(self, graph):
        """
        Generate a new graph based on the input graph and pre-defined node and edge layers.

        Parameters:
            graph (Graph): :math:`n` graph(s)

        Returns:
            graph (Graph): new graph(s)
        """
        graph = self.apply_node_layer(graph)
        graph = self.apply_edge_layer(graph)
        return graph


@R.register("layers.SpatialLineGraph")
class SpatialLineGraph(nn.Module, core.Configurable):
    """
    Spatial line graph construction module from `Protein Representation Learning by Geometric Structure Pretraining`_.

    .. _Protein Representation Learning by Geometric Structure Pretraining:
        https://arxiv.org/pdf/2203.06125.pdf

    Parameters:
        num_angle_bin (int, optional): number of bins to discretize angles between edges
    """

    def __init__(self, num_angle_bin=8):
        super(SpatialLineGraph, self).__init__()
        self.num_angle_bin = num_angle_bin

    def forward(self, graph):
        """
        Generate the spatial line graph of the input graph.
        The edge types are decided by the angles between two adjacent edges in the input graph.

        Parameters:
            graph (PackedGraph): :math:`n` graph(s)

        Returns:
            graph (PackedGraph): the spatial line graph
        """
        line_graph = graph.line_graph()
        node_in, node_out = graph.edge_list[:, :2].t()
        edge_in, edge_out = line_graph.edge_list.t()

        # compute the angle ijk
        node_i = node_out[edge_out]
        node_j = node_in[edge_out]
        node_k = node_in[edge_in]
        vector1 = graph.node_position[node_i] - graph.node_position[node_j]
        vector2 = graph.node_position[node_k] - graph.node_position[node_j]
        x = (vector1 * vector2).sum(dim=-1)
        y = torch.cross(vector1, vector2).norm(dim=-1)
        angle = torch.atan2(y, x)
        relation = (angle / math.pi * self.num_angle_bin).long()
        edge_list = torch.cat([line_graph.edge_list, relation.unsqueeze(-1)], dim=-1)

        return type(line_graph)(edge_list, num_nodes=line_graph.num_nodes, offsets=line_graph._offsets,
                                num_edges=line_graph.num_edges, num_relation=self.num_angle_bin,
                                meta_dict=line_graph.meta_dict, **line_graph.data_dict)
