import torch
from torch import nn
from torch_cluster import knn_graph, radius_graph

from torchdrug import core, data
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("layers.geometry.BondEdge")
class BondEdge(nn.Module, core.Configurable):
    """
    Construct all bond edges.
    """

    def forward(self, graph):
        """
        Return bond edges from the input graph. Edge types are inherited from the input graph.

        Parameters:
            graph (Graph): :math:`n` graph(s)

        Returns:
            (Tensor, int): edge list of shape :math:`(|E|, 3)`, number of relations
        """
        return graph.edge_list, graph.num_relation


@R.register("layers.geometry.KNNEdge")
class KNNEdge(nn.Module, core.Configurable):
    """
    Construct edges between each node and its nearest neighbors.

    Parameters:
        k (int, optional): number of neighbors
        min_distance (int, optional): minimum distance between the residues of two nodes
    """

    eps = 1e-10

    def __init__(self, k=10, min_distance=5, max_distance=0):
        super(KNNEdge, self).__init__()
        self.k = k
        self.min_distance = min_distance
        self.max_distance = max_distance

    def forward(self, graph):
        """
        Return KNN edges constructed from the input graph.

        Parameters:
            graph (Graph): :math:`n` graph(s)

        Returns:
            (Tensor, int): edge list of shape :math:`(|E|, 3)`, number of relations
        """
        edge_list = knn_graph(graph.node_position, k=self.k, batch=graph.node2graph).t()
        relation = torch.zeros(len(edge_list), 1, dtype=torch.long, device=graph.device)
        edge_list = torch.cat([edge_list, relation], dim=-1)

        if self.min_distance > 0:
            node_in, node_out = edge_list.t()[:2]
            mask = (graph.atom2residue[node_in] - graph.atom2residue[node_out]).abs() < self.min_distance
            edge_list = edge_list[~mask]

        if self.max_distance > 0:
            node_in, node_out = edge_list.t()[:2]
            mask = (graph.atom2residue[node_in] - graph.atom2residue[node_out]).abs() > self.max_distance
            edge_list = edge_list[~mask]

        node_in, node_out = edge_list.t()[:2]
        mask = (graph.node_position[node_in] - graph.node_position[node_out]).norm(dim=-1) < self.eps
        edge_list = edge_list[~mask]

        return edge_list, 1


@R.register("layers.geometry.SpatialEdge")
class SpatialEdge(nn.Module, core.Configurable):
    """
    Construct edges between nodes within a specified radius.

    Parameters:
        radius (float, optional): spatial radius
        min_distance (int, optional): minimum distance between the residues of two nodes
    """

    eps = 1e-10

    def __init__(self, radius=5, min_distance=5, max_num_neighbors=32):
        super(SpatialEdge, self).__init__()
        self.radius = radius
        self.min_distance = min_distance
        self.max_num_neighbors = max_num_neighbors

    def forward(self, graph):
        """
        Return spatial radius edges constructed based on the input graph.

        Parameters:
            graph (Graph): :math:`n` graph(s)

        Returns:
            (Tensor, int): edge list of shape :math:`(|E|, 3)`, number of relations
        """
        edge_list = radius_graph(graph.node_position, r=self.radius, batch=graph.node2graph, max_num_neighbors=self.max_num_neighbors).t()
        relation = torch.zeros(len(edge_list), 1, dtype=torch.long, device=graph.device)
        edge_list = torch.cat([edge_list, relation], dim=-1)

        if self.min_distance > 0:
            node_in, node_out = edge_list.t()[:2]
            mask = (graph.atom2residue[node_in] - graph.atom2residue[node_out]).abs() < self.min_distance
            edge_list = edge_list[~mask]

        node_in, node_out = edge_list.t()[:2]
        mask = (graph.node_position[node_in] - graph.node_position[node_out]).norm(dim=-1) < self.eps
        edge_list = edge_list[~mask]

        return edge_list, 1


@R.register("layers.geometry.SequentialEdge")
class SequentialEdge(nn.Module, core.Configurable):
    """
    Construct edges between atoms within close residues.

    Parameters:
        max_distance (int, optional): maximum distance between two residues in the sequence
    """

    def __init__(self, max_distance=2):
        super(SequentialEdge, self).__init__()
        self.max_distance = max_distance

    def forward(self, graph):
        """
        Return sequential edges constructed based on the input graph.
        Edge types are defined by the relative distance between two residues in the sequence

        Parameters:
            graph (Graph): :math:`n` graph(s)

        Returns:
            (Tensor, int): edge list of shape :math:`(|E|, 3)`, number of relations
        """
        residue2num_atom = graph.atom2residue.bincount(minlength=graph.num_residue)
        edge_list = []
        for i in range(-self.max_distance, self.max_distance + 1):
            node_index = torch.arange(graph.num_node, device=graph.device)
            residue_index = torch.arange(graph.num_residue, device=graph.device)
            if i > 0:
                is_node_in = graph.atom2residue < graph.num_cum_residues[graph.atom2graph] - i
                is_node_out = graph.atom2residue >= (graph.num_cum_residues - graph.num_residues)[graph.atom2graph] + i
                is_residue_in = residue_index < graph.num_cum_residues[graph.residue2graph] - i
                is_residue_out = residue_index >= (graph.num_cum_residues - graph.num_residues)[graph.residue2graph] + i
            else:
                is_node_in = graph.atom2residue >= (graph.num_cum_residues - graph.num_residues)[graph.atom2graph] - i
                is_node_out = graph.atom2residue < graph.num_cum_residues[graph.atom2graph] + i
                is_residue_in = residue_index >= (graph.num_cum_residues - graph.num_residues)[graph.residue2graph] - i
                is_residue_out = residue_index < graph.num_cum_residues[graph.residue2graph] + i
            node_in = node_index[is_node_in]
            node_out = node_index[is_node_out]
            # group atoms by residue ids
            node_in = node_in[graph.atom2residue[node_in].argsort()]
            node_out = node_out[graph.atom2residue[node_out].argsort()]
            num_node_in = residue2num_atom[is_residue_in]
            num_node_out = residue2num_atom[is_residue_out]
            node_in, node_out = functional.variadic_meshgrid(node_in, num_node_in, node_out, num_node_out)
            # exclude cross-chain edges
            is_same_chain = (graph.chain_id[graph.atom2residue[node_in]] == graph.chain_id[graph.atom2residue[node_out]])
            node_in = node_in[is_same_chain]
            node_out = node_out[is_same_chain]
            relation = torch.ones(len(node_in), dtype=torch.long, device=graph.device) * (i + self.max_distance)
            edges = torch.stack([node_in, node_out, relation], dim=-1)
            edge_list.append(edges)

        edge_list = torch.cat(edge_list)

        return edge_list, 2 * self.max_distance + 1


@R.register("layers.geometry.AlphaCarbonNode")
class AlphaCarbonNode(nn.Module, core.Configurable):
    """
    Construct only alpha carbon atoms.
    """

    def forward(self, graph):
        """
        Return a subgraph that only consists of alpha carbon nodes.

        Parameters:
            graph (Graph): :math:`n` graph(s)
        """
        mask = (graph.atom_name == data.Protein.atom_name2id["CA"]) & (graph.atom2residue != -1)
        residue2num_atom = graph.atom2residue[mask].bincount(minlength=graph.num_residue)
        residue_mask = residue2num_atom > 0
        mask = mask & residue_mask[graph.atom2residue]
        graph = graph.subgraph(mask).subresidue(residue_mask)
        assert (graph.num_node == graph.num_residue).all()

        return graph


@R.register("layers.geometry.IdentityNode")
class IdentityNode(nn.Module, core.Configurable):
    """
    Construct all nodes as the input.
    """

    def forward(self, graph):
        """
        Return the input graph as is.

        Parameters:
            graph (Graph): :math:`n` graph(s)
        """
        return graph


@R.register("layers.geometry.RandomEdgeMask")
class RandomEdgeMask(nn.Module, core.Configurable):
    """
    Construct nodes by random edge masking.

    Parameters:
        mask_rate (float, optional): rate of masked edges
    """

    def __init__(self, mask_rate=0.15):
        super(RandomEdgeMask, self).__init__()
        self.mask_rate = mask_rate

    def forward(self, graph):
        """
        Return a graph with some edges masked out.

        Parameters:
            graph (Graph): :math:`n` graph(s)
        """
        num_samples = (graph.num_edges * self.mask_rate).long().clamp(min=1)
        num_sample = num_samples.sum()
        sample2graph = functional._size_to_index(num_samples)
        edge_index = (torch.rand(num_sample, device=graph.device) * graph.num_edges[sample2graph]).long()
        edge_index = edge_index + (graph.num_cum_edges - graph.num_edges)[sample2graph]
        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)

        return graph.edge_mask(edge_mask)


@R.register("layers.geometry.SubsequenceNode")
class SubsequenceNode(nn.Module, core.Configurable):
    """
    Construct nodes by taking a random subsequence of the original graph.

    Parameters:
        max_length (int, optional): maximal length of the sequence after cropping
    """

    def __init__(self, max_length=100):
        super(SubsequenceNode, self).__init__()
        self.max_length = max_length

    def forward(self, graph):
        """
        Randomly take a subsequence of the specified length.
        Return the full sequence if the sequence is shorter than the specified length.

        Parameters:
            graph (Graph): :math:`n` graph(s)
        """
        starts = (torch.rand(graph.batch_size, device=graph.device) *
                  (graph.num_residues - self.max_length).clamp(min=0)).long()
        ends = torch.min(starts + self.max_length, graph.num_residues)
        starts = starts + graph.num_cum_residues - graph.num_residues
        ends = ends + graph.num_cum_residues - graph.num_residues

        node_mask = functional.multi_slice_mask(starts, ends, graph.num_residue)
        residue_mask = node_mask[graph.atom2residue]
        graph = graph.subresidue(residue_mask)

        return graph


@R.register("layers.geometry.SubspaceNode")
class SubspaceNode(nn.Module, core.Configurable):
    """
    Construct nodes by taking a spatial ball of the original graph.

    Parameters:
        entity_level (str, optional): level to perform cropping. 
            Available options are ``node``, ``atom`` and ``residue``.
        min_radius (float, optional): minimum radius of the spatial ball
        min_neighbor (int, optional): minimum number of nodes in the spatial ball
    """

    def __init__(self, entity_level="node", min_radius=15.0, min_neighbor=50):
        super(SubspaceNode, self).__init__()
        self.entity_level = entity_level
        self.min_radius = min_radius
        self.min_neighbor = min_neighbor

    def forward(self, graph):
        """
        Randomly pick a node as the center, and crop a spatial ball
        that is at least `radius` large and contain at least `k` nodes.

        Parameters:
            graph (Graph): :math:`n` graph(s)
        """
        node_in = torch.arange(graph.num_node, device=graph.device)
        node_in = functional.variadic_sample(node_in, graph.num_nodes, 1).squeeze(-1)
        node_in = node_in.repeat_interleave(graph.num_nodes)
        node_out = torch.arange(graph.num_node, device=graph.device)
        dist = (graph.node_position[node_in] - graph.node_position[node_out]).norm(dim=-1)
        topk_dist = functional.variadic_topk(dist, graph.num_nodes, self.min_neighbor, largest=False)[0]
        radius = (topk_dist[:, -1] * 1.5).clamp(min=self.min_radius)
        radius = radius.repeat_interleave(graph.num_nodes)
        node_index = node_out[dist < radius]

        if self.entity_level in ["node", "atom"]:
            graph = graph.subgraph(node_index)
        else:
            residue_index = graph.atom2residue[node_index].unique()
            graph = graph.subresidue(residue_index)

        return graph
