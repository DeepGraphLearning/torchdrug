import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_min

from torchdrug import core, tasks, layers
from torchdrug.data import constant
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("tasks.EdgePrediction")
class EdgePrediction(tasks.Task, core.Configurable):
    """
    Edge prediction task proposed in `Inductive Representation Learning on Large Graphs`_.

    .. _Inductive Representation Learning on Large Graphs:
        https://arxiv.org/abs/1706.02216

    Parameters:
        model (nn.Module): node representation model
    """

    def __init__(self, model):
        super(EdgePrediction, self).__init__()
        self.model = model

    def _get_directed(self, graph):
        mask = graph.edge_list[:, 0] < graph.edge_list[:, 1]
        graph = graph.edge_mask(mask)
        return graph

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]

        output = self.model(graph, graph.node_feature.float(), all_loss, metric)
        node_feature = output["node_feature"]

        graph = self._get_directed(graph)
        node_in, node_out = graph.edge_list.t()[:2]
        neg_index = (torch.rand(2, graph.num_edge, device=self.device) * graph.num_nodes[graph.edge2graph]).long()
        neg_index = neg_index + (graph.num_cum_nodes - graph.num_nodes)[graph.edge2graph]
        node_in = torch.cat([node_in, neg_index[0]])
        node_out = torch.cat([node_out, neg_index[1]])

        pred = torch.einsum("bd, bd -> b", node_feature[node_in], node_feature[node_out])
        return pred

    def target(self, batch):
        graph = batch["graph"]
        target = torch.ones(graph.num_edge, device=self.device)
        target[graph.num_edge // 2:] = 0
        return target

    def evaluate(self, pred, target):
        metric = {}
        accuracy = ((pred > 0) == (target > 0.5)).float().mean()

        name = tasks._get_metric_name("acc")
        metric[name] = accuracy

        return metric

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)
        target = self.target(batch)
        loss = F.binary_cross_entropy_with_logits(pred, target)
        name = tasks._get_criterion_name("bce")
        metric[name] = loss
        metric.update(self.evaluate(pred, target))

        all_loss += loss

        return all_loss, metric


@R.register("tasks.AttributeMasking")
class AttributeMasking(tasks.Task, core.Configurable):
    """
    Attribute masking proposed in `Strategies for Pre-training Graph Neural Networks`_.

    .. _Strategies for Pre-training Graph Neural Networks:
        https://arxiv.org/abs/1905.12265

    Parameters:
        model (nn.Module): node representation model
        mask_rate (float, optional): rate of masked nodes
        num_mlp_layer (int, optional): number of MLP layers
    """

    def __init__(self, model, mask_rate=0.15, num_mlp_layer=2, graph_construction_model=None):
        super(AttributeMasking, self).__init__()
        self.model = model
        self.mask_rate = mask_rate
        self.num_mlp_layer = num_mlp_layer
        self.graph_construction_model = graph_construction_model

    def preprocess(self, train_set, valid_set, test_set):
        data = train_set[0]
        self.view = getattr(data["graph"], "view", "atom")
        if hasattr(self.model, "node_output_dim"):
            model_output_dim = self.model.node_output_dim
        else:
            model_output_dim = self.model.output_dim
        if self.view == "atom":
            num_label = constant.NUM_ATOM
        else:
            num_label = constant.NUM_AMINO_ACID
        self.mlp = layers.MLP(model_output_dim, [model_output_dim] * (self.num_mlp_layer - 1) + [num_label])

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model.apply_node_layer(graph)

        num_nodes = graph.num_nodes if self.view in ["atom", "node"] else graph.num_residues
        num_cum_nodes = num_nodes.cumsum(0)
        num_samples = (num_nodes * self.mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        sample2graph = functional._size_to_index(num_samples)
        node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
        node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]

        if self.view == "atom":
            target = graph.atom_type[node_index]
            input = graph.node_feature.float()
            input[node_index] = 0
        else:
            target = graph.residue_type[node_index]
            with graph.residue():
                graph.residue_feature[node_index] = 0
                graph.residue_type[node_index] = 0
            # Generate masked edge features. Any better implementation?
            if self.graph_construction_model:
                graph = self.graph_construction_model.apply_edge_layer(graph)
            input = graph.residue_feature.float()

        output = self.model(graph, input, all_loss, metric)
        if self.view in ["node", "atom"]:
            node_feature = output["node_feature"]
        else:
            node_feature = output.get("residue_feature", output.get("node_feature"))
        node_feature = node_feature[node_index]
        pred = self.mlp(node_feature)

        return pred, target

    def evaluate(self, pred, target):
        metric = {}
        accuracy = (pred.argmax(dim=-1) == target).float().mean()

        name = tasks._get_metric_name("acc")
        metric[name] = accuracy

        return metric

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(pred, target))

        loss = F.cross_entropy(pred, target)
        name = tasks._get_criterion_name("ce")
        metric[name] = loss

        all_loss += loss

        return all_loss, metric


@R.register("tasks.ContextPrediction")
class ContextPrediction(tasks.Task, core.Configurable):
    """
    Context prediction task proposed in `Strategies for Pre-training Graph Neural Networks`_.

    .. _Strategies for Pre-training Graph Neural Networks:
        https://arxiv.org/abs/1905.12265

    For a given center node, the subgraph is defined as a k-hop neighborhood (inclusive) around the selected node.
    The context graph is defined as the surrounding graph structure between r1- (exclusive) and r2-hop (inclusive)
    from the center node. Nodes between k- and r1-hop are picked as anchor nodes for the context representation.

    Parameters:
        model (nn.Module): node representation model for subgraphs.
        context_model (nn.Module, optional): node representation model for context graphs.
            By default, use the same architecture as ``model`` without parameter sharing.
        k (int, optional): radius for subgraphs
        r1 (int, optional): inner radius for context graphs
        r2 (int, optional): outer radius for context graphs
        readout (nn.Module, optional): readout function over context anchor nodes
        num_negative (int, optional): number of negative samples per positive sample
    """

    def __init__(self, model, context_model=None, k=5, r1=4, r2=7, readout="mean", num_negative=1):
        super(ContextPrediction, self).__init__()
        self.model = model
        self.k = k
        self.r1 = r1
        self.r2 = r2
        self.num_negative = num_negative
        assert r1 < k < r2

        if context_model is None:
            self.context_model = copy.deepcopy(model)
        else:
            self.context_model = context_model
        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def substruct_and_context(self, graph):
        center_index = (torch.rand(len(graph), device=self.device) * graph.num_nodes).long()
        center_index = center_index + graph.num_cum_nodes - graph.num_nodes
        dist = torch.full((graph.num_node,), self.r2 + 1, dtype=torch.long, device=self.device)
        dist[center_index] = 0

        # single source shortest path
        node_in, node_out = graph.edge_list.t()[:2]
        for i in range(self.r2):
            new_dist = scatter_min(dist[node_in], node_out, dim_size=graph.num_node)[0] + 1
            dist = torch.min(dist, new_dist)

        substruct_mask = dist <= self.k
        context_mask = (dist > self.r1) & (dist <= self.r2)
        is_center_node = functional.as_mask(center_index, graph.num_node)
        is_anchor_node = (dist > self.r1) & (dist <= self.k)

        substruct = graph.clone()
        context = graph.clone()
        with substruct.node():
            substruct.is_center_node = is_center_node
        with context.node():
            context.is_anchor_node = is_anchor_node

        substruct = substruct.subgraph(substruct_mask)
        context = context.subgraph(context_mask)
        valid = context.num_nodes > 0
        substruct = substruct[valid]
        context = context[valid]

        return substruct, context

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        substruct, context = self.substruct_and_context(graph)
        anchor = context.subgraph(context.is_anchor_node)

        substruct_output = self.model(substruct, substruct.node_feature.float(), all_loss, metric)
        substruct_feature = substruct_output["node_feature"][substruct.is_center_node]

        context_output = self.context_model(context, context.node_feature.float(), all_loss, metric)
        anchor_feature = context_output["node_feature"][context.is_anchor_node]
        context_feature = self.readout(anchor, anchor_feature)

        shift = torch.arange(self.num_negative, device=self.device) + 1
        neg_index = (torch.arange(len(context), device=self.device).unsqueeze(-1) + shift) % len(context) # (batch_size, num_negative)
        context_feature = torch.cat([context_feature.unsqueeze(1), context_feature[neg_index]], dim=1)
        substruct_feature = substruct_feature.unsqueeze(1).expand_as(context_feature)

        pred = torch.einsum("bnd, bnd -> bn", substruct_feature, context_feature)
        target = torch.zeros_like(pred)
        target[:, 0] = 1
        return pred, target

    def evaluate(self, pred, target):
        metric = {}
        accuracy = ((pred > 0) == (target > 0.5)).float().mean()

        name = tasks._get_metric_name("acc")
        metric[name] = accuracy

        return metric

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(pred, target))

        loss = F.binary_cross_entropy_with_logits(pred, target)
        name = tasks._get_criterion_name("bce")
        metric[name] = loss

        all_loss += loss

        return all_loss, metric


@R.register("tasks.DistancePrediction")
class DistancePrediction(tasks.Task, core.Configurable):
    """
    Pairwise spatial distance prediction task proposed in
    `Protein Representation Learning by Geometric Structure Pretraining`_.

    .. _Protein Representation Learning by Geometric Structure Pretraining:
        https://arxiv.org/pdf/2203.06125.pdf

    Randomly select some edges and predict the lengths of the edges using the representations of two nodes.
    The selected edges are removed from the input graph to prevent trivial solutions.

    Parameters:
        model (nn.Module): node representation model
        num_sample (int, optional): number of edges selected from each graph
        num_mlp_layer (int, optional): number of MLP layers in distance predictor
        graph_construction_model (nn.Module, optional): graph construction model
    """

    def __init__(self, model, num_sample=256, num_mlp_layer=2, graph_construction_model=None):
        super(DistancePrediction, self).__init__()
        self.model = model
        self.num_sample = num_sample
        self.num_mlp_layer = num_mlp_layer
        self.graph_construction_model = graph_construction_model

        self.mlp = layers.MLP(2 * model.output_dim, [model.output_dim] * (num_mlp_layer - 1) + [1])

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)

        node_in, node_out = graph.edge_list[:, :2].t()
        indices = torch.arange(graph.num_edge, device=self.device)
        indices = functional.variadic_sample(indices, graph.num_edges, self.num_sample).flatten(-2, -1)
        node_i = node_in[indices]
        node_j = node_out[indices]
        graph = graph.edge_mask(~functional.as_mask(indices, graph.num_edge))

        # Calculate distance
        target = (graph.node_position[node_i] - graph.node_position[node_j]).norm(p=2, dim=-1)

        output = self.model(graph, graph.node_feature.float() , all_loss, metric)["node_feature"]
        node_feature = torch.cat([output[node_i], output[node_j]], dim=-1)
        pred = self.mlp(node_feature).squeeze(-1)

        return pred, target

    def evaluate(self, pred, target):
        metric = {}
        mse = F.mse_loss(pred, target)

        name = tasks._get_metric_name("mse")
        metric[name] = mse

        return metric

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(pred, target))

        loss = F.mse_loss(pred, target)
        name = tasks._get_criterion_name("mse")
        metric[name] = loss

        all_loss += loss

        return all_loss, metric


@R.register("tasks.AnglePrediction")
class AnglePrediction(tasks.Task, core.Configurable):
    """
    Angle prediction task proposed in `Protein Representation Learning by Geometric Structure Pretraining`_.

    .. _Protein Representation Learning by Geometric Structure Pretraining:
        https://arxiv.org/pdf/2203.06125.pdf

    Randomly select pairs of adjacent edges and predict the angles between them using the representations of three
    nodes. The selected edges are removed from the input graph to prevent trivial solutions.

    Parameters:
        model (nn.Module): node representation model
        num_sample (int, optional): number of edge pairs selected from each graph
        num_class (int, optional): number of classes to discretize the angles
        num_mlp_layer (int, optional): number of MLP layers in angle predictor
        graph_construction_model (nn.Module, optional): graph construction model
    """

    def __init__(self, model, num_sample=256, num_class=8, num_mlp_layer=2, graph_construction_model=None):
        super(AnglePrediction, self).__init__()
        self.model = model
        self.num_sample = num_sample
        self.num_mlp_layer = num_mlp_layer
        self.graph_construction_model = graph_construction_model

        boundary = torch.arange(0, math.pi, math.pi / num_class)
        self.register_buffer("boundary", boundary)

        self.mlp = layers.MLP(3 * model.output_dim, [model.output_dim] * (num_mlp_layer - 1) + [num_class])

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)

        node_in, node_out = graph.edge_list[:, :2].t()

        line_graph = graph.line_graph()
        edge_in, edge_out = line_graph.edge_list[:, :2].t()
        is_self_loop1 = (edge_in == edge_out)
        is_self_loop2 = (node_in[edge_in] == node_out[edge_out])
        is_remove = is_self_loop1 | is_self_loop2
        line_graph = line_graph.edge_mask(~is_remove)
        edge_in, edge_out = line_graph.edge_list[:, :2].t()
        # (k->j) - (j->i)
        node_i = node_out[edge_out]
        node_j = node_in[edge_out]
        node_k = node_in[edge_in]
        indices = torch.arange(line_graph.num_edge, device=self.device)
        indices = functional.variadic_sample(indices, line_graph.num_edges, self.num_sample).flatten(-2, -1)
        node_i = node_i[indices]
        node_j = node_j[indices]
        node_k = node_k[indices]

        mask = torch.ones((graph.num_edge,), device=graph.device, dtype=torch.bool)
        mask[edge_out[indices]] = 0
        mask[edge_in[indices]] = 0
        graph = graph.edge_mask(mask)

        # Calculate angles
        vector1 = graph.node_position[node_i] - graph.node_position[node_j]
        vector2 = graph.node_position[node_k] - graph.node_position[node_j]
        x = (vector1 * vector2).sum(dim=-1)
        y = torch.cross(vector1, vector2).norm(dim=-1)
        angle = torch.atan2(y, x)
        target = torch.bucketize(angle, self.boundary, right=True) - 1

        output = self.model(graph, graph.node_feature.float() , all_loss, metric)["node_feature"]
        node_feature = torch.cat([output[node_i], output[node_j], output[node_k]], dim=-1)
        pred = self.mlp(node_feature)

        return pred, target

    def evaluate(self, pred, target):
        metric = {}
        accuracy = (pred.argmax(dim=-1) == target).float().mean()

        name = tasks._get_metric_name("acc")
        metric[name] = accuracy

        return metric

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(pred, target))

        loss = F.cross_entropy(pred, target)
        name = tasks._get_criterion_name("ce")
        metric[name] = loss

        all_loss += loss

        return all_loss, metric


@R.register("tasks.DihedralPrediction")
class DihedralPrediction(tasks.Task, core.Configurable):
    """
    Dihedral prediction task proposed in `Protein Representation Learning by Geometric Structure Pretraining`_.

    .. _Protein Representation Learning by Geometric Structure Pretraining:
        https://arxiv.org/pdf/2203.06125.pdf

    Randomly select three consecutive edges and predict the dihedrals among them using the representations of four
    nodes. The selected edges are removed from the input graph to prevent trivial solutions.

    Parameters:
        model (nn.Module): node representation model
        num_sample (int, optional): number of edge triplets selected from each graph
        num_class (int, optional): number of classes for discretizing the dihedrals
        num_mlp_layer (int, optional): number of MLP layers in dihedral angle predictor
        graph_construction_model (nn.Module, optional): graph construction model
    """

    def __init__(self, model, num_sample=256, num_class=8, num_mlp_layer=2, graph_construction_model=None):
        super(DihedralPrediction, self).__init__()
        self.model = model
        self.num_sample = num_sample
        self.num_mlp_layer = num_mlp_layer
        self.graph_construction_model = graph_construction_model

        boundary = torch.arange(0, math.pi, math.pi / num_class)
        self.register_buffer("boundary", boundary)

        self.mlp = layers.MLP(4 * model.output_dim, [model.output_dim] * (num_mlp_layer - 1) + [num_class])

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)

        node_in, node_out = graph.edge_list[:, :2].t()
        line_graph = graph.line_graph()
        edge_in, edge_out = line_graph.edge_list[:, :2].t()
        is_self_loop1 = (edge_in == edge_out)
        is_self_loop2 = (node_in[edge_in] == node_out[edge_out])
        is_remove = is_self_loop1 | is_self_loop2
        line_graph = line_graph.edge_mask(~is_remove)
        edge_in, edge_out = line_graph.edge_list[:, :2].t()

        line2_graph = line_graph.line_graph()
        edge2_in, edge2_out = line2_graph.edge_list.t()[:2]
        is_self_loop1 = (edge2_in == edge2_out)
        is_self_loop2 = (edge_in[edge2_in] == edge_out[edge2_out])
        is_remove = is_self_loop1 | is_self_loop2
        line2_graph = line2_graph.edge_mask(~is_remove)
        edge2_in, edge2_out = line2_graph.edge_list[:, :2].t()
        # (k->t->j) - (t->j->i)
        node_i = node_out[edge_out[edge2_out]]
        node_j = node_in[edge_out[edge2_out]]
        node_t = node_in[edge_out[edge2_in]]
        node_k = node_in[edge_in[edge2_in]]
        indices = torch.arange(line2_graph.num_edge, device=self.device)
        indices = functional.variadic_sample(indices, line2_graph.num_edges, self.num_sample).flatten(-2, -1)
        node_i = node_i[indices]
        node_j = node_j[indices]
        node_t = node_t[indices]
        node_k = node_k[indices]
        mask = torch.ones((graph.num_edge,), device=graph.device, dtype=torch.bool)
        mask[edge_out[edge2_out[indices]]] = 0
        mask[edge_out[edge2_in[indices]]] = 0
        mask[edge_in[edge2_in[indices]]] = 0
        graph = graph.edge_mask(mask)

        v_ctr = graph.node_position[node_t] - graph.node_position[node_j]   # (A, 3)
        v1 = graph.node_position[node_i] - graph.node_position[node_j]
        v2 = graph.node_position[node_k] - graph.node_position[node_t]
        n1 = torch.cross(v_ctr, v1, dim=-1) # Normal vectors of the two planes
        n2 = torch.cross(v_ctr, v2, dim=-1)
        a = (n1 * n2).sum(dim=-1)
        b = torch.cross(n1, n2).norm(dim=-1)
        dihedral = torch.atan2(b, a)
        target = torch.bucketize(dihedral, self.boundary, right=True) - 1

        output = self.model(graph, graph.node_feature.float() , all_loss, metric)["node_feature"]
        node_feature = torch.cat([output[node_i], output[node_j], output[node_k], output[node_t]], dim=-1)
        pred = self.mlp(node_feature)

        return pred, target

    def evaluate(self, pred, target):
        metric = {}
        accuracy = (pred.argmax(dim=-1) == target).float().mean()

        name = tasks._get_metric_name("acc")
        metric[name] = accuracy

        return metric

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(pred, target))

        loss = F.cross_entropy(pred, target)
        name = tasks._get_criterion_name("ce")
        metric[name] = loss

        all_loss += loss

        return all_loss, metric
