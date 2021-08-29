from collections import deque

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data as torch_data
from torch_scatter import scatter_max, scatter_add

from torchdrug import core, tasks, data, metrics, transforms
from torchdrug.layers import functional
from torchdrug.core import Registry as R
from torchdrug import layers

import logging
logger = logging.getLogger(__name__)


@R.register("tasks.CenterIdentification")
class CenterIdentification(tasks.Task, core.Configurable):
    """
    Reaction center identification task.

    This class is a part of retrosynthesis prediction.

    Parameters:
        model (nn.Module): graph representation model
        feature (str or list of str, optional): additional features for prediction. Available features are
            reaction: type of the reaction
            graph: graph representation of the product
            atom: original atom feature
            bond: original bond feature
        num_mlp_layer (int, optional): number of MLP layers
    """

    _option_members = set(["feature"])

    def __init__(self, model, feature=("reaction", "graph", "atom", "bond"), num_mlp_layer=2):
        super(CenterIdentification, self).__init__()
        self.model = model
        self.num_mlp_layer = num_mlp_layer
        self.feature = feature

    def preprocess(self, train_set, valid_set, test_set):
        reaction_types = set()
        bond_types = set()
        for data in train_set:
            reaction_types.add(data["reaction"])
            for graph in data["graph"]:
                bond_types.update(graph.edge_list[:, 2].tolist())
        self.num_reaction = len(reaction_types)
        self.num_relation = len(bond_types)
        node_feature_dim = train_set[0]["graph"][0].node_feature.shape[-1]
        edge_feature_dim = train_set[0]["graph"][0].edge_feature.shape[-1]

        node_dim = self.model.output_dim
        edge_dim = 0
        graph_dim = 0
        for _feature in sorted(self.feature):
            if _feature == "reaction":
                graph_dim += self.num_reaction
            elif _feature == "graph":
                graph_dim += self.model.output_dim
            elif _feature == "atom":
                node_dim += node_feature_dim
            elif _feature == "bond":
                edge_dim += edge_feature_dim
            else:
                raise ValueError("Unknown feature `%s`" % _feature)

        node_dim += graph_dim # inherit graph features
        edge_dim += node_dim * 2 # inherit node features

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.edge_mlp = layers.MLP(edge_dim, hidden_dims + [1])
        self.node_mlp = layers.MLP(node_dim, hidden_dims + [1])

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)
        target = self.target(batch)
        metric.update(self.evaluate(pred, target))

        target, size = target
        target = functional.variadic_max(target, size)[1]
        loss = functional.variadic_cross_entropy(pred, target, size)

        name = tasks._get_criterion_name("ce")
        metric[name] = loss

        all_loss += loss

        return all_loss, metric

    def _collate(self, edge_data, node_data, graph):
        new_data = torch.zeros(len(edge_data) + len(node_data), *edge_data.shape[1:],
                               dtype=edge_data.dtype, device=edge_data.device)
        num_cum_xs = graph.num_cum_edges + graph.num_cum_nodes
        num_xs = graph.num_edges + graph.num_nodes
        starts = num_cum_xs - num_xs
        ends = starts + graph.num_edges
        index = functional.multi_slice_mask(starts, ends, num_cum_xs[-1])
        new_data[index] = edge_data
        new_data[~index] = node_data
        return new_data

    def target(self, batch):
        reactant, product = batch["graph"]
        graph = product.directed()

        target = self._collate(graph.edge_label, graph.node_label, graph)
        size = graph.num_edges + graph.num_nodes
        return target, size

    def predict(self, batch, all_loss=None, metric=None):
        reactant, product = batch["graph"]
        output = self.model(product, product.node_feature.float(), all_loss, metric)

        graph = product.directed()

        node_feature = [output["node_feature"]]
        edge_feature = []
        graph_feature = []
        for _feature in sorted(self.feature):
            if _feature == "reaction":
                reaction_feature = torch.zeros(len(graph), self.num_reaction, dtype=torch.float32, device=self.device)
                reaction_feature.scatter_(1, batch["reaction"].unsqueeze(-1), 1)
                graph_feature.append(reaction_feature)
            elif _feature == "graph":
                graph_feature.append(output["graph_feature"])
            elif _feature == "atom":
                node_feature.append(graph.node_feature.float())
            elif _feature == "bond":
                edge_feature.append(graph.edge_feature.float())
            else:
                raise ValueError("Unknown feature `%s`" % _feature)

        graph_feature = torch.cat(graph_feature, dim=-1)
        # inherit graph features
        node_feature.append(graph_feature[graph.node2graph])
        node_feature = torch.cat(node_feature, dim=-1)
        # inherit node features
        edge_feature.append(node_feature[graph.edge_list[:, :2]].flatten(1))
        edge_feature = torch.cat(edge_feature, dim=-1)
        
        edge_pred = self.edge_mlp(edge_feature).squeeze(-1)
        node_pred = self.node_mlp(node_feature).squeeze(-1)

        pred = self._collate(edge_pred, node_pred, graph)

        return pred

    def evaluate(self, pred, target):
        target, size = target

        metric = {}
        target = functional.variadic_max(target, size)[1]
        accuracy = metrics.variadic_accuracy(pred, target, size).mean()

        name = tasks._get_metric_name("acc")
        metric[name] = accuracy

        return metric

    @torch.no_grad()
    def predict_synthon(self, batch, k=1):
        """
        Predict top-k synthons from target molecules.

        Parameters:
            batch (dict): batch of target molecules
            k (int, optional): return top-k results

        Returns:
            list of dict: top k records.
                Each record is a batch dict of keys ``synthon``, ``num_synthon``, ``reaction_center``,
                ``log_likelihood`` and ``reaction``.
        """
        pred = self.predict(batch)
        target, size = self.target(batch)
        logp = functional.variadic_log_softmax(pred, size)

        reactant, product = batch["graph"]
        graph = product.directed()
        with graph.graph():
            graph.product_id = torch.arange(len(graph), device=self.device)

        graph = graph.repeat(k)
        order = torch.arange(len(graph), device=self.device)
        order = order.view(k, -1).t().flatten()
        graph = graph[order]
        with graph.graph():
            graph.split_id = torch.arange(k, device=self.device).repeat(len(graph) // k)
        reaction = batch["reaction"].repeat(k)[order]

        logp, center_topk = functional.variadic_topk(logp, size, k)
        logp = logp.flatten()
        center_topk = center_topk.flatten()

        is_edge = center_topk < graph.num_edges
        node_index = center_topk + graph.num_cum_nodes - graph.num_nodes - graph.num_edges
        edge_index = center_topk + graph.num_cum_edges - graph.num_edges
        center_topk_shifted = torch.cat([-torch.ones(1, dtype=torch.long, device=self.device),
                                         center_topk[:-1]])
        product_id_shifted = torch.cat([-torch.ones(1, dtype=torch.long, device=self.device),
                                         graph.product_id[:-1]])
        is_duplicate = (center_topk == center_topk_shifted) & (graph.product_id == product_id_shifted)
        node_index = node_index[~is_edge]
        edge_index = edge_index[is_edge]
        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)

        reaction_center = torch.zeros(len(graph), 2, dtype=torch.long, device=self.device)
        reaction_center[is_edge] = graph.atom_map[graph.edge_list[edge_index, :2]]
        reaction_center[~is_edge, 0] = graph.atom_map[node_index]

        # remove the edges from products
        graph = graph.edge_mask(edge_mask)
        graph = graph[~is_duplicate]
        reaction_center = reaction_center[~is_duplicate]
        logp = logp[~is_duplicate]
        reaction = reaction[~is_duplicate]
        synthon, num_synthon = graph.connected_components()
        synthon = synthon.undirected() # (< num_graph * k)

        result = {
            "synthon": synthon,
            "num_synthon": num_synthon,
            "reaction_center": reaction_center,
            "log_likelihood": logp,
            "reaction": reaction,
        }

        return result


class RandomBFSOrder(object):

    def __call__(self, item):
        assert hasattr(item["graph"][0], "reaction_center")
        reactant, synthon = item["graph"]

        edge_list = reactant.edge_list[:, :2].tolist()
        neighbor = [[] for _ in range(reactant.num_node)]
        for h, t in edge_list:
            neighbor[h].append(t)
        depth = [-1] * reactant.num_node

        # select a mapped atom as BFS root
        reactant2id = reactant.atom_map
        id2synthon = -torch.ones(synthon.atom_map.max() + 1, dtype=torch.long, device=synthon.device)
        id2synthon[synthon.atom_map] = torch.arange(synthon.num_node, device=synthon.device)
        reactant2synthon = id2synthon[reactant2id]

        candidate = (reactant2synthon != -1).nonzero().squeeze(-1)
        i = candidate[torch.randint(len(candidate), (1,))].item()

        queue = deque([i])
        depth[i] = 0
        order = []
        while queue:
            h = queue.popleft()
            order.append(h)
            for t in neighbor[h]:
                if depth[t] == -1:
                    depth[t] = depth[h] + 1
                    queue.append(t)

        reactant = reactant.subgraph(order)

        if reactant.num_edge > 0:
            node_index = reactant.edge_list[:, :2]
            node_large = node_index.max(dim=-1)[0]
            node_small = node_index.min(dim=-1)[0]
            undirected_edge_id = node_large * (node_large + 1) + node_small
            undirected_edge_id = undirected_edge_id * 2 + (node_index[:, 0] > node_index[:, 1])

            # rearrange edges into autoregressive order
            edge_order = undirected_edge_id.argsort()
            reactant = reactant.edge_mask(edge_order)

        assert hasattr(reactant, "reaction_center")

        item = item.copy()
        item["graph"] = (reactant, synthon)

        return item


@R.register("tasks.SynthonCompletion")
class SynthonCompletion(tasks.Task, core.Configurable):
    """
    Synthon completion task.

    This class is a part of retrosynthesis prediction.

    Parameters:
        model (nn.Module): graph representation model
        feature (str or list of str, optional): additional features for prediction. Available features are
            reaction: type of the reaction
            graph: graph representation of the synthon
            atom: original atom feature
        num_mlp_layer (int, optional): number of MLP layers
    """

    _option_members = set(["feature"])

    def __init__(self, model, feature=("reaction", "graph", "atom"), num_mlp_layer=2):
        super(SynthonCompletion, self).__init__()
        self.model = model
        self.num_mlp_layer = num_mlp_layer
        self.feature = feature
        self.input_linear = nn.Linear(2, self.model.input_dim)

    def preprocess(self, train_set, valid_set, test_set):
        reaction_types = set()
        for data in train_set:
            reaction_types.add(data["reaction"])
        self.num_reaction = len(reaction_types)

        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        dataset.transform = transforms.Compose([
            dataset.transform,
            RandomBFSOrder(),
        ])

        # atom_types = set()
        # bond_types = set()
        # for data in train_set:
        #     for graph in data["graph"]:
        #         atom_types.update(graph.atom_type.tolist())
        #         bond_types.update(graph.edge_list[:, 2].tolist())
        # atom_types = torch.tensor(sorted(atom_types))

        # TODO: only for fast debugging, to remove
        atom_types = torch.tensor([5, 6, 7, 8, 9, 12, 14, 15, 16, 17, 29, 30, 34, 35, 50, 53])
        bond_types = torch.tensor([0, 1, 2])

        atom2id = -torch.ones(atom_types.max() + 1, dtype=torch.long)
        atom2id[atom_types] = torch.arange(len(atom_types))
        self.register_buffer("id2atom", atom_types)
        self.register_buffer("atom2id", atom2id)
        self.num_atom_type = len(atom_types)
        self.num_bond_type = len(bond_types)
        node_feature_dim = train_set[0]["graph"][0].node_feature.shape[-1]

        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.dataset_kwargs = dataset.config_dict().get("kwargs")

        node_dim = self.model.output_dim
        edge_dim = 0
        graph_dim = 0
        for _feature in sorted(self.feature):
            if _feature == "reaction":
                graph_dim += self.num_reaction
            elif _feature == "graph":
                graph_dim += self.model.output_dim
            elif _feature == "atom":
                node_dim += node_feature_dim
            else:
                raise ValueError("Unknown feature `%s`" % _feature)

        self.new_atom_feature = nn.Embedding(self.num_atom_type, node_dim)

        node_dim += graph_dim  # inherit graph features
        edge_dim += node_dim * 2  # inherit node features

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.node_in_mlp = layers.MLP(node_dim, hidden_dims + [1])
        self.node_out_mlp = layers.MLP(edge_dim, hidden_dims + [1])
        self.edge_mlp = layers.MLP(edge_dim, hidden_dims + [1])
        self.bond_mlp = layers.MLP(edge_dim, hidden_dims + [self.num_bond_type])
        self.stop_mlp = layers.MLP(graph_dim, hidden_dims + [1])

    def _update_molecule_feature(self, graphs):
        # This function is very slow
        graphs = graphs.ion_to_molecule()
        mols = graphs.to_molecule(ignore_error=True)
        valid = [mol is not None for mol in mols]
        valid = torch.tensor(valid, device=graphs.device)
        new_graphs = type(graphs).from_molecule(mols, node_feature="synthon_completion", kekulize=True)

        node_feature = torch.zeros(graphs.num_node, *new_graphs.node_feature.shape[1:],
                                   dtype=new_graphs.node_feature.dtype, device=graphs.device)
        edge_feature = torch.zeros(graphs.num_edge, *new_graphs.edge_feature.shape[1:],
                                   dtype=new_graphs.edge_feature.dtype, device=graphs.device)
        bond_type = torch.zeros_like(graphs.bond_type)
        node_mask = valid[graphs.node2graph]
        edge_mask = valid[graphs.edge2graph]
        node_feature[node_mask] = new_graphs.node_feature.to(device=graphs.device)
        edge_feature[edge_mask] = new_graphs.edge_feature.to(device=graphs.device)
        bond_type[edge_mask] = new_graphs.bond_type.to(device=graphs.device)

        with graphs.node():
            graphs.node_feature = node_feature
        with graphs.edge():
            graphs.edge_feature = edge_feature
            graphs.bond_type = bond_type

        return graphs, valid

    @torch.no_grad()
    def _all_prefix_slice(self, num_xs, lengths=None):
        # extract a bunch of slices that correspond to the following num_repeat * n masks
        # ------ repeat 0 -----
        # graphs[0]: [0, 0, ..., 0]
        # ...
        # graphs[-1]: [0, 0, ..., 0]
        # ------ repeat 1 -----
        # graphs[0]: [1, 0, ..., 0]
        # ...
        # graphs[-1]: [1, 0, ..., 0]
        # ...
        # ------ repeat -1 -----
        # graphs[0]: [1, ..., 1, 0]
        # ...
        # graphs[-1]: [1, ..., 1, 0]
        num_cum_xs = num_xs.cumsum(0)
        starts = num_cum_xs - num_xs
        if lengths is None:
            num_max_x = num_xs.max().item()
            lengths = torch.arange(0, num_max_x, 2, device=num_xs.device)

        pack_offsets = torch.arange(len(lengths), device=num_xs.device) * num_cum_xs[-1]
        # starts, lengths, ends: (num_repeat, num_graph)
        starts = starts.unsqueeze(0) + pack_offsets.unsqueeze(-1)
        valid = lengths.unsqueeze(-1) <= num_xs.unsqueeze(0) - 2
        lengths = torch.min(lengths.unsqueeze(-1), num_xs.unsqueeze(0) - 2).clamp(0)
        ends = starts + lengths

        starts = starts.flatten()
        ends = ends.flatten()
        valid = valid.flatten()

        return starts, ends, valid

    @torch.no_grad()
    def _get_reaction_feature(self, reactant, synthon):

        def get_edge_map(graph, num_nodes):
            node_in, node_out = graph.edge_list.t()[:2]
            node_in2id = graph.atom_map[node_in]
            node_out2id = graph.atom_map[node_out]
            edge_map = node_in2id * num_nodes[graph.edge2graph] + node_out2id
            # edges containing any unmapped node is considered to be unmapped
            edge_map[(node_in2id == 0) | (node_out2id == 0)] = 0
            return edge_map

        def get_mapping(reactant_x, synthon_x, reactant_x2graph, synthon_x2graph):
            num_xs = scatter_max(reactant_x, reactant_x2graph)[0]
            num_xs = num_xs.clamp(0) + 1
            num_cum_xs = num_xs.cumsum(0)
            offset = num_cum_xs - num_xs
            reactant2id = reactant_x + offset[reactant_x2graph]
            synthon2id = synthon_x + offset[synthon_x2graph]
            assert synthon2id.min() > 0
            id2synthon = -torch.ones(num_cum_xs[-1], dtype=torch.long, device=self.device)
            id2synthon[synthon2id] = torch.arange(len(synthon2id), device=self.device)
            reactant2synthon = id2synthon[reactant2id]

            return reactant2synthon

        # reactant & synthon may have different number of nodes
        # reactant.num_nodes >= synthon.num_nodes
        assert (reactant.num_nodes >= synthon.num_nodes).all()
        reactant_edge_map = get_edge_map(reactant, reactant.num_nodes)
        synthon_edge_map = get_edge_map(synthon, reactant.num_nodes)

        node_r2s = get_mapping(reactant.atom_map, synthon.atom_map, reactant.node2graph, synthon.node2graph)
        edge_r2s = get_mapping(reactant_edge_map, synthon_edge_map, reactant.edge2graph, synthon.edge2graph)

        is_new_node = node_r2s == -1
        is_new_edge = edge_r2s == -1
        is_modified_edge = (edge_r2s != -1) & (reactant.bond_type != synthon.bond_type[edge_r2s])
        is_reaction_center = (reactant.atom_map > 0) & \
                             (reactant.atom_map.unsqueeze(-1) ==
                              reactant.reaction_center[reactant.node2graph]).any(dim=-1)

        return node_r2s, edge_r2s, is_new_node, is_new_edge, is_modified_edge, is_reaction_center

    @torch.no_grad()
    def all_edge(self, reactant, synthon):
        graph = reactant.clone()
        node_r2s, edge_r2s, is_new_node, is_new_edge, is_modified_edge, is_reaction_center = \
            self._get_reaction_feature(reactant, synthon)
        with graph.node():
            graph.node_r2s = node_r2s
            graph.is_new_node = is_new_node
            graph.is_reaction_center = is_reaction_center
        with graph.edge():
            graph.edge_r2s = edge_r2s
            graph.is_new_edge = is_new_edge
            graph.is_modified_edge = is_modified_edge

        starts, ends, valid = self._all_prefix_slice(reactant.num_edges)
        num_repeat = len(starts) // len(reactant)
        graph = graph.repeat(num_repeat)

        # autoregressive condition range for each sample
        condition_mask = functional.multi_slice_mask(starts, ends, graph.num_edge)
        # special case: end == graph.num_edge. In this case, valid is always false
        assert ends.max() <= graph.num_edge
        ends = ends.clamp(0, graph.num_edge - 1)
        node_in, node_out, bond_target = graph.edge_list[ends].t()
        # modified edges which don't appear in conditions should keep their old bond types
        # i.e. bond types in synthons
        unmodified = ~condition_mask & graph.is_modified_edge
        unmodified = unmodified.nonzero().squeeze(-1)
        assert not (graph.bond_type[unmodified] == synthon.bond_type[graph.edge_r2s[unmodified]]).any()
        graph.edge_list[unmodified, 2] = synthon.edge_list[graph.edge_r2s[unmodified], 2]

        reverse_target = graph.edge_list[ends][:, [1, 0, 2]]
        is_reverse_target = (graph.edge_list == reverse_target[graph.edge2graph]).all(dim=-1)
        # keep edges that exist in the synthon
        # remove the reverse of new target edges
        edge_mask = (condition_mask & ~is_reverse_target) | ~graph.is_new_edge

        atom_in = graph.atom_type[node_in]
        atom_out = graph.atom_type[node_out]
        # keep one supervision for undirected edges
        # remove samples that try to predict existing edges
        valid &= (node_in < node_out) & (graph.is_new_edge[ends] | graph.is_modified_edge[ends])
        graph = graph.edge_mask(edge_mask)

        # sanitize the molecules
        # this will change atom index, so we manually remap the target nodes
        compact_mapping = -torch.ones(graph.num_node, dtype=torch.long, device=self.device)
        node_mask = graph.degree_in + graph.degree_out > 0
        # special case: for graphs without any edge, the first node should be kept
        index = torch.arange(graph.num_node, device=self.device)
        single_node_mask = (graph.num_edges == 0)[graph.node2graph] & \
                           (index == (graph.num_cum_nodes - graph.num_nodes)[graph.node2graph])
        node_index = (node_mask | single_node_mask).nonzero().squeeze(-1)
        compact_mapping[node_index] = torch.arange(len(node_index), device=self.device)
        node_in = compact_mapping[node_in]
        node_out = compact_mapping[node_out]
        graph = graph.subgraph(node_index)

        node_in_target = node_in - graph.num_cum_nodes + graph.num_nodes
        assert (node_in_target[valid] < graph.num_nodes[valid]).all() and (node_in_target[valid] >= 0).all()
        # node2 might be a new node
        node_out_target = torch.where(node_out == -1, self.atom2id[atom_out] + graph.num_nodes,
                                  node_out - graph.num_cum_nodes + graph.num_nodes)
        stop_target = torch.zeros(len(node_in_target), device=self.device)

        graph = graph[valid]
        node_in_target = node_in_target[valid]
        node_out_target = node_out_target[valid]
        bond_target = bond_target[valid]
        stop_target = stop_target[valid]

        assert (graph.num_edges % 2 == 0).all()
        # node / edge features may change because we mask some nodes / edges
        graph, feature_valid = self._update_molecule_feature(graph)

        return graph[feature_valid], node_in_target[feature_valid], node_out_target[feature_valid], \
               bond_target[feature_valid], stop_target[feature_valid]

    @torch.no_grad()
    def all_stop(self, reactant, synthon):
        graph = reactant.clone()
        node_r2s, edge_r2s, is_new_node, is_new_edge, is_modified_edge, is_reaction_center = \
            self._get_reaction_feature(reactant, synthon)
        with graph.node():
            graph.node_r2s = node_r2s
            graph.is_new_node = is_new_node
            graph.is_reaction_center = is_reaction_center
        with graph.edge():
            graph.edge_r2s = edge_r2s
            graph.is_new_edge = is_new_edge
            graph.is_modified_edge = is_modified_edge

        node_in_target = torch.zeros(len(graph), dtype=torch.long, device=self.device)
        node_out_target = torch.zeros_like(node_in_target)
        bond_target = torch.zeros_like(node_in_target)
        stop_target = torch.ones(len(graph), device=self.device)

        # keep consistent with other training data
        graph, feature_valid = self._update_molecule_feature(graph)

        return graph[feature_valid], node_in_target[feature_valid], node_out_target[feature_valid], \
               bond_target[feature_valid], stop_target[feature_valid]

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        node_in_pred, node_out_pred, bond_pred, stop_pred = pred
        node_in_target, node_out_target, bond_target, stop_target, size = target

        loss = functional.variadic_cross_entropy(node_in_pred, node_in_target, size, reduction="none")
        loss = functional.masked_mean(loss, stop_target == 0)
        metric["node in ce loss"] = loss
        all_loss += loss

        loss = functional.variadic_cross_entropy(node_out_pred, node_out_target, size, reduction="none")
        loss = functional.masked_mean(loss, stop_target == 0)
        metric["node out ce loss"] = loss
        all_loss += loss

        loss = F.cross_entropy(bond_pred, bond_target, reduction="none")
        loss = functional.masked_mean(loss, stop_target == 0)
        metric["bond ce loss"] = loss
        all_loss += loss

        # Do we need to balance stop pred?
        loss = F.binary_cross_entropy_with_logits(stop_pred, stop_target)
        metric["stop bce loss"] = loss
        all_loss += loss

        metric["total loss"] = all_loss
        metric.update(self.evaluate(pred, target))

        return all_loss, metric

    def evaluate(self, pred, target):
        node_in_pred, node_out_pred, bond_pred, stop_pred = pred
        node_in_target, node_out_target, bond_target, stop_target, size = target

        metric = {}

        node_in_acc = metrics.variadic_accuracy(node_in_pred, node_in_target, size)
        accuracy = functional.masked_mean(node_in_acc, stop_target == 0)
        metric["node in accuracy"] = accuracy

        node_out_acc = metrics.variadic_accuracy(node_out_pred, node_out_target, size)
        accuracy = functional.masked_mean(node_out_acc, stop_target == 0)
        metric["node out accuracy"] = accuracy

        bond_acc = (bond_pred.argmax(-1) == bond_target).float()
        accuracy = functional.masked_mean(bond_acc, stop_target == 0)
        metric["bond accuracy"] = accuracy

        stop_acc = ((stop_pred > 0.5) == (stop_target > 0.5)).float()
        metric["stop accuracy"] = stop_acc.mean()

        total_acc = (node_in_acc > 0.5) & (node_out_acc > 0.5) & (bond_acc > 0.5) & (stop_acc > 0.5)
        total_acc = torch.where(stop_target == 0, total_acc, stop_acc > 0.5).float()
        metric["total accuracy"] = total_acc.mean()

        return metric

    def _cat(self, graphs):
        for i, graph in enumerate(graphs):
            if not isinstance(graph, data.PackedGraph):
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

    def target(self, batch):
        reactant, synthon = batch["graph"]

        graph1, node_in_target1, node_out_target1, bond_target1, stop_target1 = self.all_edge(reactant, synthon)
        graph2, node_in_target2, node_out_target2, bond_target2, stop_target2 = self.all_stop(reactant, synthon)

        node_in_target = torch.cat([node_in_target1, node_in_target2])
        node_out_target = torch.cat([node_out_target1, node_out_target2])
        bond_target = torch.cat([bond_target1, bond_target2])
        stop_target = torch.cat([stop_target1, stop_target2])
        size = torch.cat([graph1.num_nodes, graph2.num_nodes])
        # add new atom candidates into the size of each graph
        size_ext = size + self.num_atom_type

        return node_in_target, node_out_target, bond_target, stop_target, size_ext

    def _topk_action(self, graph, k):
        synthon_feature = torch.stack([graph.is_new_node, graph.is_reaction_center], dim=-1).float()
        node_feature = graph.node_feature.float() + self.input_linear(synthon_feature)
        output = self.model(graph, node_feature)

        node_feature = [output["node_feature"]]
        graph_feature = []
        for _feature in sorted(self.feature):
            if _feature == "reaction":
                reaction_feature = torch.zeros(len(graph), self.num_reaction, dtype=torch.float32, device=self.device)
                reaction_feature.scatter_(1, graph.reaction.unsqueeze(-1), 1)
                graph_feature.append(reaction_feature)
            elif _feature == "graph":
                graph_feature.append(output["graph_feature"])
            elif _feature == "atom":
                node_feature.append(graph.node_feature.float())
            else:
                raise ValueError("Unknown feature `%s`" % _feature)

        graph_feature = torch.cat(graph_feature, dim=-1)
        # inherit graph features
        node_feature.append(graph_feature[graph.node2graph])
        node_feature = torch.cat(node_feature, dim=-1)

        new_node_feature = self.new_atom_feature.weight.repeat(len(graph), 1)
        new_graph_feature = graph_feature.unsqueeze(1).repeat(1, self.num_atom_type, 1).flatten(0, 1)
        new_node_feature = torch.cat([new_node_feature, new_graph_feature], dim=-1)
        node_feature, num_nodes_ext = self._extend(node_feature, graph.num_nodes, new_node_feature)

        node2graph_ext = functional._size_to_index(num_nodes_ext)
        num_cum_nodes_ext = num_nodes_ext.cumsum(0)
        starts = num_cum_nodes_ext - num_nodes_ext + graph.num_nodes
        ends = num_cum_nodes_ext
        is_new_node = functional.multi_slice_mask(starts, ends, num_cum_nodes_ext[-1])
        infinity = torch.tensor(float("inf"), device=self.device)

        node_in_pred = self.node_in_mlp(node_feature).squeeze(-1)
        stop_pred = self.stop_mlp(graph_feature).squeeze(-1)

        # mask out node-in prediction on new atoms
        node_in_pred[is_new_node] = -infinity
        node_in_logp = functional.variadic_log_softmax(node_in_pred, num_nodes_ext) # (num_node,)
        stop_logp = F.logsigmoid(stop_pred)
        act_logp = F.logsigmoid(-stop_pred)
        node_in_topk = functional.variadic_topk(node_in_logp, num_nodes_ext, k)[1]
        assert (node_in_topk >= 0).all() and (node_in_topk < num_nodes_ext.unsqueeze(-1)).all()
        node_in = node_in_topk + (num_cum_nodes_ext - num_nodes_ext).unsqueeze(-1) # (num_graph, k)

        # (num_node, node_in_k, feature_dim)
        node_out_feature = torch.cat([node_feature[node_in][node2graph_ext],
                                      node_feature.unsqueeze(1).expand(-1, k, -1)], dim=-1)
        node_out_pred = self.node_out_mlp(node_out_feature).squeeze(-1)
        # mask out node-out prediction on self-loops
        node_out_pred.scatter_(0, node_in, -infinity)
        # (num_node, node_in_k)
        node_out_logp = functional.variadic_log_softmax(node_out_pred, num_nodes_ext)
        # (num_graph, node_out_k, node_in_k)
        node_out_topk = functional.variadic_topk(node_out_logp, num_nodes_ext, k)[1]
        assert (node_out_topk >= 0).all() and (node_out_topk < num_nodes_ext.view(-1, 1, 1)).all()
        node_out = node_out_topk + (num_cum_nodes_ext - num_nodes_ext).view(-1, 1, 1)

        # (num_graph, node_out_k, node_in_k, feature_dim * 2)
        edge = torch.stack([node_in.unsqueeze(1).expand_as(node_out), node_out], dim=-1)
        bond_feature = node_feature[edge].flatten(-2)
        bond_pred = self.bond_mlp(bond_feature).squeeze(-1)
        bond_logp = F.log_softmax(bond_pred, dim=-1) # (num_graph, node_out_k, node_in_k, num_relation)
        bond_type = torch.arange(bond_pred.shape[-1], device=self.device)
        bond_type = bond_type.view(1, 1, 1, -1).expand_as(bond_logp)

        # (num_graph, node_out_k, node_in_k, num_relation)
        node_in_logp = node_in_logp.gather(0, node_in.flatten(0, 1)).view(-1, 1, k, 1)
        node_out_logp = node_out_logp.gather(0, node_out.flatten(0, 1)).view(-1, k, k, 1)
        act_logp = act_logp.view(-1, 1, 1, 1)
        logp = node_in_logp + node_out_logp + bond_logp + act_logp

        # (num_graph, node_out_k, node_in_k, num_relation, 4)
        node_in_topk = node_in_topk.view(-1, 1, k, 1).expand_as(logp)
        node_out_topk = node_out_topk.view(-1, k, k, 1).expand_as(logp)
        action = torch.stack([node_in_topk, node_out_topk, bond_type, torch.zeros_like(bond_type)], dim=-1)

        # add stop action
        logp = torch.cat([logp.flatten(1), stop_logp.unsqueeze(-1)], dim=1)
        stop = torch.tensor([0, 0, 0, 1], device=self.device)
        stop = stop.view(1, 1, -1).expand(len(graph), -1, -1)
        action = torch.cat([action.flatten(1, -2), stop], dim=1)
        topk = logp.topk(k, dim=-1)[1]

        return action.gather(1, topk.unsqueeze(-1).expand(-1, -1, 4)), logp.gather(1, topk)

    def _apply_action(self, graph, action, logp):
        # only support non-variadic k-actions
        assert len(graph) == len(action)
        num_action = action.shape[1]

        graph = graph.repeat(num_action)
        order = torch.arange(len(graph), device=self.device)
        order = order.view(num_action, -1).t().flatten()
        graph = graph[order]

        action = action.flatten(0, 1) # (num_graph * k, 4)
        logp = logp.flatten(0, 1) # (num_graph * k)
        new_node_in, new_node_out, new_bond_type, stop = action.t()

        # add new nodes
        has_new_node = (new_node_out >= graph.num_nodes) & (stop == 0)
        new_atom_id = (new_node_out - graph.num_nodes)[has_new_node]
        new_atom_type = self.id2atom[new_atom_id]
        is_new_node = torch.ones(len(new_atom_type), dtype=torch.bool, device=self.device)
        is_reaction_center = torch.zeros(len(new_atom_type), dtype=torch.bool, device=self.device)
        atom_type, num_nodes = functional._extend(graph.atom_type, graph.num_nodes, new_atom_type, has_new_node)
        is_new_node = functional._extend(graph.is_new_node, graph.num_nodes, is_new_node, has_new_node)[0]
        is_reaction_center = functional._extend(graph.is_reaction_center, graph.num_nodes, is_reaction_center, has_new_node)[0]

        # cast to regular node ids
        new_node_out = torch.where(has_new_node, graph.num_nodes, new_node_out)

        # modify edges
        new_edge = torch.stack([new_node_in, new_node_out], dim=-1)
        edge_list = graph.edge_list.clone()
        bond_type = graph.bond_type.clone()
        edge_list[:, :2] -= graph._offsets.unsqueeze(-1)
        is_modified_edge = (edge_list[:, :2] == new_edge[graph.edge2graph]).all(dim=-1) & \
                           (stop[graph.edge2graph] == 0)
        has_modified_edge = scatter_max(is_modified_edge.long(), graph.edge2graph, dim_size=len(graph))[0] > 0
        bond_type[is_modified_edge] = new_bond_type[has_modified_edge]
        edge_list[is_modified_edge, 2] = new_bond_type[has_modified_edge]
        # modify reverse edges
        new_edge = new_edge.flip(-1)
        is_modified_edge = (edge_list[:, :2] == new_edge[graph.edge2graph]).all(dim=-1) & \
                           (stop[graph.edge2graph] == 0)
        bond_type[is_modified_edge] = new_bond_type[has_modified_edge]
        edge_list[is_modified_edge, 2] = new_bond_type[has_modified_edge]

        # add new edges
        has_new_edge = (~has_modified_edge) & (stop == 0)
        new_edge_list = torch.stack([new_node_in, new_node_out, new_bond_type], dim=-1)[has_new_edge]
        bond_type = functional._extend(bond_type, graph.num_edges, new_bond_type[has_new_edge], has_new_edge)[0]
        edge_list, num_edges = functional._extend(edge_list, graph.num_edges, new_edge_list, has_new_edge)
        # add reverse edges
        new_edge_list = torch.stack([new_node_out, new_node_in, new_bond_type], dim=-1)[has_new_edge]
        bond_type = functional._extend(bond_type, num_edges, new_bond_type[has_new_edge], has_new_edge)[0]
        edge_list, num_edges = functional._extend(edge_list, num_edges, new_edge_list, has_new_edge)

        logp = logp + graph.logp

        # inherit attributes
        data_dict = graph.data_dict
        meta_dict = graph.meta_dict
        for key in ["atom_type", "bond_type", "is_new_node", "is_reaction_center", "logp"]:
            data_dict.pop(key)
        # pad 0 for node / edge attributes
        for k, v in data_dict.items():
            if meta_dict[k] == "node":
                shape = (len(new_atom_type), *v.shape[1:])
                new_data = torch.zeros(shape, dtype=v.dtype, device=self.device)
                data_dict[k] = functional._extend(v, graph.num_nodes, new_data, has_new_node)[0]
            if meta_dict[k] == "edge":
                shape = (len(new_edge_list) * 2, *v.shape[1:])
                new_data = torch.zeros(shape, dtype=v.dtype, device=self.device)
                data_dict[k] = functional._extend(v, graph.num_edges, new_data, has_new_edge * 2)[0]

        new_graph = type(graph)(edge_list, atom_type=atom_type, bond_type=bond_type, num_nodes=num_nodes,
                                num_edges=num_edges, num_relation=graph.num_relation,
                                is_new_node=is_new_node, is_reaction_center=is_reaction_center, logp=logp,
                                meta_dict=meta_dict, **data_dict)
        with new_graph.graph():
            new_graph.is_stopped = stop == 1
        valid = logp > float("-inf")
        new_graph = new_graph[valid]

        new_graph, feature_valid = self._update_molecule_feature(new_graph)
        return new_graph[feature_valid]

    @torch.no_grad()
    def predict_reactant(self, batch, num_beam=10, max_prediction=20, max_step=20):
        if "synthon" in batch:
            synthon = batch["synthon"]
            synthon2product = functional._size_to_index(batch["num_synthon"])
            assert (synthon2product < len(batch["reaction"])).all()
            reaction = batch["reaction"][synthon2product]
        else:
            reactant, synthon = batch["graph"]
            reaction = batch["reaction"]

        # In any case, ensure that the synthon is a molecule rather than an ion
        # This is consistent across train/test routines in synthon completion
        synthon, feature_valid = self._update_molecule_feature(synthon)
        synthon = synthon[feature_valid]
        reaction = reaction[feature_valid]

        graph = synthon
        with graph.graph():
            # for convenience, because we need to manipulate graph a lot
            graph.reaction = reaction
            graph.synthon_id = torch.arange(len(graph), device=graph.device)
            if not hasattr(graph, "logp"):
                graph.logp = torch.zeros(len(graph), device=graph.device)
        with graph.node():
            graph.is_new_node = torch.zeros(graph.num_node, dtype=torch.bool, device=graph.device)
            graph.is_reaction_center = (graph.atom_map > 0) & \
                                       (graph.atom_map.unsqueeze(-1) ==
                                        graph.reaction_center[graph.node2graph]).any(dim=-1)

        result = []
        num_prediction = torch.zeros(len(synthon), dtype=torch.long, device=self.device)
        for i in range(max_step):
            logger.warning("action step: %d" % i)
            logger.warning("batched beam size: %d" % len(graph))
            # each candidate has #beam actions
            action, logp = self._topk_action(graph, num_beam)

            # each candidate is expanded to at most #beam (depending on validity) new candidates
            new_graph = self._apply_action(graph, action, logp)
            # assert (new_graph[is_stopped].logp > float("-inf")).all()
            offset = -2 * (new_graph.logp.max() - new_graph.logp.min())
            key = new_graph.synthon_id * offset + new_graph.logp
            order = key.argsort(descending=True)
            new_graph = new_graph[order]

            num_candidate = scatter_add(torch.ones_like(new_graph.synthon_id), new_graph.synthon_id,
                                        dim_size=len(synthon))
            topk = functional.variadic_topk(new_graph.logp, num_candidate, num_beam)[1]
            topk_index = topk + (num_candidate.cumsum(0) - num_candidate).unsqueeze(-1)
            topk_index = torch.unique(topk_index)
            new_graph = new_graph[topk_index]
            result.append(new_graph[new_graph.is_stopped])
            num_added = scatter_add(new_graph.is_stopped.long(), new_graph.synthon_id, dim_size=len(synthon))
            num_prediction += num_added

            # remove samples that already hit max prediction
            is_continue = (~new_graph.is_stopped) & (num_prediction[new_graph.synthon_id] < max_prediction)
            graph = new_graph[is_continue]
            if len(graph) == 0:
                break

        result = self._cat(result)
        # sort by synthon id
        order = result.synthon_id.argsort()
        result = result[order]

        # remove duplicate predictions
        is_duplicate = []
        synthon_id = -1
        for graph in result:
            if graph.synthon_id != synthon_id:
                synthon_id = graph.synthon_id
                smiles_set = set()
            smiles = graph.to_smiles(isomeric=False, atom_map=False, canonical=True)
            is_duplicate.append(smiles in smiles_set)
            smiles_set.add(smiles)
        is_duplicate = torch.tensor(is_duplicate, device=self.device)
        result = result[~is_duplicate]
        num_prediction = torch.bincount(result.synthon_id)

        # remove extra predictions
        topk = functional.variadic_topk(result.logp, num_prediction, max_prediction)[1]
        topk_index = topk + (num_prediction.cumsum(0) - num_prediction).unsqueeze(-1)
        topk_index = topk_index.flatten(0)
        topk_index_shifted = torch.cat([-torch.ones(1, dtype=torch.long, device=self.device), topk_index[:-1]])
        is_duplicate = topk_index == topk_index_shifted
        result = result[topk_index[~is_duplicate]]

        return result # (< num_graph * max_prediction)

    def _extend(self, data, num_xs, input, input2graph=None):
        if input2graph is None:
            num_input_per_graph = len(input) // len(num_xs)
            input2graph = torch.arange(len(num_xs), device=data.device).unsqueeze(-1)
            input2graph = input2graph.repeat(1, num_input_per_graph).flatten()
        num_inputs = scatter_add(torch.ones_like(input2graph), input2graph, dim_size=len(num_xs))
        new_num_xs = num_xs + num_inputs
        new_num_cum_xs = new_num_xs.cumsum(0)
        new_num_x = new_num_cum_xs[-1].item()
        new_data = torch.zeros(new_num_x, *data.shape[1:], dtype=data.dtype, device=data.device)
        starts = new_num_cum_xs - new_num_xs
        ends = starts + num_xs
        index = functional.multi_slice_mask(starts, ends, new_num_x)
        new_data[index] = data
        new_data[~index] = input
        return new_data, new_num_xs

    def predict_and_target(self, batch, all_loss=None, metric=None):
        reactant, synthon = batch["graph"]
        reactant = reactant.clone()
        with reactant.graph():
            reactant.reaction = batch["reaction"]

        graph1, node_in_target1, node_out_target1, bond_target1, stop_target1 = self.all_edge(reactant, synthon)
        graph2, node_in_target2, node_out_target2, bond_target2, stop_target2 = self.all_stop(reactant, synthon)

        graph = self._cat([graph1, graph2])

        node_in_target = torch.cat([node_in_target1, node_in_target2])
        node_out_target = torch.cat([node_out_target1, node_out_target2])
        bond_target = torch.cat([bond_target1, bond_target2])
        stop_target = torch.cat([stop_target1, stop_target2])
        size = graph.num_nodes
        # add new atom candidates into the size of each graph
        size_ext = size + self.num_atom_type

        synthon_feature = torch.stack([graph.is_new_node, graph.is_reaction_center], dim=-1).float()
        node_feature = graph.node_feature.float() + self.input_linear(synthon_feature)
        output = self.model(graph, node_feature, all_loss, metric)

        node_feature = [output["node_feature"]]
        graph_feature = []
        for _feature in sorted(self.feature):
            if _feature == "reaction":
                reaction_feature = torch.zeros(len(graph), self.num_reaction, dtype=torch.float32, device=self.device)
                reaction_feature.scatter_(1, graph.reaction.unsqueeze(-1), 1)
                graph_feature.append(reaction_feature)
            elif _feature == "graph":
                graph_feature.append(output["graph_feature"])
            elif _feature == "atom":
                node_feature.append(graph.node_feature)
            else:
                raise ValueError("Unknown feature `%s`" % _feature)

        graph_feature = torch.cat(graph_feature, dim=-1)
        # inherit graph features
        node_feature.append(graph_feature[graph.node2graph])
        node_feature = torch.cat(node_feature, dim=-1)

        new_node_feature = self.new_atom_feature.weight.repeat(len(graph), 1)
        new_graph_feature = graph_feature.unsqueeze(1).repeat(1, self.num_atom_type, 1).flatten(0, 1)
        new_node_feature = torch.cat([new_node_feature, new_graph_feature], dim=-1)
        node_feature, num_nodes_ext = self._extend(node_feature, graph.num_nodes, new_node_feature)
        assert (num_nodes_ext == size_ext).all()

        node2graph_ext = functional._size_to_index(num_nodes_ext)
        num_cum_nodes_ext = num_nodes_ext.cumsum(0)
        starts = num_cum_nodes_ext - num_nodes_ext + graph.num_nodes
        ends = num_cum_nodes_ext
        is_new_node = functional.multi_slice_mask(starts, ends, num_cum_nodes_ext[-1])

        node_in = node_in_target + num_cum_nodes_ext - num_nodes_ext
        node_out = node_out_target + num_cum_nodes_ext - num_nodes_ext
        edge = torch.stack([node_in, node_out], dim=-1)

        node_out_feature = torch.cat([node_feature[node_in][node2graph_ext], node_feature], dim=-1)
        bond_feature = node_feature[edge].flatten(-2)
        node_in_pred = self.node_in_mlp(node_feature).squeeze(-1)
        node_out_pred = self.node_out_mlp(node_out_feature).squeeze(-1)
        bond_pred = self.bond_mlp(bond_feature).squeeze(-1)
        stop_pred = self.stop_mlp(graph_feature).squeeze(-1)

        infinity = torch.tensor(float("inf"), device=self.device)
        # mask out node-in prediction on new atoms
        node_in_pred[is_new_node] = -infinity
        # mask out node-out prediction on self-loops
        node_out_pred[node_in] = -infinity

        return (node_in_pred, node_out_pred, bond_pred, stop_pred), \
               (node_in_target, node_out_target, bond_target, stop_target, size_ext)


@R.register("tasks.Retrosynthesis")
class Retrosynthesis(tasks.Task, core.Configurable):
    """
    Retrosynthesis task.

    This class wraps pretrained center identification and synthon completion modeules into a pipeline.

    Parameters:
        center_identification (CenterIdentification): sub task of center identification
        synthon_completion (SynthonCompletion): sub task of synthon completion
        center_topk (int, optional): number of reaction centers to predict for each product
        num_synthon_beam (int, optional): size of beam search for each synthon
        max_prediction (int, optional): max number of final predictions for each product
        metric (str or list of str, optional): metric(s). Available metrics are ``top-K``.
    """

    _option_members = set(["metric"])

    def __init__(self, center_identification, synthon_completion, center_topk=2, num_synthon_beam=10, max_prediction=20,
                 metric=("top-1", "top-3", "top-5", "top-10")):
        super(Retrosynthesis, self).__init__()
        self.center_identification = center_identification
        self.synthon_completion = synthon_completion
        self.center_topk = center_topk
        self.num_synthon_beam = num_synthon_beam
        self.max_prediction = max_prediction
        self.metric = metric

    def load_state_dict(self, state_dict, strict=True):
        if not strict:
            raise ValueError("Retrosynthesis only supports load_state_dict() with strict=True")
        keys = set(state_dict.keys())
        for model in [self.center_identification, self.synthon_completion]:
            if set(model.state_dict().keys()) == keys:
                return model.load_state_dict(state_dict, strict)
        raise RuntimeError("Neither of sub modules matches with state_dict")

    def predict(self, batch, all_loss=None, metric=None):
        synthon_batch = self.center_identification.predict_synthon(batch, self.center_topk)

        synthon = synthon_batch["synthon"]
        num_synthon = synthon_batch["num_synthon"]
        assert (num_synthon >= 1).all() and (num_synthon <= 2).all()
        synthon2split = functional._size_to_index(num_synthon)
        with synthon.graph():
            synthon.reaction_center = synthon_batch["reaction_center"][synthon2split]
            synthon.split_logp = synthon_batch["log_likelihood"][synthon2split]

        reactant = self.synthon_completion.predict_reactant(synthon_batch, self.num_synthon_beam, self.max_prediction)

        logps = []
        reactant_ids = []
        product_ids = []

        # case 1: one synthon
        is_single = num_synthon[synthon2split[reactant.synthon_id]] == 1
        reactant_id = is_single.nonzero().squeeze(-1)
        logps.append(reactant.split_logp[reactant_id] + reactant.logp[reactant_id])
        product_ids.append(reactant.product_id[reactant_id])
        # pad -1
        reactant_ids.append(torch.stack([reactant_id, -torch.ones_like(reactant_id)], dim=-1))

        # case 2: two synthons
        # use proposal to avoid O(n^2) complexity
        reactant1 = torch.arange(len(reactant), device=self.device)
        reactant1 = reactant1.unsqueeze(-1).expand(-1, self.max_prediction * 2)
        reactant2 = reactant1 + torch.arange(self.max_prediction * 2, device=self.device)
        valid = reactant2 < len(reactant)
        reactant1 = reactant1[valid]
        reactant2 = reactant2[valid]
        synthon1 = reactant.synthon_id[reactant1]
        synthon2 = reactant.synthon_id[reactant2]
        valid = (synthon1 < synthon2) & (synthon2split[synthon1] == synthon2split[synthon2])
        reactant1 = reactant1[valid]
        reactant2 = reactant2[valid]
        logps.append(reactant.split_logp[reactant1] + reactant.logp[reactant1] + reactant.logp[reactant2])
        product_ids.append(reactant.product_id[reactant1])
        reactant_ids.append(torch.stack([reactant1, reactant2], dim=-1))

        # combine case 1 & 2
        logps = torch.cat(logps)
        reactant_ids = torch.cat(reactant_ids)
        product_ids = torch.cat(product_ids)

        order = product_ids.argsort()
        logps = logps[order]
        reactant_ids = reactant_ids[order]
        num_prediction = torch.bincount(product_ids)
        logps, topk = functional.variadic_topk(logps, num_prediction, self.max_prediction)
        topk_index = topk + (num_prediction.cumsum(0) - num_prediction).unsqueeze(-1)
        topk_index_shifted = torch.cat([-torch.ones(len(topk_index), 1, dtype=torch.long, device=self.device),
                                        topk_index[:, :-1]], dim=-1)
        is_duplicate = topk_index == topk_index_shifted
        reactant_id = reactant_ids[topk_index] # (num_graph, k, 2)

        # why we need to repeat the graph?
        # because reactant_id may be duplicated, which is not directly supported by graph indexing
        is_padding = reactant_id == -1
        num_synthon = (~is_padding).sum(dim=-1)
        num_synthon = num_synthon[~is_duplicate]
        logps = logps[~is_duplicate]
        offset = torch.arange(self.max_prediction, device=self.device) * len(reactant)
        reactant_id = reactant_id + offset.view(1, -1, 1)
        reactant_id = reactant_id[~(is_padding | is_duplicate.unsqueeze(-1))]
        reactant = reactant.repeat(self.max_prediction)
        reactant = reactant[reactant_id]
        assert num_synthon.sum() == len(reactant)
        synthon2graph = functional._size_to_index(num_synthon)
        first_synthon = num_synthon.cumsum(0) - num_synthon
        # inherit graph attributes from the first synthon
        data_dict = reactant.data_mask(graph_index=first_synthon, include="graph")[0]
        # merge synthon pairs from the same split into a single graph
        reactant = reactant.merge(synthon2graph)
        with reactant.graph():
            for k, v in data_dict.items():
                setattr(reactant, k, v)
            reactant.logps = logps

        num_prediction = torch.bincount(reactant.product_id)

        return reactant, num_prediction # (num_graph * k)

    def target(self, batch):
        reactant, product = batch["graph"]
        reactant = reactant.ion_to_molecule()
        return reactant

    def evaluate(self, pred, target):
        pred, num_prediction = pred
        infinity = torch.iinfo(torch.long).max - 1

        metric = {}
        ranking = []
        # any better solution for parallel graph isomorphism?
        num_cum_prediction = num_prediction.cumsum(0)
        for i in range(len(target)):
            target_smiles = target[i].to_smiles(isomeric=False, atom_map=False, canonical=True)
            offset = (num_cum_prediction[i] - num_prediction[i]).item()
            for j in range(num_prediction[i]):
                pred_smiles = pred[offset + j].to_smiles(isomeric=False, atom_map=False, canonical=True)
                if pred_smiles == target_smiles:
                    break
            else:
                j = infinity
            ranking.append(j + 1)

        ranking = torch.tensor(ranking, device=self.device)
        for _metric in self.metric:
            if _metric.startswith("top-"):
                threshold = int(_metric[4:])
                score = (ranking <= threshold).float().mean()
                metric["top-%d accuracy" % threshold] = score
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

        return metric