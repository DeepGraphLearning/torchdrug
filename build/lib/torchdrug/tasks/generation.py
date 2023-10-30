import copy
import logging
import warnings
from collections import defaultdict

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_add, scatter_max
from torch_scatter.composite import scatter_log_softmax, scatter_softmax

from torchdrug import core, data, tasks, metrics, transforms
from torchdrug.core import Registry as R
from torchdrug.layers import functional
from torchdrug import layers


logger = logging.getLogger(__name__)


@R.register("tasks.AutoregressiveGeneration")
class AutoregressiveGeneration(tasks.Task, core.Configurable):
    """
    Autoregressive graph generation task.

    This class can be used to implement GraphAF proposed in
    `GraphAF: A Flow-based Autoregressive Model for Molecular Graph Generation`_.
    To do so, instantiate the node model and the edge model with two
    :class:`GraphAutoregressiveFlow <torchdrug.models.GraphAutoregressiveFlow>` models.

    .. _GraphAF\: A Flow-based Autoregressive Model for Molecular Graph Generation:
        https://arxiv.org/pdf/2001.09382.pdf

    Parameters:
        node_model (nn.Module): node likelihood model
        edge_model (nn.Module): edge likelihood model
        task (str or list of str, optional): property optimization task(s). Available tasks are ``plogp`` and ``qed``.
        num_node_sample (int, optional): number of node samples per graph. -1 for all samples.
        num_edge_sample (int, optional): number of edge samples per graph. -1 for all samples.
        max_edge_unroll (int, optional): max node id difference.
            If not provided, use the statistics from the training set.
        max_node (int, optional): max number of node.
            If not provided, use the statistics from the training set.
        criterion (str, list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``nll`` and ``ppo``.
        agent_update_interval (int, optional): update agent every n batch
        gamma (float, optional): reward discount rate
        reward_temperature (float, optional): temperature for reward. Higher temperature encourages larger mean reward,
            while lower temperature encourages larger maximal reward.
        baseline_momentum (float, optional): momentum for value function baseline
    """

    eps = 1e-10
    top_k = 10
    _option_members = {"task", "criterion"}

    def __init__(self, node_model, edge_model, task=(), num_node_sample=-1, num_edge_sample=-1,
                 max_edge_unroll=None, max_node=None, criterion="nll", agent_update_interval=5, gamma=0.9,
                 reward_temperature=1, baseline_momentum=0.9):
        super(AutoregressiveGeneration, self).__init__()
        self.node_model = node_model
        self.edge_model = edge_model
        self.agent_node_model = copy.deepcopy(node_model)
        self.agent_edge_model = copy.deepcopy(edge_model)
        self.task = task
        self.num_atom_type = self.node_model.input_dim
        self.num_bond_type = self.edge_model.input_dim
        self.num_node_sample = num_node_sample
        self.num_edge_sample = num_edge_sample
        self.max_edge_unroll = max_edge_unroll
        self.max_node = max_node
        self.criterion = criterion
        self.agent_update_interval = agent_update_interval
        self.gamma = gamma
        self.reward_temperature = reward_temperature
        self.baseline_momentum = baseline_momentum
        self.best_results = defaultdict(list)
        self.batch_id = 0

    def preprocess(self, train_set, valid_set, test_set):
        """
        Add atom id mapping and random BFS order to the training set.

        Compute ``max_edge_unroll`` and ``max_node`` on the training set if not provided.
        """
        remap_atom_type = transforms.RemapAtomType(train_set.atom_types)
        train_set.transform = transforms.Compose([
            train_set.transform,
            remap_atom_type,
            transforms.RandomBFSOrder(),
        ])
        self.register_buffer("id2atom", remap_atom_type.id2atom)
        self.register_buffer("atom2id", remap_atom_type.atom2id)

        if self.max_edge_unroll is None or self.max_node is None:
            self.max_edge_unroll = 0
            self.max_node = 0

            train_set = tqdm(train_set, "Computing max number of nodes and edge unrolling")
            for sample in train_set:
                graph = sample["graph"]
                if graph.edge_list.numel():
                    edge_unroll = (graph.edge_list[:, 0] - graph.edge_list[:, 1]).abs().max().item()
                    self.max_edge_unroll = max(self.max_edge_unroll, edge_unroll)
                self.max_node = max(self.max_node, graph.num_node)

            logger.warning("max node = %d, max edge unroll = %d" % (self.max_node, self.max_edge_unroll))

        self.register_buffer("node_baseline", torch.zeros(self.max_node + 1))
        self.register_buffer("edge_baseline", torch.zeros(self.max_node + 1))

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        for criterion, weight in self.criterion.items():
            if criterion == "nll":
                _loss, _metric = self.density_estimation_forward(batch)
                all_loss += _loss * weight
                metric.update(_metric)
            elif criterion == "ppo":
                _loss, _metric = self.reinforce_forward(batch)
                all_loss += _loss * weight
                metric.update(_metric)
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)

        return all_loss, metric

    def reinforce_forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        if self.batch_id % self.agent_update_interval == 0:
            self.agent_node_model.load_state_dict(self.node_model.state_dict())
            self.agent_edge_model.load_state_dict(self.edge_model.state_dict())
        self.batch_id += 1

        # generation takes less time when early_stop=True
        graph = self.generate(len(batch["graph"]), off_policy=True, early_stop=True)
        if len(graph) == 0 or graph.num_nodes.max() == 1:
            logger.error("Generation results collapse to singleton molecules")

            all_loss.requires_grad_()
            nan = torch.tensor(float("nan"), device=self.device)
            for task in self.task:
                if task == "plogp":
                    metric["Penalized logP"] = nan
                    metric["Penalized logP (max)"] = nan
                elif task == "qed":
                    metric["QED"] = nan
                    metric["QED (max)"] = nan
            metric["node PPO objective"] = nan
            metric["edge PPO objective"] = nan

            return all_loss, metric

        reward = torch.zeros(len(graph), device=self.device)
        for task in self.task:
            if task == "plogp":
                plogp = metrics.penalized_logP(graph)
                metric["Penalized logP"] = plogp.mean()
                metric["Penalized logP (max)"] = plogp.max()
                self.update_best_result(graph, plogp, "Penalized logP")
                reward += (plogp / self.reward_temperature).exp()

                if plogp.max().item() > 5:
                    print("Penalized logP max = %s" % plogp.max().item())
                    print(self.best_results["Penalized logP"])

            elif task == "qed":
                qed = metrics.QED(graph)
                metric["QED"] = qed.mean()
                metric["QED (max)"] = qed.max()
                self.update_best_result(graph, qed, "QED")
                reward += (qed / self.reward_temperature).exp()
                #reward += qed * 3

                if qed.max().item() > 0.93:
                    print("QED max = %s" % qed.max().item())
                    print(self.best_results["QED"])

            else:
                raise ValueError("Unknown task `%s`" % task)

        # these graph-level features will broadcast to all masked graphs
        with graph.graph():
            graph.reward = reward
            graph.original_num_nodes = graph.num_nodes
        graph.atom_type = self.atom2id[graph.atom_type]

        is_training = self.training
        # easily got nan if BN is trained
        self.bn_eval()

        masked_graph, node_target = self.mask_node(graph, metric)
        # reward reshaping
        reward = masked_graph.reward
        masked_graph.atom_type = self.id2atom[masked_graph.atom_type]
        reward = reward * self.gamma ** (masked_graph.original_num_nodes - masked_graph.num_nodes).float()

        # per graph size reward baseline
        weight = torch.ones_like(masked_graph.num_nodes, dtype=torch.float)
        baseline = scatter_add(reward, masked_graph.num_nodes, dim_size=self.max_node + 1) / \
                   (scatter_add(weight, masked_graph.num_nodes, dim_size=self.max_node + 1) + self.eps)
        self.node_baseline = self.node_baseline * self.baseline_momentum + baseline * (1 - self.baseline_momentum)
        reward -= self.node_baseline[masked_graph.num_nodes]
        reward += masked_graph.is_valid
        masked_graph.atom_type = self.atom2id[masked_graph.atom_type]

        log_likelihood = self.node_model(masked_graph, node_target, None, all_loss, metric)
        agent_log_likelihood = self.agent_node_model(masked_graph, node_target, None, all_loss, metric)
        objective = functional.clipped_policy_gradient_objective(log_likelihood, agent_log_likelihood, reward)
        objective = objective.mean()
        metric["node PPO objective"] = objective
        all_loss += -objective

        masked_graph, edge_target, edge = self.mask_edge(graph, metric)
        # reward reshaping
        reward = masked_graph.reward
        masked_graph.atom_type = self.id2atom[masked_graph.atom_type]
        reward = reward * self.gamma ** (masked_graph.original_num_nodes - masked_graph.num_nodes).float()

        # per graph size reward baseline
        weight = torch.ones_like(masked_graph.num_nodes, dtype=torch.float)
        baseline = scatter_add(reward, masked_graph.num_nodes, dim_size=self.max_node + 1) / \
                   (scatter_add(weight, masked_graph.num_nodes, dim_size=self.max_node + 1) + self.eps)
        self.edge_baseline = self.edge_baseline * self.baseline_momentum + baseline * (1 - self.baseline_momentum)
        reward -= self.edge_baseline[masked_graph.num_nodes]
        reward += masked_graph.is_valid
        masked_graph.atom_type = self.atom2id[masked_graph.atom_type]

        log_likelihood = self.edge_model(masked_graph, edge_target, edge, all_loss, metric)
        agent_log_likelihood = self.agent_edge_model(masked_graph, edge_target, edge, all_loss, metric)
        objective = functional.clipped_policy_gradient_objective(log_likelihood, agent_log_likelihood, reward)
        objective = objective.mean()
        metric["edge PPO objective"] = objective
        all_loss += -objective

        self.bn_train(is_training)

        return all_loss, metric

    def density_estimation_forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        graph = batch["graph"]
        masked_graph, node_target = self.mask_node(graph, metric)
        log_likelihood = self.node_model(masked_graph, node_target, None, all_loss, metric)
        log_likelihood = log_likelihood.mean()
        metric["node log likelihood"] = log_likelihood
        all_loss += -log_likelihood

        masked_graph, edge_target, edge = self.mask_edge(graph, metric)
        log_likelihood = self.edge_model(masked_graph, edge_target, edge, all_loss, metric)
        log_likelihood = log_likelihood.mean()
        metric["edge log likelihood"] = log_likelihood
        all_loss += -log_likelihood

        return all_loss, metric

    def evaluate(self, batch):
        pred = None
        metric = {}

        graph, target = self.all_node(batch["graph"])
        log_likelihood = self.node_model(graph, target)
        log_likelihood = log_likelihood.mean()
        metric["node log likelihood"] = log_likelihood

        graph, target = self.all_edge(batch["graph"])
        log_likelihood = self.edge_model(graph, target)
        log_likelihood = log_likelihood.mean()
        metric["edge log likelihood"] = log_likelihood

        return pred, metric

    def bn_train(self, mode=True):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.train(mode)

    def bn_eval(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()

    def update_best_result(self, graph, score, task):
        score = score.cpu()
        best_results = self.best_results[task]
        for s, i in zip(*score.sort(descending=True)):
            s = s.item()
            i = i.item()
            if len(best_results) == self.top_k and s < best_results[-1][0]:
                break
            best_results.append((s, graph[i].to_smiles()))
            best_results.sort(reverse=True)
            best_results = best_results[:self.top_k]
        self.best_results[task] = best_results

    @torch.no_grad()
    def generate(self, num_sample, max_resample=20, off_policy=False, early_stop=False, verbose=0):
        num_relation = self.num_bond_type - 1
        is_training = self.training
        self.eval()

        if off_policy:
            node_model = self.agent_node_model
            edge_model = self.agent_edge_model
        else:
            node_model = self.node_model
            edge_model = self.edge_model

        edge_list = torch.zeros(0, 3, dtype=torch.long, device=self.device)
        num_nodes = torch.zeros(num_sample, dtype=torch.long, device=self.device)
        num_edges = torch.zeros_like(num_nodes)
        atom_type = torch.zeros(0, dtype=torch.long, device=self.device)
        graph = data.PackedMolecule(edge_list, atom_type, edge_list[:, -1], num_nodes, num_edges,
                                    num_relation=num_relation)
        completed = torch.zeros(num_sample, dtype=torch.bool, device=self.device)

        for node_in in range(self.max_node):
            atom_pred = node_model.sample(graph)
            # why we add atom_pred even if it is completed?
            # because we need to batch edge model over (node_in, node_out), even on completed graphs
            atom_type, num_nodes = self._append(atom_type, num_nodes, atom_pred)
            graph = node_graph = data.PackedMolecule(edge_list, atom_type, edge_list[:, -1], num_nodes, num_edges,
                                                     num_relation=num_relation)

            start = max(0, node_in - self.max_edge_unroll)
            for node_out in range(start, node_in):
                is_valid = completed.clone()
                edge = torch.tensor([node_in, node_out], device=self.device).repeat(num_sample, 1)
                # default: non-edge
                bond_pred = (self.num_bond_type - 1) * torch.ones(num_sample, dtype=torch.long, device=self.device)
                for i in range(max_resample):
                    # only resample invalid graphs
                    mask = ~is_valid
                    bond_pred[mask] = edge_model.sample(graph, edge)[mask]
                    # check valency
                    mask = (bond_pred < edge_model.input_dim - 1) & ~completed
                    edge_pred = torch.cat([edge, bond_pred.unsqueeze(-1)], dim=-1)
                    tmp_edge_list, tmp_num_edges = self._append(edge_list, num_edges, edge_pred, mask)
                    edge_pred = torch.cat([edge.flip(-1), bond_pred.unsqueeze(-1)], dim=-1)
                    tmp_edge_list, tmp_num_edges = self._append(tmp_edge_list, tmp_num_edges, edge_pred, mask)
                    tmp_graph = data.PackedMolecule(tmp_edge_list, self.id2atom[atom_type], tmp_edge_list[:, -1],
                                                    num_nodes, tmp_num_edges, num_relation=num_relation)

                    is_valid = tmp_graph.is_valid | completed

                    if is_valid.all():
                        break

                if not is_valid.all() and verbose:
                    num_invalid = num_sample - is_valid.sum().item()
                    num_working = num_sample - completed.sum().item()
                    logger.warning("edge (%d, %d): %d / %d molecules are invalid even after %d resampling" %
                                   (node_in, node_out, num_invalid, num_working, max_resample))

                mask = (bond_pred < edge_model.input_dim - 1) & ~completed
                edge_pred = torch.cat([edge, bond_pred.unsqueeze(-1)], dim=-1)
                edge_list, num_edges = self._append(edge_list, num_edges, edge_pred, mask)
                edge_pred = torch.cat([edge.flip(-1), bond_pred.unsqueeze(-1)], dim=-1)
                edge_list, num_edges = self._append(edge_list, num_edges, edge_pred, mask)
                graph = data.PackedMolecule(edge_list, atom_type, edge_list[:, -1], num_nodes, num_edges,
                                            num_relation=num_relation)

            if node_in > 0:
                assert (graph.num_edges[completed] == node_graph.num_edges[completed]).all()
                completed |= graph.num_edges == node_graph.num_edges
                if early_stop:
                    graph.atom_type = self.id2atom[graph.atom_type]
                    completed |= ~graph.is_valid
                    graph.atom_type = self.atom2id[graph.atom_type]
                if completed.all():
                    break

        self.train(is_training)

        # remove isolated atoms
        index = graph.degree_out > 0
        # keep at least the first atom for each graph
        index[graph.num_cum_nodes - graph.num_nodes] = 1
        graph = graph.subgraph(index)
        graph.atom_type = self.id2atom[graph.atom_type]

        graph = graph[graph.is_valid_rdkit]
        return graph

    def _append(self, data, num_xs, input, mask=None):
        if mask is None:
            mask = torch.ones_like(num_xs, dtype=torch.bool)
        new_num_xs = num_xs + mask
        new_num_cum_xs = new_num_xs.cumsum(0)
        new_num_x = new_num_cum_xs[-1].item()
        new_data = torch.zeros(new_num_x, *data.shape[1:], dtype=data.dtype, device=data.device)
        starts = new_num_cum_xs - new_num_xs
        ends = starts + num_xs
        index = functional.multi_slice_mask(starts, ends, new_num_x)
        new_data[index] = data
        new_data[~index] = input[mask]
        return new_data, new_num_xs

    @torch.no_grad()
    def mask_node(self, graph, metric=None):
        if self.num_node_sample == -1:
            masked_graph, node_target = self.all_node(graph)
            if metric is not None:
                metric["node mask / graph"] = torch.tensor([len(masked_graph) / len(graph)], device=graph.device)
        else:
            masked_graph, node_target = self.sample_node(graph, self.num_node_sample)
        return masked_graph, node_target

    @torch.no_grad()
    def mask_edge(self, graph, metric=None):
        if self.num_edge_sample == -1:
            masked_graph, edge_target, edge = self.all_edge(graph)
            if metric is not None:
                metric["edge mask / graph"] = torch.tensor([len(masked_graph) / len(graph)], device=graph.device)
        else:
            masked_graph, edge_target, edge = self.sample_edge(graph, self.num_edge_sample)
        return masked_graph, edge_target, edge

    @torch.no_grad()
    def sample_node(self, graph, num_sample):
        graph = graph.repeat(num_sample)
        num_nodes = graph.num_nodes
        num_keep_nodes = torch.rand(len(graph), device=graph.device) * num_nodes # [0, num_nodes)
        num_keep_nodes = num_keep_nodes.long() # [0, num_nodes - 1]

        starts = graph.num_cum_nodes - graph.num_nodes
        ends = starts + num_keep_nodes
        mask = functional.multi_slice_mask(starts, ends, graph.num_node)

        new_graph = graph.subgraph(mask)
        target = graph.subgraph(ends).atom_type
        return new_graph, target

    @torch.no_grad()
    def all_node(self, graph):
        starts, ends, valid = self._all_prefix_slice(graph.num_nodes)

        num_repeat = len(starts) // len(graph)
        graph = graph.repeat(num_repeat)
        mask = functional.multi_slice_mask(starts, ends, graph.num_node)
        new_graph = graph.subgraph(mask)
        target = graph.subgraph(ends).atom_type

        return new_graph[valid], target[valid]

    @torch.no_grad()
    def sample_edge(self, graph, num_sample):
        if (graph.num_nodes < 2).any():
            graph = graph[graph.num_nodes >= 2]
            warnings.warn("Graphs with less than 2 nodes can't be used for edge generation learning. Dropped")

        lengths = self._valid_edge_prefix_lengths(graph)
        graph = graph.repeat(num_sample)

        num_max_node = graph.num_nodes.max().item()
        num_node2num_dense_edge = torch.arange(num_max_node + 1, device=graph.device) ** 2
        num_node2length_idx = (lengths.unsqueeze(-1) < num_node2num_dense_edge.unsqueeze(0)).sum(dim=0)
        # uniformly sample a mask from each graph's valid masks
        length_indexes = torch.rand(len(graph), device=graph.device) * num_node2length_idx[graph.num_nodes]
        length_indexes = length_indexes.long()
        num_keep_dense_edges = lengths[length_indexes]

        # undirected: all upper triangular edge ids are flipped to lower triangular ids
        # 1 -> 2, 4 -> 6, 5 -> 7
        node_index = graph.edge_list[:, :2] - graph._offsets.unsqueeze(-1)
        node_in, node_out = node_index.t()
        node_large = node_index.max(dim=-1)[0]
        node_small = node_index.min(dim=-1)[0]
        edge_id = node_large ** 2 + (node_in >= node_out) * node_large + node_small
        undirected_edge_id = node_large * (node_large + 1) + node_small

        edge_mask = undirected_edge_id < num_keep_dense_edges[graph.edge2graph]
        circum_box_size = (num_keep_dense_edges + 1.0).sqrt().ceil().long()
        starts = graph.num_cum_nodes - graph.num_nodes
        ends = starts + circum_box_size
        node_mask = functional.multi_slice_mask(starts, ends, graph.num_node)
        # compact nodes so that succeeding nodes won't affect graph pooling
        new_graph = graph.edge_mask(edge_mask).node_mask(node_mask, compact=True)

        positive_edge = edge_id == num_keep_dense_edges[graph.edge2graph]
        positive_graph = scatter_add(positive_edge.long(), graph.edge2graph, dim=0, dim_size=len(graph)).bool()
        # default: non-edge
        target = (self.num_bond_type - 1) * torch.ones(graph.batch_size, dtype=torch.long, device=graph.device)
        target[positive_graph] = graph.edge_list[positive_edge, 2]

        node_in = circum_box_size - 1
        node_out = num_keep_dense_edges - node_in * circum_box_size
        edge = torch.stack([node_in, node_out], dim=-1)

        return new_graph, target, edge

    @torch.no_grad()
    def all_edge(self, graph):
        if (graph.num_nodes < 2).any():
            graph = graph[graph.num_nodes >= 2]
            warnings.warn("Graphs with less than 2 nodes can't be used for edge generation learning. Dropped")

        lengths = self._valid_edge_prefix_lengths(graph)

        starts, ends, valid = self._all_prefix_slice(graph.num_nodes ** 2, lengths)

        num_keep_dense_edges = ends - starts
        num_repeat = len(starts) // len(graph)
        graph = graph.repeat(num_repeat)

        # undirected: all upper triangular edge ids are flipped to lower triangular ids
        # 1 -> 2, 4 -> 6, 5 -> 7
        node_index = graph.edge_list[:, :2] - graph._offsets.unsqueeze(-1)
        node_in, node_out = node_index.t()
        node_large = node_index.max(dim=-1)[0]
        node_small = node_index.min(dim=-1)[0]
        edge_id = node_large ** 2 + (node_in >= node_out) * node_large + node_small
        undirected_edge_id = node_large * (node_large + 1) + node_small

        edge_mask = undirected_edge_id < num_keep_dense_edges[graph.edge2graph]
        circum_box_size = (num_keep_dense_edges + 1.0).sqrt().ceil().long()
        starts = graph.num_cum_nodes - graph.num_nodes
        ends = starts + circum_box_size
        node_mask = functional.multi_slice_mask(starts, ends, graph.num_node)
        # compact nodes so that succeeding nodes won't affect graph pooling
        new_graph = graph.edge_mask(edge_mask).node_mask(node_mask, compact=True)

        positive_edge = edge_id == num_keep_dense_edges[graph.edge2graph]
        positive_graph = scatter_add(positive_edge.long(), graph.edge2graph, dim=0, dim_size=len(graph)).bool()
        # default: non-edge
        target = (self.num_bond_type - 1) * torch.ones(graph.batch_size, dtype=torch.long, device=graph.device)
        target[positive_graph] = graph.edge_list[positive_edge, 2]

        node_in = circum_box_size - 1
        node_out = num_keep_dense_edges - node_in * circum_box_size
        edge = torch.stack([node_in, node_out], dim=-1)

        return new_graph[valid], target[valid], edge[valid]

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
            lengths = torch.arange(num_max_x, device=num_xs.device)

        pack_offsets = torch.arange(len(lengths), device=num_xs.device) * num_cum_xs[-1]
        # starts, lengths, ends: (num_repeat, num_graph)
        starts = starts.unsqueeze(0) + pack_offsets.unsqueeze(-1)
        valid = lengths.unsqueeze(-1) <= num_xs.unsqueeze(0) - 1
        lengths = torch.min(lengths.unsqueeze(-1), num_xs.unsqueeze(0) - 1)
        ends = starts + lengths

        starts = starts.flatten()
        ends = ends.flatten()
        valid = valid.flatten()

        return starts, ends, valid

    @torch.no_grad()
    def _valid_edge_prefix_lengths(self, graph):
        # valid prefix lengths are across a batch, according to the largest graph
        num_max_node = graph.num_nodes.max().item()
        # edge id in an adjacency (snake pattern)
        #    in
        # o 0 1 4
        # u 2 3 5
        # t 6 7 8
        lengths = torch.arange(num_max_node ** 2, device=graph.device)
        circum_box_size = (lengths + 1.0).sqrt().ceil().long()
        # only keep lengths that ends in the lower triangular part of adjacency matrix
        lengths = lengths[lengths >= circum_box_size * (circum_box_size - 1)]
        # lengths: [0, 2, 3, 6, 7, 8, ...]
        # num_node2length_idx: [0, 1, 4, 6, ...]
        # num_edge_unrolls
        # 0
        # 1 0
        # 2 1 0
        num_edge_unrolls = (lengths + 1.0).sqrt().ceil().long() ** 2 - lengths - 1
        # num_edge_unrolls: [0, 1, 0, 2, 1, 0, ...]
        # remove lengths that unroll too much. they always lead to empty targets
        lengths = lengths[(num_edge_unrolls <= self.max_edge_unroll) & (num_edge_unrolls > 0)]

        return lengths


@R.register("tasks.GCPNGeneration")
class GCPNGeneration(tasks.Task, core.Configurable):
    """
    The graph generative model from `Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation`_.

    .. _Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation:
        https://papers.nips.cc/paper/7877-graph-convolutional-policy-network-for-goal-directed-molecular-graph-generation.pdf

    Parameters:
        model (nn.Module): graph representation model
        atom_types (list or set): set of all possible atom types
        task (str or list of str, optional): property optimization task(s)
        max_edge_unroll (int, optional): max node id difference.
            If not provided, use the statistics from the training set.
        max_node (int, optional): max number of node.
            If not provided, use the statistics from the training set.
        criterion (str, list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``nll`` and ``ppo``.
        agent_update_interval (int, optional): update the agent every n batch
        gamma (float, optional): reward discount rate
        reward_temperature (float, optional): temperature for reward. Higher temperature encourages larger mean reward,
            while lower temperature encourages larger maximal reward.
        baseline_momentum (float, optional): momentum for value function baseline
    """

    eps = 1e-10
    top_k = 10
    _option_members = {"task", "criterion"}

    def __init__(self, model, atom_types, max_edge_unroll=None, max_node=None, task=(), criterion="nll",
                 hidden_dim_mlp=128, agent_update_interval=10, gamma=0.9, reward_temperature=1, baseline_momentum=0.9):
        super(GCPNGeneration, self).__init__()
        self.model = model
        self.task = task
        self.max_edge_unroll = max_edge_unroll
        self.max_node = max_node
        self.criterion = criterion
        self.hidden_dim_mlp = hidden_dim_mlp
        self.agent_update_interval = agent_update_interval
        self.gamma = gamma
        self.reward_temperature = reward_temperature
        self.baseline_momentum = baseline_momentum
        self.best_results = defaultdict(list)
        self.batch_id = 0


        remap_atom_type = transforms.RemapAtomType(atom_types)
        self.register_buffer("id2atom", remap_atom_type.id2atom)
        self.register_buffer("atom2id", remap_atom_type.atom2id)

        self.new_atom_embeddings = nn.Parameter(torch.zeros(self.id2atom.size(0), self.model.output_dim))
        nn.init.normal_(self.new_atom_embeddings, mean=0, std=0.1)
        self.inp_dim_stop = self.model.output_dim
        self.mlp_stop = layers.MultiLayerPerceptron(self.inp_dim_stop, [self.hidden_dim_mlp, 2], activation='tanh')

        self.inp_dim_node1 = self.model.output_dim + self.model.output_dim
        self.mlp_node1 = layers.MultiLayerPerceptron(self.inp_dim_node1, [self.hidden_dim_mlp, 1], activation='tanh')
        self.inp_dim_node2 = 2 * self.model.output_dim + self.model.output_dim
        self.mlp_node2 = layers.MultiLayerPerceptron(self.inp_dim_node2, [self.hidden_dim_mlp, 1], activation='tanh')
        self.inp_dim_edge = 2 * self.model.output_dim
        self.mlp_edge = layers.MultiLayerPerceptron(self.inp_dim_edge, [self.hidden_dim_mlp, self.model.num_relation], activation='tanh')

        self.agent_model = copy.deepcopy(self.model)
        self.agent_new_atom_embeddings = copy.deepcopy(self.new_atom_embeddings)
        self.agent_mlp_stop = copy.deepcopy(self.mlp_stop)
        self.agent_mlp_node1 = copy.deepcopy(self.mlp_node1)
        self.agent_mlp_node2 = copy.deepcopy(self.mlp_node2)
        self.agent_mlp_edge = copy.deepcopy(self.mlp_edge)


    def preprocess(self, train_set, valid_set, test_set):
        """
        Add atom id mapping and random BFS order to the training set.

        Compute ``max_edge_unroll`` and ``max_node`` on the training set if not provided.
        """
        remap_atom_type = transforms.RemapAtomType(train_set.atom_types)
        train_set.transform = transforms.Compose([
            train_set.transform,
            transforms.RandomBFSOrder(),
        ])
        
        if self.max_edge_unroll is None or self.max_node is None:
            self.max_edge_unroll = 0
            self.max_node = 0

            train_set = tqdm(train_set, "Computing max number of nodes and edge unrolling")
            for sample in train_set:
                graph = sample["graph"]
                if graph.edge_list.numel():
                    edge_unroll = (graph.edge_list[:, 0] - graph.edge_list[:, 1]).abs().max().item()
                    self.max_edge_unroll = max(self.max_edge_unroll, edge_unroll)
                self.max_node = max(self.max_node, graph.num_node)

            logger.warning("max node = %d, max edge unroll = %d" % (self.max_node, self.max_edge_unroll))

        self.register_buffer("moving_baseline", torch.zeros(self.max_node + 1))

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        for criterion, weight in self.criterion.items():
            if criterion == "nll":
                _loss, _metric = self.MLE_forward(batch)
                all_loss += _loss * weight
                metric.update(_metric)
            elif criterion == "ppo":
                _loss, _metric = self.reinforce_forward(batch)
                all_loss += _loss * weight
                metric.update(_metric)
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)

        return all_loss, metric

    def predict(self, graph, label_dict, use_agent=False):
        # step1: get node/graph embeddings
        if not use_agent:
            output = self.model(graph, graph.node_feature.float())
        else:
            output = self.agent_model(graph, graph.node_feature.float())

        extended_node2graph = torch.arange(graph.num_nodes.size(0), 
                device=self.device).unsqueeze(1).repeat([1, self.id2atom.size(0)]).view(-1) # (num_graph * 16)
        extended_node2graph = torch.cat((graph.node2graph, extended_node2graph)) # (num_node + 16 * num_graph)

        graph_feature_per_node = output["graph_feature"][extended_node2graph]

        # step2: predict stop
        stop_feature = output["graph_feature"] #(num_graph, n_out)
        if not use_agent:
            stop_logits = self.mlp_stop(stop_feature) #(num_graph, 2)
        else:
            stop_logits = self.agent_mlp_stop(stop_feature) #(num_graph, 2)

        if label_dict == None:
            return stop_logits
        # step3: predict first node: node1
        node1_feature = output["node_feature"] #(num_node, n_out)

        node1_feature = torch.cat((node1_feature, 
                    self.new_atom_embeddings.repeat([graph.num_nodes.size(0), 1])), 0) # (num_node + 16 * num_graph, n_out)

        node2_feature_node2 = node1_feature.clone() # (num_node + 16 * num_graph, n_out)
        # cat graph emb
        node1_feature = torch.cat((node1_feature, graph_feature_per_node), 1)

        if not use_agent:
            node1_logits = self.mlp_node1(node1_feature).squeeze(1)  #(num_node + 16 * num_graph)
        else:
            node1_logits = self.agent_mlp_node1(node1_feature).squeeze(1)  #(num_node + 16 * num_graph)

        #mask the extended part
        mask = torch.zeros(node1_logits.size(), device=self.device)
        mask[:graph.num_node] = 1
        node1_logits = torch.where(mask>0, node1_logits, -10000.0*torch.ones(node1_logits.size(), device=self.device))

        # step4: predict second node: node2

        node1_index_per_graph = (graph.num_cum_nodes - graph.num_nodes) + label_dict["label1"] #(num_graph)
        node1_index = node1_index_per_graph[extended_node2graph] # (num_node + 16 * num_graph)
        node2_feature_node1 = node1_feature[node1_index] #(num_node + 16 * num_graph, n_out)
        node2_feature = torch.cat((node2_feature_node1, node2_feature_node2), 1) #(num_node + 16 * num_graph, 2n_out) 
        if not use_agent:
            node2_logits = self.mlp_node2(node2_feature).squeeze(1)  #(num_node + 16 * num_graph)        
        else:
            node2_logits = self.agent_mlp_node2(node2_feature).squeeze(1)  #(num_node + 16 * num_graph)        

        #mask the selected node1
        mask = torch.zeros(node2_logits.size(), device=self.device)
        mask[node1_index_per_graph] = 1
        node2_logits = torch.where(mask==0, node2_logits, -10000.0*torch.ones(node2_logits.size(), device=self.device))

        # step5: predict edge type
        is_new_node = label_dict["label2"] - graph.num_nodes # if an entry is non-negative, this is a new added node. (num_graph)
        graph_offset = torch.arange(graph.num_nodes.size(0), device=self.device)
        node2_index_per_graph = torch.where(is_new_node >= 0, 
                    graph.num_node + graph_offset * self.id2atom.size(0) + is_new_node, 
                    label_dict["label2"] + graph.num_cum_nodes - graph.num_nodes) # (num_graph)
        node2_index = node2_index_per_graph[extended_node2graph]

        edge_feature_node1 = node2_feature_node2[node1_index_per_graph] #(num_graph, n_out)
        edge_feature_node2 = node2_feature_node2[node2_index_per_graph] # #(num_graph, n_out)
        edge_feature = torch.cat((edge_feature_node1, edge_feature_node2), 1) #(num_graph, 2n_out)
        if not use_agent:
            edge_logits = self.mlp_edge(edge_feature) # (num_graph, num_relation)
        else:
            edge_logits = self.agent_mlp_edge(edge_feature) # (num_graph, num_relation)

        index_dict = {
            "node1_index_per_graph": node1_index_per_graph,
            "node2_index_per_graph": node2_index_per_graph,
            "extended_node2graph": extended_node2graph
        }
        return stop_logits, node1_logits, node2_logits, edge_logits, index_dict

    def reinforce_forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        if self.batch_id % self.agent_update_interval == 0:
            self.agent_model.load_state_dict(self.model.state_dict())
            self.agent_mlp_stop.load_state_dict(self.mlp_stop.state_dict())
            self.agent_mlp_node1.load_state_dict(self.mlp_node1.state_dict())
            self.agent_mlp_node2.load_state_dict(self.mlp_node2.state_dict())
            self.agent_mlp_edge.load_state_dict(self.mlp_edge.state_dict())
            self.agent_new_atom_embeddings.data = self.new_atom_embeddings.data.clone()

        self.batch_id += 1

        # generation takes less time when early_stop=True
        graph = self.generate(len(batch["graph"]), max_resample=20, off_policy=True, max_step=40 * 2, verbose=1)
        if len(graph) == 0 or graph.num_nodes.max() == 1:
            logger.error("Generation results collapse to singleton molecules")

            all_loss.requires_grad_()
            nan = torch.tensor(float("nan"), device=self.device)
            for task in self.task:
                if task == "plogp":
                    metric["Penalized logP"] = nan
                    metric["Penalized logP (max)"] = nan
                elif task == "qed":
                    metric["QED"] = nan
                    metric["QED (max)"] = nan
            metric["PPO objective"] = nan

            return all_loss, metric

        reward = torch.zeros(len(graph), device=self.device)
        for task in self.task:
            if task == "plogp":
                plogp = metrics.penalized_logP(graph)
                metric["Penalized logP"] = plogp.mean()
                metric["Penalized logP (max)"] = plogp.max()
                self.update_best_result(graph, plogp, "Penalized logP")
                # TODO: 
                reward += (plogp / self.reward_temperature).exp()

                if plogp.max().item() > 5:
                    print("Penalized logP max = %s" % plogp.max().item())
                    print(self.best_results["Penalized logP"])

            elif task == "qed":
                qed = metrics.QED(graph)
                metric["QED"] = qed.mean()
                metric["QED (max)"] = qed.max()
                self.update_best_result(graph, qed, "QED")
                # TODO:                 
                #reward += ((qed - 0.9) * 20).exp()
                #reward += ((qed - 0.4) * 4 / self.reward_temperature).exp()
                #reward += qed
                reward += (qed / self.reward_temperature).exp()


                if qed.max().item() > 0.93:
                    print("QED max = %s" % qed.max().item())
                    print(self.best_results["QED"])
            else:
                raise ValueError("Unknown task `%s`" % task)

        # these graph-level features will broadcast to all masked graphs
        with graph.graph():
            graph.reward = reward
            graph.original_num_nodes = graph.num_nodes

        #graph.atom_type = self.atom2id[graph.atom_type]

        is_training = self.training
        # easily got nan if BN is trained
        self.bn_eval()



        stop_graph, stop_label1, stop_label2, stop_label3, stop_label4 = self.all_stop(graph)
        edge_graph, edge_label1, edge_label2, edge_label3, edge_label4 = self.all_edge(graph)        

        graph = self._cat([stop_graph, edge_graph])
        label1_target = torch.cat([stop_label1, edge_label1])
        label2_target = torch.cat([stop_label2, edge_label2])
        label3_target = torch.cat([stop_label3, edge_label3])
        label4_target = torch.cat([stop_label4, edge_label4])
        label_dict = {"label1": label1_target, "label2": label2_target, "label3": label3_target, "label4": label4_target}

        # reward reshaping
        reward = graph.reward
        reward = reward * self.gamma ** (graph.original_num_nodes - graph.num_nodes).float()

        # per graph size reward baseline
        weight = torch.ones_like(graph.num_nodes, dtype=torch.float)
        baseline = scatter_add(reward, graph.num_nodes, dim_size=self.max_node + 1) / \
                   (scatter_add(weight, graph.num_nodes, dim_size=self.max_node + 1) + self.eps)
        # TODO:
        self.moving_baseline = self.moving_baseline * self.baseline_momentum + baseline * (1 - self.baseline_momentum)
        reward -= self.moving_baseline[graph.num_nodes]
        reward += graph.is_valid

        # calculate object
        stop_logits, node1_logits, node2_logits, edge_logits, index_dict = self.predict(graph, label_dict)
        with torch.no_grad():
            old_stop_logits, old_node1_logits, old_node2_logits, old_edge_logits, old_index_dict = self.predict(graph, label_dict, use_agent=True)

        stop_prob = F.log_softmax(stop_logits, dim=-1)
        node1_prob = scatter_log_softmax(node1_logits, index_dict["extended_node2graph"])
        node2_prob = scatter_log_softmax(node2_logits, index_dict["extended_node2graph"])
        edge_prob = F.log_softmax(edge_logits, dim=-1)
        old_stop_prob = F.log_softmax(old_stop_logits, dim=-1)
        old_node1_prob = scatter_log_softmax(old_node1_logits, old_index_dict["extended_node2graph"])
        old_node2_prob = scatter_log_softmax(old_node2_logits, old_index_dict["extended_node2graph"])
        old_edge_prob = F.log_softmax(old_edge_logits, dim=-1)

        cur_logp = stop_prob[:, 0] + node1_prob[index_dict["node1_index_per_graph"]] \
                        + node2_prob[index_dict["node2_index_per_graph"]] + torch.gather(edge_prob, -1, label3_target.view(-1, 1)).view(-1)
        cur_logp[label4_target==1] = stop_prob[:, 1][label4_target==1]

        old_logp = old_stop_prob[:, 0] + old_node1_prob[old_index_dict["node1_index_per_graph"]] \
                        + old_node2_prob[index_dict["node2_index_per_graph"]] + torch.gather(old_edge_prob, -1, label3_target.view(-1, 1)).view(-1)
        old_logp[label4_target==1] = old_stop_prob[:, 1][label4_target==1]
        objective = functional.clipped_policy_gradient_objective(cur_logp, old_logp, reward)
        objective = objective.mean()
        metric["PPO objective"] = objective
        all_loss += (-objective)

        self.bn_train(is_training)

        return all_loss, metric
    
    
    def MLE_forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        graph = batch["graph"]
        stop_graph, stop_label1, stop_label2, stop_label3, stop_label4 = self.all_stop(graph)
        edge_graph, edge_label1, edge_label2, edge_label3, edge_label4 = self.all_edge(graph)        

        graph = self._cat([stop_graph, edge_graph])
        label1_target = torch.cat([stop_label1, edge_label1])
        label2_target = torch.cat([stop_label2, edge_label2])
        label3_target = torch.cat([stop_label3, edge_label3])
        label4_target = torch.cat([stop_label4, edge_label4])
        label_dict = {"label1": label1_target, "label2": label2_target, "label3": label3_target, "label4": label4_target}
        stop_logits, node1_logits, node2_logits, edge_logits, index_dict = self.predict(graph, label_dict)

        loss_stop = F.nll_loss(F.log_softmax(stop_logits, dim=-1), label4_target, reduction='none')
        loss_stop = 0.5 * (torch.mean(loss_stop[label4_target==0]) + torch.mean(loss_stop[label4_target==1]))
        #loss_stop = torch.mean(loss_stop)
        metric["stop bce loss"] = loss_stop
        all_loss += loss_stop

        loss_node1 = -(scatter_log_softmax(node1_logits, index_dict["extended_node2graph"])[index_dict["node1_index_per_graph"]])
        loss_node1 = torch.mean(loss_node1[label4_target==0])
        metric["node1 loss"] = loss_node1
        all_loss += loss_node1

        loss_node2 = -(scatter_log_softmax(node2_logits, index_dict["extended_node2graph"])[index_dict["node2_index_per_graph"]])
        loss_node2 = torch.mean(loss_node2[label4_target==0])
        metric["node2 loss"] = loss_node2
        all_loss += loss_node2

        loss_edge = F.nll_loss(F.log_softmax(edge_logits, dim=-1), label3_target, reduction='none')

        loss_edge = torch.mean(loss_edge[label4_target==0])
        metric["edge loss"] = loss_edge
        all_loss += loss_edge

        metric["total loss"] = all_loss

        pred = stop_logits, node1_logits, node2_logits, edge_logits
        target = label1_target, label2_target, label3_target, label4_target, index_dict

        metric.update(self.evaluate(pred, target))

        return all_loss, metric

    def evaluate(self, pred, target):
        stop_logits, node1_logits, node2_logits, edge_logits = pred 
        label1_target, label2_target, label3_target, label4_target, index_dict = target
        metric = {}
        stop_acc = torch.argmax(stop_logits, -1) == label4_target
        metric["stop acc"] = stop_acc.float().mean()

        node1_pred = scatter_max(node1_logits, index_dict["extended_node2graph"])[1]
        node1_acc = node1_pred == index_dict["node1_index_per_graph"]
        metric["node1 acc"] = node1_acc[label4_target == 0].float().mean()

        node2_pred = scatter_max(node2_logits, index_dict["extended_node2graph"])[1]
        node2_acc = node2_pred == index_dict["node2_index_per_graph"]
        metric["node2 acc"] = node2_acc[label4_target == 0].float().mean()

        edge_acc = torch.argmax(edge_logits, -1) == label3_target
        metric["edge acc"] = edge_acc[label4_target == 0].float().mean()
        return metric

    # generation step
    # 1. top-1 action
    # 2. apply action

    @torch.no_grad()    
    def _construct_dist(self, prob_, graph):
        max_size = max(graph.num_nodes) + self.id2atom.size(0)
        probs = torch.zeros((len(graph), max_size), device=prob_.device).view(-1)
        start = (graph.num_cum_nodes - graph.num_nodes)[graph.node2graph]
        start = torch.arange(graph.num_node, device=self.device) - start
        index = torch.arange(graph.num_nodes.size(0), device=self.device) * max_size
        index = index[graph.node2graph] + start 
        probs[index] = prob_[:graph.num_node] 

        start_extend = torch.arange(len(self.id2atom), device=self.device).repeat(graph.num_nodes.size()) # (num_graph * 16)
        index_extend = torch.arange(len(graph), device=self.device) * max_size + graph.num_nodes
        index2graph = torch.arange(len(graph), device=self.device).repeat_interleave(len(self.id2atom))
        index_extend = index_extend[index2graph] + start_extend
        probs[index_extend] = prob_[graph.num_node:]
        probs = probs.view(len(graph.num_nodes), max_size)
        return torch.distributions.Categorical(probs), probs # (n_graph, max_size)

    @torch.no_grad()
    def _sample_action(self, graph, off_policy):
        if off_policy:
            model = self.agent_model
            new_atom_embeddings = self.agent_new_atom_embeddings
            mlp_stop = self.agent_mlp_stop
            mlp_node1 = self.agent_mlp_node1
            mlp_node2 = self.agent_mlp_node2
            mlp_edge = self.agent_mlp_edge
        else:
            model = self.model
            new_atom_embeddings = self.new_atom_embeddings
            mlp_stop = self.mlp_stop
            mlp_node1 = self.mlp_node1
            mlp_node2 = self.mlp_node2
            mlp_edge = self.mlp_edge

        # step1: get feature
        output = model(graph, graph.node_feature.float())

        extended_node2graph = torch.arange(len(graph), device=self.device).repeat_interleave(len(self.id2atom)) # (num_graph * 16)
        extended_node2graph = torch.cat((graph.node2graph, extended_node2graph)) # (num_node + 16 * num_graph)

        graph_feature_per_node = output["graph_feature"][extended_node2graph]

        # step2: predict stop
        stop_feature = output["graph_feature"] # (num_graph, n_out)
        stop_logits = mlp_stop(stop_feature) # (num_graph, 2)
        stop_prob = F.softmax(stop_logits, -1) # (num_graph, 2)
        stop_prob_dist = torch.distributions.Categorical(stop_prob)
        stop_pred = stop_prob_dist.sample()
        # step3: predict first node: node1

        node1_feature = output["node_feature"] #(num_node, n_out)

        node1_feature = torch.cat((node1_feature, 
                    new_atom_embeddings.repeat([graph.num_nodes.size(0), 1])), 0) # (num_node + 16 * num_graph, n_out)
        node2_feature_node2 = node1_feature.clone() # (num_node + 16 * num_graph, n_out)

        node1_feature = torch.cat((node1_feature, graph_feature_per_node), 1)

        node1_logits = mlp_node1(node1_feature).squeeze(1)  #(num_node + 16 * num_graph)
        #mask the extended part
        mask = torch.zeros(node1_logits.size(), device=self.device)
        mask[:graph.num_node] = 1
        node1_logits = torch.where(mask>0, node1_logits, -10000.0*torch.ones(node1_logits.size(), device=self.device))

        node1_prob = scatter_softmax(node1_logits, extended_node2graph) # (num_node + 16 * num_graph)
        node1_prob_dist, tmp = self._construct_dist(node1_prob, graph) # (num_graph, max)

        node1_pred = node1_prob_dist.sample() #(num_graph)
        node1_index_per_graph = node1_pred + (graph.num_cum_nodes - graph.num_nodes)
        # step4: predict second node: node2
        node1_index = node1_index_per_graph[extended_node2graph] # (num_node + 16 * num_graph)
        node2_feature_node1 = node1_feature[node1_index] # (num_node + 16 * num_graph, n_out)

        node2_feature = torch.cat((node2_feature_node1, node2_feature_node2), 1) # (num_node + 16 * num_graph, 2n_out)
        node2_logits = mlp_node2(node2_feature).squeeze(1)  # (num_node + 16 * num_graph)

        # mask the selected node1
        mask = torch.zeros(node2_logits.size(), device=self.device)
        mask[node1_index_per_graph] = 1
        node2_logits = torch.where(mask==0, node2_logits, -10000.0*torch.ones(node2_logits.size(), device=self.device))
        node2_prob = scatter_softmax(node2_logits, extended_node2graph) # (num_node + 16 * num_graph)
        node2_prob_dist, tmp = self._construct_dist(node2_prob, graph) # (num_graph, max)
        node2_pred = node2_prob_dist.sample() # (num_graph,)
        is_new_node = node2_pred - graph.num_nodes
        graph_offset = torch.arange(graph.num_nodes.size(0), device=self.device)
        node2_index_per_graph = torch.where(is_new_node >= 0,
                                            graph.num_node + graph_offset * self.id2atom.size(0) + is_new_node,
                                            node2_pred + graph.num_cum_nodes - graph.num_nodes)


        # step5: predict edge type
        edge_feature_node1 = node2_feature_node2[node1_index_per_graph] # (num_graph, n_out)
        edge_feature_node2 = node2_feature_node2[node2_index_per_graph] # (num_graph, n_out)
        edge_feature = torch.cat((edge_feature_node1, edge_feature_node2), 1) # (num_graph, 2n_out)
        edge_logits = mlp_edge(edge_feature)
        edge_prob = F.softmax(edge_logits, -1) # (num_graph, 3)
        edge_prob_dist = torch.distributions.Categorical(edge_prob)
        edge_pred = edge_prob_dist.sample()        

        return stop_pred, node1_pred, node2_pred, edge_pred

    @torch.no_grad()
    def _top1_action(self, graph, off_policy):

        if off_policy:
            model = self.agent_model
            new_atom_embeddings = self.agent_new_atom_embeddings
            mlp_stop = self.agent_mlp_stop
            mlp_node1 = self.agent_mlp_node1
            mlp_node2 = self.agent_mlp_node2
            mlp_edge = self.agent_mlp_edge
        else:
            model = self.model
            new_atom_embeddings = self.new_atom_embeddings
            mlp_stop = self.mlp_stop
            mlp_node1 = self.mlp_node1
            mlp_node2 = self.mlp_node2
            mlp_edge = self.mlp_edge

        # step1: get feature
        output = model(graph, graph.node_feature.float())

        extended_node2graph = torch.arange(graph.num_nodes.size(0), 
                device=self.device).unsqueeze(1).repeat([1, self.id2atom.size(0)]).view(-1) # (num_graph * 16)
        extended_node2graph = torch.cat((graph.node2graph, extended_node2graph)) # (num_node + 16 * num_graph)

        graph_feature_per_node = output["graph_feature"][extended_node2graph]

        # step2: predict stop
        stop_feature = output["graph_feature"] # (num_graph, n_out)
        stop_logits = mlp_stop(stop_feature) # (num_graph, 2)
        stop_pred = torch.argmax(stop_logits, -1) # (num_graph,)
        # step3: predict first node: node1

        node1_feature = output["node_feature"] #(num_node, n_out)

        node1_feature = torch.cat((node1_feature, 
                    new_atom_embeddings.repeat([graph.num_nodes.size(0), 1])), 0) # (num_node + 16 * num_graph, n_out)
        node2_feature_node2 = node1_feature.clone() # (num_node + 16 * num_graph, n_out)

        node1_feature = torch.cat((node1_feature, graph_feature_per_node), 1)

        node1_logits = mlp_node1(node1_feature).squeeze(1)  # (num_node + 16 * num_graph)
        # mask the extended part
        mask = torch.zeros(node1_logits.size(), device=self.device)
        mask[:graph.num_node] = 1
        node1_logits = torch.where(mask>0, node1_logits, -10000.0*torch.ones(node1_logits.size(), device=self.device))

        node1_index_per_graph = scatter_max(node1_logits, extended_node2graph)[1] # (num_node + 16 * num_graph)
        node1_pred = node1_index_per_graph - (graph.num_cum_nodes - graph.num_nodes)

        # step4: predict second node: node2
        node1_index = node1_index_per_graph[extended_node2graph] # (num_node + 16 * num_graph)
        node2_feature_node1 = node1_feature[node1_index] # (num_node + 16 * num_graph, n_out)

        node2_feature = torch.cat((node2_feature_node1, node2_feature_node2), 1) # (num_node + 16 * num_graph, 2n_out
        node2_logits = mlp_node2(node2_feature).squeeze(1)  # (num_node + 16 * num_graph)

        #mask the selected node1
        mask = torch.zeros(node2_logits.size(), device=self.device)
        mask[node1_index_per_graph] = 1
        node2_logits = torch.where(mask==0, node2_logits, -10000.0*torch.ones(node2_logits.size(), device=self.device))
        node2_index_per_graph = scatter_max(node2_logits, extended_node2graph)[1] # (num_node + 16 * num_graph)

        is_new_node =  node2_index_per_graph - graph.num_node # non negative if is new node
        graph_offset = torch.arange(graph.num_nodes.size(0), device=self.device)
        node2_pred = torch.where(is_new_node>=0, graph.num_nodes + is_new_node - graph_offset * self.id2atom.size(0),
                            node2_index_per_graph - (graph.num_cum_nodes - graph.num_nodes))

        # step5: predict edge type
        edge_feature_node1 = node2_feature_node2[node1_index_per_graph] #(num_graph, n_out)
        edge_feature_node2 = node2_feature_node2[node2_index_per_graph] # #(num_graph, n_out)
        edge_feature = torch.cat((edge_feature_node1, edge_feature_node2), 1) #(num_graph, 2n_out)
        edge_logits = mlp_edge(edge_feature)
        edge_pred = torch.argmax(edge_logits, -1)

        return stop_pred, node1_pred, node2_pred, edge_pred        

    @torch.no_grad()
    def _apply_action(self, graph, off_policy, max_resample=10, verbose=0, min_node=5):
        # action (num_graph, 4)

        # stopped graph is removed, initialize is_valid as False
        is_valid = torch.zeros(len(graph), dtype=torch.bool, device=self.device)
        stop_action = torch.zeros(len(graph), dtype=torch.long, device=self.device)
        node1_action = torch.zeros(len(graph), dtype=torch.long, device=self.device)
        node2_action = torch.zeros(len(graph), dtype=torch.long, device=self.device)
        edge_action = torch.zeros(len(graph), dtype=torch.long, device=self.device)

        for i in range(max_resample):
            # maximal resample time
            mask = ~is_valid
            if max_resample == 1:
                tmp_stop_action, tmp_node1_action, tmp_node2_action, tmp_edge_action = \
                    self._top1_action(graph, off_policy)
            else:
                tmp_stop_action, tmp_node1_action, tmp_node2_action, tmp_edge_action = \
                    self._sample_action(graph, off_policy)

            stop_action[mask] = tmp_stop_action[mask]
            node1_action[mask] = tmp_node1_action[mask]
            node2_action[mask] = tmp_node2_action[mask]
            edge_action[mask] = tmp_edge_action[mask]

            stop_action[graph.num_nodes <= 5] = 0
            # tmp add new nodes
            has_new_node = (node2_action >= graph.num_nodes) & (stop_action == 0)
            new_atom_id = (node2_action - graph.num_nodes)[has_new_node]
            new_atom_type = self.id2atom[new_atom_id]

            atom_type, num_nodes = functional._extend(graph.atom_type, graph.num_nodes, new_atom_type, has_new_node)

            # tmp cast to regular node ids
            node2_action = torch.where(has_new_node, graph.num_nodes, node2_action)

            # tmp modify edges
            new_edge = torch.stack([node1_action, node2_action], dim=-1)
            edge_list = graph.edge_list.clone()
            bond_type = graph.bond_type.clone()
            edge_list[:, :2] -= graph._offsets.unsqueeze(-1)
            is_modified_edge = (edge_list[:, :2] == new_edge[graph.edge2graph]).all(dim=-1) & \
                        (stop_action[graph.edge2graph] == 0)
            has_modified_edge = scatter_max(is_modified_edge.long(), graph.edge2graph, dim_size=len(graph))[0] > 0
            bond_type[is_modified_edge] = edge_action[has_modified_edge]
            edge_list[is_modified_edge, 2] = edge_action[has_modified_edge]
            # tmp modify reverse edges
            new_edge = new_edge.flip(-1)
            is_modified_edge = (edge_list[:, :2] == new_edge[graph.edge2graph]).all(dim=-1) & \
                        (stop_action[graph.edge2graph] == 0)   
            bond_type[is_modified_edge] = edge_action[has_modified_edge]
            edge_list[is_modified_edge, 2] = edge_action[has_modified_edge]             


            # tmp add new edges
            has_new_edge = (~has_modified_edge) & (stop_action == 0)
            new_edge_list = torch.stack([node1_action, node2_action, edge_action], dim=-1)[has_new_edge]
            bond_type = functional._extend(bond_type, graph.num_edges, edge_action[has_new_edge], has_new_edge)[0]
            edge_list, num_edges = functional._extend(edge_list, graph.num_edges, new_edge_list, has_new_edge)

            # tmp add reverse edges
            new_edge_list = torch.stack([node2_action, node1_action, edge_action], dim=-1)[has_new_edge]
            bond_type = functional._extend(bond_type, num_edges, edge_action[has_new_edge], has_new_edge)[0]
            edge_list, num_edges = functional._extend(edge_list, num_edges, new_edge_list, has_new_edge)

            tmp_graph = type(graph)(edge_list, atom_type=atom_type, bond_type=bond_type, num_nodes=num_nodes,
                                    num_edges=num_edges, num_relation=graph.num_relation)
            is_valid = tmp_graph.is_valid | (stop_action == 1)
            if is_valid.all():
                break
        if not is_valid.all() and verbose:
            num_invalid = len(graph) - is_valid.sum().item()
            num_working = len(graph)
            logger.warning("%d / %d molecules are invalid even after %d resampling" %
                           (num_invalid, num_working, max_resample))

        # apply the true action
        # inherit attributes
        data_dict = graph.data_dict
        meta_dict = graph.meta_dict
        for key in ["atom_type", "bond_type"]:
            data_dict.pop(key)
        # pad 0 for node / edge attributes
        for k, v in data_dict.items():
            if "node" in meta_dict[k]:
                shape = (len(new_atom_type), *v.shape[1:])
                new_data = torch.zeros(shape, dtype=v.dtype, device=self.device)
                data_dict[k] = functional._extend(v, graph.num_nodes, new_data, has_new_node)[0]
            if "edge" in meta_dict[k]:
                shape = (len(new_edge_list) * 2, *v.shape[1:])
                new_data = torch.zeros(shape, dtype=v.dtype, device=self.device)
                data_dict[k] = functional._extend(v, graph.num_edges, new_data, has_new_edge * 2)[0]

        new_graph = type(graph)(edge_list, atom_type=atom_type, bond_type=bond_type, num_nodes=num_nodes,
                                num_edges=num_edges, num_relation=graph.num_relation,
                                meta_dict=meta_dict, **data_dict)
        with new_graph.graph():
            new_graph.is_stopped = stop_action == 1

        new_graph, feature_valid = self._update_molecule_feature(new_graph)

        return new_graph[feature_valid]

    def _update_molecule_feature(self, graphs):
        # This function is very slow
        mols = graphs.to_molecule(ignore_error=True)
        valid = [mol is not None for mol in mols]
        valid = torch.tensor(valid, device=graphs.device)
        new_graphs = type(graphs).from_molecule(mols, kekulize=True, atom_feature="symbol")

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

    def bn_train(self, mode=True):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.train(mode)

    def bn_eval(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()

    def update_best_result(self, graph, score, task):
        score = score.cpu()
        best_results = self.best_results[task]
        for s, i in zip(*score.sort(descending=True)):
            s = s.item()
            i = i.item()
            if len(best_results) == self.top_k and s < best_results[-1][0]:
                break
            best_results.append((s, graph[i].to_smiles()))
            best_results.sort(reverse=True)
            best_results = best_results[:self.top_k]
        self.best_results[task] = best_results

    @torch.no_grad()
    def generate(self, num_sample, max_resample=20, off_policy=False, max_step=30 * 2, initial_smiles="C", verbose=0):
        is_training = self.training
        self.eval()

        graph = data.Molecule.from_smiles(initial_smiles, kekulize=True, atom_feature="symbol").repeat(num_sample)

        # TODO: workaround
        if self.device.type == "cuda":
            graph = graph.cuda(self.device)

        result = []
        for i in range(max_step):
            new_graph = self._apply_action(graph, off_policy, max_resample, verbose=1)
            if i == max_step - 1:
                # last step, collect all graph that is valid
                result.append(new_graph[(new_graph.num_nodes <= (self.max_node))])            
            else:
                result.append(new_graph[new_graph.is_stopped | (new_graph.num_nodes == (self.max_node))])

                is_continue = (~new_graph.is_stopped) & (new_graph.num_nodes < (self.max_node))
                graph = new_graph[is_continue]
                if len(graph) == 0:
                    break

        self.train(is_training)

        result = self._cat(result)
        return result

    def _append(self, data, num_xs, input, mask=None):
        if mask is None:
            mask = torch.ones_like(num_xs, dtype=torch.bool)
        new_num_xs = num_xs + mask
        new_num_cum_xs = new_num_xs.cumsum(0)
        new_num_x = new_num_cum_xs[-1].item()
        new_data = torch.zeros(new_num_x, *data.shape[1:], dtype=data.dtype, device=data.device)
        starts = new_num_cum_xs - new_num_xs
        ends = starts + num_xs
        index = functional.multi_slice_mask(starts, ends, new_num_x)
        new_data[index] = data
        new_data[~index] = input[mask]
        return new_data, new_num_xs

    @torch.no_grad()
    def all_stop(self, graph):
        if (graph.num_nodes < 2).any():
            graph = graph[graph.num_nodes >= 2]
            warnings.warn("Graphs with less than 2 nodes can't be used for stop prediction learning. Dropped")

        label1 = torch.zeros(len(graph), dtype=torch.long, device=self.device)
        label2 = torch.zeros_like(label1)
        label3 = torch.zeros_like(label1)        
        return graph, label1, label2, label3, torch.ones(len(graph), dtype=torch.long, device=self.device)


    @torch.no_grad()
    def all_edge(self, graph):
        if (graph.num_nodes < 2).any():
            graph = graph[graph.num_nodes >= 2]
            warnings.warn("Graphs with less than 2 nodes can't be used for edge generation learning. Dropped")

        lengths = self._valid_edge_prefix_lengths(graph)

        starts, ends, valid = self._all_prefix_slice(graph.num_nodes ** 2, lengths)

        num_keep_dense_edges = ends - starts
        num_repeat = len(starts) // len(graph)
        graph = graph.repeat(num_repeat)

        # undirected: all upper triangular edge ids are flipped to lower triangular ids
        # 1 -> 2, 4 -> 6, 5 -> 7
        node_index = graph.edge_list[:, :2] - graph._offsets.unsqueeze(-1)
        node_in, node_out = node_index.t()
        node_large = node_index.max(dim=-1)[0]
        node_small = node_index.min(dim=-1)[0]
        edge_id = node_large ** 2 + (node_in >= node_out) * node_large + node_small
        undirected_edge_id = node_large * (node_large + 1) + node_small

        edge_mask = undirected_edge_id < num_keep_dense_edges[graph.edge2graph]
        circum_box_size = (num_keep_dense_edges + 1.0).sqrt().ceil().long()

        # check whether we need to add a new node for the current edge 
        masked_undirected_edge_id = torch.where(edge_mask, undirected_edge_id, -torch.ones(undirected_edge_id.size(), 
                                                        dtype=torch.long, device=graph.device))
        current_circum_box_size = scatter_max(masked_undirected_edge_id, graph.edge2graph, dim=0)[0]
        current_circum_box_size = (current_circum_box_size + 1.0).sqrt().ceil().long()
        is_new_node_edge = (circum_box_size > current_circum_box_size).long()

        starts = graph.num_cum_nodes - graph.num_nodes
        ends = starts + circum_box_size - is_new_node_edge
        node_mask = functional.multi_slice_mask(starts, ends, graph.num_node)
        # compact nodes so that succeeding nodes won't affect graph pooling
        new_graph = graph.edge_mask(edge_mask).node_mask(node_mask, compact=True)

        positive_edge = edge_id == num_keep_dense_edges[graph.edge2graph]
        positive_graph = scatter_add(positive_edge.long(), graph.edge2graph, dim=0, dim_size=len(graph)).bool()
        # default: non-edge
        target = (self.model.num_relation) * torch.ones(graph.batch_size, dtype=torch.long, device=graph.device)
        target[positive_graph] = graph.edge_list[:, 2][positive_edge]

        # node_in > node_out
        node_in = circum_box_size - 1
        node_out = num_keep_dense_edges - node_in * circum_box_size
        # if we need to add a new node, what will be its atomid?
        new_node_atomid = self.atom2id[graph.atom_type[starts +node_in]]

        # keep only the positive graph, as we will add an edge at each step
        new_graph = new_graph[positive_graph]
        target = target[positive_graph]
        node_in = node_in[positive_graph]
        node_out = node_out[positive_graph]
        is_new_node_edge = is_new_node_edge[positive_graph]
        new_node_atomid = new_node_atomid[positive_graph]

        node_in_extend = new_graph.num_nodes + new_node_atomid
        node_in_final = torch.where(is_new_node_edge == 0, node_in, node_in_extend)        

        return new_graph, node_out, node_in_final, target, torch.zeros_like(node_out)

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
            lengths = torch.arange(num_max_x, device=num_xs.device)

        pack_offsets = torch.arange(len(lengths), device=num_xs.device) * num_cum_xs[-1]
        # starts, lengths, ends: (num_repeat, num_graph)
        starts = starts.unsqueeze(0) + pack_offsets.unsqueeze(-1)
        valid = lengths.unsqueeze(-1) <= num_xs.unsqueeze(0) - 1
        lengths = torch.min(lengths.unsqueeze(-1), num_xs.unsqueeze(0) - 1).clamp(0)
        ends = starts + lengths

        starts = starts.flatten()
        ends = ends.flatten()
        valid = valid.flatten()

        return starts, ends, valid

    @torch.no_grad()
    def _valid_edge_prefix_lengths(self, graph):
        num_max_node = graph.num_nodes.max().item()
        # edge id in an adjacency (snake pattern)
        #    in
        # o 0 1 4
        # u 2 3 5
        # t 6 7 8
        lengths = torch.arange(num_max_node ** 2, device=graph.device)
        circum_box_size = (lengths + 1.0).sqrt().ceil().long()
        # only keep lengths that ends in the lower triangular part of adjacency matrix
        lengths = lengths[lengths >= circum_box_size * (circum_box_size - 1)]
        # lengths: [0, 2, 3, 6, 7, 8, ...]
        # num_node2length_idx: [0, 1, 4, 6, ...]
        # num_edge_unrolls
        # 0
        # 1 0
        # 2 1 0
        num_edge_unrolls = (lengths + 1.0).sqrt().ceil().long() ** 2 - lengths - 1
        # num_edge_unrolls: [0, 1, 0, 2, 1, 0, ...]
        # remove lengths that unroll too much. they always lead to empty targets.
        lengths = lengths[(num_edge_unrolls <= self.max_edge_unroll) & (num_edge_unrolls > 0)]

        return lengths

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