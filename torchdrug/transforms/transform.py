import copy
import logging
from collections import deque

import torch

from torchdrug import core
from torchdrug.core import Registry as R


logger = logging.getLogger(__name__)


@R.register("transforms.NormalizeTarget")
class NormalizeTarget(core.Configurable):
    """
    Normalize the target values in a sample.

    Parameters:
        mean (dict of float): mean of targets
        std (dict of float): standard deviation of targets
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, item):
        item = item.copy()
        for k in self.mean:
            if k in item:
                item[k] = (item[k] - self.mean[k]) / self.std[k]
            else:
                raise ValueError("Can't find target `%s` in data item" % k)
        return item


@R.register("transforms.RemapAtomType")
class RemapAtomType(core.Configurable):
    """
    Map atom types to their index in a vocabulary. Atom types that don't present in the vocabulary are mapped to -1.

    Parameters:
        atom_types (array_like): vocabulary of atom types
    """

    def __init__(self, atom_types):
        atom_types = torch.as_tensor(atom_types)
        self.id2atom = atom_types
        self.atom2id = - torch.ones(atom_types.max() + 1, dtype=torch.long, device=atom_types.device)
        self.atom2id[atom_types] = torch.arange(len(atom_types), device=atom_types.device)

    def __call__(self, item):
        graph = copy.copy(item["graph"])
        graph.atom_type = self.atom2id[graph.atom_type]
        item = item.copy()
        item["graph"] = graph
        return item


@R.register("transforms.RandomBFSOrder")
class RandomBFSOrder(core.Configurable):
    """
    Order the nodes in a graph according to a random BFS order.
    """

    def __call__(self, item):
        graph = item["graph"]
        edge_list = graph.edge_list[:, :2].tolist()
        neighbor = [[] for _ in range(graph.num_node)]
        for h, t in edge_list:
            neighbor[h].append(t)
        depth = [-1] * graph.num_node

        i = torch.randint(graph.num_node, (1,)).item()
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

        item = item.copy()
        item["graph"] = graph.subgraph(order)
        return item


@R.register("transforms.Shuffle")
class Shuffle(core.Configurable):
    """
    Shuffle the order of nodes and edges in a graph.

    Parameters:
        shuffle_node (bool, optional): shuffle node order or not
        shuffle_edge (bool, optional): shuffle edge order or not
    """

    def __init__(self, shuffle_node=True, shuffle_edge=True):
        self.shuffle_node = shuffle_node
        self.shuffle_edge = shuffle_edge

    def __call__(self, item):
        graph = item["graph"]
        data = self.transform_data(graph.data_dict, graph.meta)

        item = item.copy()
        item["graph"] = type(graph)(**data)
        return item

    def transform_data(self, data, meta):
        edge_list = data["edge_list"]
        num_node = data["num_node"]
        num_edge = data["num_edge"]
        if self.shuffle_edge:
            node_perm = torch.randperm(num_node, device=edge_list.device)
        else:
            node_perm = torch.arange(num_node, device=edge_list.device)
        if self.shuffle_edge:
            edge_perm = torch.randperm(num_edge, device=edge_list.device)
        else:
            edge_perm = torch.randperm(num_edge, device=edge_list.device)
        new_data = {}
        for key in data:
            if meta[key] == "node":
                new_data[key] = data[key][node_perm]
            elif meta[key] == "edge":
                new_data[key] = node_perm[data[key][edge_perm]]
            else:
                new_data[key] = data[key]

        return new_data


@R.register("transforms.VirtualNode")
class VirtualNode(core.Configurable):
    """
    Add a virtual node and connect it with every node in the graph.

    Parameters:
        relation (int, optional): relation of virtual edges.
            By default, use the maximal relation in the graph plus 1.
        weight (int, optional): weight of virtual edges
        node_feature (array_like, optional): feature of the virtual node
        edge_feature (array_like, optional): feature of virtual edges
        kwargs: other attributes of the virtual node or virtual edges
    """

    def __init__(self, relation=None, weight=1, node_feature=None, edge_feature=None, **kwargs):
        self.relation = relation
        self.weight = weight

        self.default = {k: torch.as_tensor(v) for k, v in kwargs.items()}
        if node_feature is not None:
            self.default["node_feature"] = torch.as_tensor(node_feature)
        if edge_feature is not None:
            self.default["edge_feature"] = torch.as_tensor(edge_feature)

    def __call__(self, item):
        graph = item["graph"]
        edge_list = graph.edge_list
        edge_weight = graph.edge_weight
        num_node = graph.num_node
        num_relation = graph.num_relation

        existing_node = torch.arange(num_node, device=edge_list.device)
        virtual_node = torch.ones(num_node, dtype=torch.long, device=edge_list.device) * num_node
        node_in = torch.cat([virtual_node, existing_node])
        node_out = torch.cat([existing_node, virtual_node])
        if edge_list.shape[1] == 2:
            new_edge = torch.stack([node_in, node_out], dim=-1)
        else:
            if self.relation is None:
                relation = num_relation
                num_relation = num_relation + 1
            else:
                relation = self.relation
            relation = relation * torch.ones(num_node * 2, dtype=torch.long, device=edge_list.device)
            new_edge = torch.stack([node_in, node_out, relation], dim=-1)
        edge_list = torch.cat([edge_list, new_edge])
        new_edge_weight = self.weight * torch.ones(num_node * 2, device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, new_edge_weight])

        # add default node/edge attributes
        data = graph.data_dict.copy()
        for key, value in graph.meta.items():
            if value == "node":
                if key in self.default:
                    new_data = self.default[key].unsqueeze(0)
                else:
                    new_data = torch.zeros(1, *data[key].shape[1:], dtype=data[key].dtype, device=data[key].device)
                data[key] = torch.cat([data[key], new_data])
            elif value == "edge":
                if key in self.default:
                    repeat = [-1] * (data[key].ndim - 1)
                    new_data = self.default[key].expand(num_node * 2, *repeat)
                else:
                    new_data = torch.zeros(num_node * 2, *data[key].shape[1:],
                                           dtype=data[key].dtype, device=data[key].device)
                data[key] = torch.cat([data[key], new_data])

        graph = type(graph)(edge_list, edge_weight=edge_weight, num_node=num_node + 1,
                            num_relation=num_relation, meta=graph.meta, **data)

        item = item.copy()
        item["graph"] = graph
        return item


@R.register("transforms.VirtualAtom")
class VirtualAtom(VirtualNode, core.Configurable):
    """
    Add a virtual atom and connect it with every atom in the molecule.

    Parameters:
        atom_type (int, optional): type of the virtual atom
        bond_type (int, optional): type of the virtual bonds
        node_feature (array_like, optional): feature of the virtual atom
        edge_feature (array_like, optional): feature of virtual bonds
        kwargs: other attributes of the virtual atoms or virtual bonds
    """

    def __init__(self, atom_type=None, bond_type=None, node_feature=None, edge_feature=None, **kwargs):
        super(VirtualAtom, self).__init__(relation=bond_type, weight=1, node_feature=node_feature,
                                          edge_feature=edge_feature, atom_type=atom_type, **kwargs)


@R.register("transforms.TruncateProtein")
class TruncateProtein(core.Configurable):
    """
    Truncate over long protein sequences into a fixed length.

    Parameters:
        max_length (int, optional): maximal length of the sequence. Truncate the sequence if it exceeds this limit.
        random (bool, optional): truncate the sequence at a random position.
            If not, truncate the suffix of the sequence.
        keys (str or list of str, optional): keys for the items that require truncation in a sample
    """

    def __init__(self, max_length=None, random=False, keys="graph"):
        self.truncate_length = max_length
        self.random = random
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys

    def __call__(self, item):
        new_item = item.copy()
        for key in self.keys:
            graph = item[key]
            if graph.num_residue > self.truncate_length:
                if self.random:
                    start = torch.randint(graph.num_residue - self.truncate_length, (1,)).item()
                else:
                    start = 0
                end = start + self.truncate_length
                mask = torch.zeros(graph.num_residue, dtype=torch.bool, device=graph.device)
                mask[start:end] = True
                graph = graph.subresidue(mask)

            new_item[key] = graph
        return new_item


@R.register("transforms.ProteinView")
class ProteinView(core.Configurable):
    """
    Convert proteins to a specific view.

    Parameters:
        view (str): protein view. Can be ``atom`` or ``residue``.
        keys (str or list of str, optional): keys for the items that require view change in a sample
    """

    def __init__(self, view, keys="graph"):
        self.view = view
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys

    def __call__(self, item):
        item = item.copy()
        for key in self.keys:
            graph = copy.copy(item[key])
            graph.view = self.view
            item[key] = graph
        return item


@R.register("transforms.Compose")
class Compose(core.Configurable):
    """
    Compose a list of transforms into one.

    Parameters:
        transforms (list of callable): list of transforms
    """

    def __init__(self, transforms):
        # flatten recursive composition
        new_transforms = []
        for transform in transforms:
            if isinstance(transform, Compose):
                new_transforms += transform.transforms
            elif transform is not None:
                new_transforms.append(transform)
        self.transforms = new_transforms

    def __call__(self, item):
        for transform in self.transforms:
            item = transform(item)
        return item
