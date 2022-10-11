import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core, layers, tasks, metrics
from torchdrug.core import Registry as R
from torchdrug.layers import functional


@R.register("tasks.ContactPrediction")
class ContactPrediction(tasks.Task, core.Configurable):
    """
    Predict whether each amino acid pair contact or not in the folding structure.

    Parameters:
        model (nn.Module): protein sequence representation model
        max_length (int, optional): maximal length of sequence. Truncate the sequence if it exceeds this limit.
        random_truncate (bool, optional): truncate the sequence at a random position.
            If not, truncate the suffix of the sequence.
        threshold (float, optional): distance threshold for contact
        gap (int, optional): sequential distance cutoff for evaluation
        criterion (str or dict, optional): training criterion. For dict, the key is criterion and the value
            is the corresponding weight. Available criterion is ``bce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``accuracy``, ``prec@Lk`` and ``prec@k``.
        num_mlp_layer (int, optional): number of layers in mlp prediction head
        verbose (int, optional): output verbose level
    """

    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(self, model, max_length=500, random_truncate=True, threshold=8.0, gap=6, criterion="bce",
                 metric=("accuracy", "prec@L5"), num_mlp_layer=1, verbose=0):
        super(ContactPrediction, self).__init__()
        self.model = model
        self.max_length = max_length
        self.random_truncate = random_truncate
        self.threshold = threshold
        self.gap = gap
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        self.verbose = verbose

        if hasattr(self.model, "node_output_dim"):
            model_output_dim = self.model.node_output_dim
        else:
            model_output_dim = self.model.output_dim
        hidden_dims = [model_output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(2 * model_output_dim, hidden_dims + [1])

    def truncate(self, batch):
        graph = batch["graph"]
        size = graph.num_residues
        if (size > self.max_length).any():
            if self.random_truncate:
                starts = (torch.rand(graph.batch_size, device=graph.device) * \
                          (graph.num_residues - self.max_length).clamp(min=0)).long()
                ends = torch.min(starts + self.max_length, graph.num_residues)
                starts = starts + (graph.num_cum_residues - graph.num_residues)
                ends = ends + (graph.num_cum_residues - graph.num_residues)
                mask = functional.multi_slice_mask(starts, ends, graph.num_residue)
            else:
                starts = size.cumsum(0) - size
                size = size.clamp(max=self.max_length)
                ends = starts + size
                mask = functional.multi_slice_mask(starts, ends, graph.num_residue)
            graph = graph.subresidue(mask)

        return {
            "graph": graph
        }

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        batch = self.truncate(batch)
        pred = self.predict(batch, all_loss, metric)
        target = self.target(batch)

        for criterion, weight in self.criterion.items():
            if criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target["label"], reduction="none")
                loss = functional.variadic_mean(loss * target["mask"].float(), size=target["size"])
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            loss = loss.mean()

            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        output = self.model(graph, graph.residue_feature.float(), all_loss=all_loss, metric=metric)
        output = output["residue_feature"]

        range = torch.arange(graph.num_residue, device=self.device)
        node_in, node_out = functional.variadic_meshgrid(range, graph.num_residues, range, graph.num_residues)
        if all_loss is None and node_in.shape[0] > (self.max_length ** 2) * graph.batch_size:
            # test
            # split large input to reduce memory cost
            size = (self.max_length ** 2) * graph.batch_size
            node_in_splits = node_in.split(size, dim=0)
            node_out_splits = node_out.split(size, dim=0)
            pred = []
            for _node_in, _node_out in zip(node_in_splits, node_out_splits):
                prod = output[_node_in] * output[_node_out]
                diff = (output[_node_in] - output[_node_out]).abs()
                pairwise_features = torch.cat((prod, diff), -1)
                _pred = self.mlp(pairwise_features)
                pred.append(_pred)
            pred = torch.cat(pred, dim=0)
        else:
            prod = output[node_in] * output[node_out]
            diff = (output[node_in] - output[node_out]).abs()
            pairwise_features = torch.cat((prod, diff), -1)
            pred = self.mlp(pairwise_features)

        return pred.squeeze(-1)

    def target(self, batch):
        graph = batch["graph"]
        valid_mask = graph.mask
        residue_position = graph.residue_position

        range = torch.arange(graph.num_residue, device=self.device)
        node_in, node_out = functional.variadic_meshgrid(range, graph.num_residues, range, graph.num_residues)
        dist = (residue_position[node_in] - residue_position[node_out]).norm(p=2, dim=-1)
        label = (dist < self.threshold).float()

        mask = valid_mask[node_in] & valid_mask[node_out] & ((node_in - node_out).abs() >= self.gap)

        return {
            "label": label,
            "mask": mask,
            "size": graph.num_residues ** 2
        }

    def evaluate(self, pred, target):
        label = target["label"]
        mask = target["mask"]
        size = functional.variadic_sum(mask.long(), target["size"])
        label = label[mask]
        pred = pred[mask]

        metric = {}
        for _metric in self.metric:
            if _metric == "accuracy":
                score = (pred > 0) == label
                score = functional.variadic_mean(score.float(), size).mean()
            elif _metric.startswith("prec@L"):
                l = target['size'].sqrt().long()
                k = int(_metric[7:]) if len(_metric) > 7 else 1
                score = metrics.variadic_top_precision(pred, label, size, l // k).mean()
            elif _metric.startswith("prec@"):
                k = int(_metric[5:])
                k = torch.full_like(size, k)
                score = metrics.variadic_top_precision(pred, label, size, k).mean()
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric
