import math
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core, tasks, metrics
from torchdrug.core import Registry as R
from torchdrug.layers import functional


@R.register("tasks.PropertyPrediction")
class PropertyPrediction(tasks.Task, core.Configurable):
    """
    Graph / molecule property prediction task.

    This class is also compatible with semi-supervised learning.

    Parameters:
        model (nn.Module): graph representation model
        task (str, list or dict, optional): training task(s).
            For dict, the keys are tasks and the values are the corresponding weights.
        criterion (str, list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``mse`` and ``bce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``mae``, ``rmse``, ``auprc`` and ``auroc``.
        verbose (int, optional): output verbose level
    """

    eps = 1e-10
    _option_members = set(["task", "criterion", "metric"])

    def __init__(self, model, task=(), criterion="mse", metric=("mae", "rmse"), verbose=0):
        super(PropertyPrediction, self).__init__()
        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.verbose = verbose

        self.linear = nn.Linear(model.output_dim, len(task))

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the mean and derivation for each task on the training set.
        """
        values = defaultdict(list)
        for data in train_set:
            if not data.get("labeled", True):
                continue
            for task in self.task:
                if data[task] != math.nan:
                    values[task].append(data[task])
        mean = []
        std = []
        weight = []
        for task, w in self.task.items():
            value = np.array(values[task])
            mean.append(value.mean())
            std.append(value.std())
            weight.append(w)

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)

        if all([t not in batch for t in self.task]):
            # unlabeled data
            return all_loss, metric

        target = self.target(batch)
        labeled = ~torch.isnan(target)
        target[~labeled] = 0

        for criterion, weight in self.criterion.items():
            if criterion == "mse":
                loss = F.mse_loss(pred, (target - self.mean) / self.std, reduction="none")
            elif criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            loss = functional.masked_mean(loss, labeled, dim=0)

            name = tasks._get_criterion_name(criterion)
            if self.verbose > 0:
                for t, l in zip(self.task, loss):
                    metric["%s [%s]" % (name, t)] = l
            loss = (loss * self.weight).sum() / self.weight.sum()
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        pred = self.linear(output["graph_feature"])
        return pred

    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device))
        target[~labeled] = math.nan
        return target

    def evaluate(self, pred, target):
        labeled = ~torch.isnan(target)

        metric = {}
        for _metric in self.metric:
            if _metric == "mae":
                score = F.l1_loss(pred * self.std + self.mean, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0)
            elif _metric == "rmse":
                score = F.mse_loss(pred * self.std + self.mean, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0).sqrt()
            elif _metric == "auroc":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.area_under_roc(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "auprc":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.area_under_prc(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "r2":
                score = []
                new_pred = pred * self.std + self.mean
                for _pred, _target, _labeled in zip(new_pred.t(), target.t(), labeled.t()):
                    _score = metrics.r2(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            for t, s in zip(self.task, score):
                metric["%s [%s]" % (name, t)] = s

        return metric


class Unsupervised(nn.Module):
    """
    Wrapper task for unsupervised learning.

    The unsupervised loss should be computed by the model.

    Parameters:
        model (nn.Module): any model
    """

    def __init__(self, model):
        super(Unsupervised, self).__init__()
        self.model = model

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        pred = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        return pred