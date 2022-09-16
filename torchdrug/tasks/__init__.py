from .task import Task

from .property_prediction import PropertyPrediction, MultipleBinaryClassification, \
    NodePropertyPrediction, InteractionPrediction, Unsupervised
from .pretrain import EdgePrediction, AttributeMasking, ContextPrediction, DistancePrediction, \
    AnglePrediction, DihedralPrediction
from .generation import AutoregressiveGeneration, GCPNGeneration
from .retrosynthesis import CenterIdentification, SynthonCompletion, Retrosynthesis
from .reasoning import KnowledgeGraphCompletion
from .contact_prediction import ContactPrediction


_criterion_name = {
    "mse": "mean squared error",
    "mae": "mean absolute error",
    "bce": "binary cross entropy",
    "ce": "cross entropy",
}

_metric_name = {
    "mae": "mean absolute error",
    "mse": "mean squared error",
    "rmse": "root mean squared error",
    "acc": "accuracy",
    "mcc": "matthews correlation coefficient",
}


def _get_criterion_name(criterion):
    if criterion in _criterion_name:
        return _criterion_name[criterion]
    return "%s loss" % criterion


def _get_metric_name(metric):
    if metric in _metric_name:
        return _metric_name[metric]
    return metric


__all__ = [
    "PropertyPrediction", "MultipleBinaryClassification", "NodePropertyPrediction", "InteractionPrediction",
    "Unsupervised",
    "EdgePrediction", "AttributeMasking", "ContextPrediction", "DistancePrediction", "AnglePrediction",
    "DihedralPrediction",
    "AutoregressiveGeneration", "GCPNGeneration",
    "CenterIdentification", "SynthonCompletion", "Retrosynthesis",
    "KnowledgeGraphCompletion",
    "ContactPrediction",
]