import os

import torch
from torch import autograd

from torchdrug import utils

backend = "fast"

path = os.path.join(os.path.dirname(__file__), "extension")


if torch.cuda.is_available():
    embedding = utils.load_extension("embedding",
                                     [os.path.join(path, "embedding.cpp"), os.path.join(path, "embedding.cu")],
                                     extra_cflags=["-g", "-Ofast", "-fopenmp", "-DCUDA_OP"], extra_cuda_cflags=["-O3"])
else:
    embedding = utils.load_extension("embedding", [os.path.join(path, "embedding.cpp")],
                                     extra_cflags=["-g", "-Ofast", "-fopenmp"])


class TransEFunction(autograd.Function):

    @staticmethod
    def forward(ctx, entity, relation, h_index, t_index, r_index):
        if entity.device.type == "cuda":
            forward = embedding.transe_forward_cuda
        else:
            forward = embedding.transe_forward_cpu
        score = forward(entity, relation, h_index, t_index, r_index)
        ctx.save_for_backward(entity, relation, h_index, t_index, r_index)
        return score

    @staticmethod
    def backward(ctx, score_grad):
        if score_grad.device.type == "cuda":
            backward = embedding.transe_backward_cuda
        else:
            backward = embedding.transe_backward_cpu
        entity_grad, relation_grad = backward(*ctx.saved_tensors, score_grad)
        return entity_grad, relation_grad, None, None, None


class DistMultFunction(autograd.Function):

    @staticmethod
    def forward(ctx, entity, relation, h_index, t_index, r_index):
        if entity.device.type == "cuda":
            forward = embedding.distmult_forward_cuda
        else:
            forward = embedding.distmult_forward_cpu
        score = forward(entity, relation, h_index, t_index, r_index)
        ctx.save_for_backward(entity, relation, h_index, t_index, r_index)
        return score

    @staticmethod
    def backward(ctx, score_grad):
        if score_grad.device.type == "cuda":
            backward = embedding.distmult_backward_cuda
        else:
            backward = embedding.distmult_backward_cpu
        entity_grad, relation_grad = backward(*ctx.saved_tensors, score_grad)
        return entity_grad, relation_grad, None, None, None


class ComplExFunction(autograd.Function):

    @staticmethod
    def forward(ctx, entity, relation, h_index, t_index, r_index):
        if entity.device.type == "cuda":
            forward = embedding.complex_forward_cuda
        else:
            forward = embedding.complex_forward_cpu
        score = forward(entity, relation, h_index, t_index, r_index)
        ctx.save_for_backward(entity, relation, h_index, t_index, r_index)
        return score

    @staticmethod
    def backward(ctx, score_grad):
        if score_grad.device.type == "cuda":
            backward = embedding.complex_backward_cuda
        else:
            backward = embedding.complex_backward_cpu
        entity_grad, relation_grad = backward(*ctx.saved_tensors, score_grad)
        return entity_grad, relation_grad, None, None, None


class SimplEFunction(autograd.Function):

    @staticmethod
    def forward(ctx, entity, relation, h_index, t_index, r_index):
        if entity.device.type == "cuda":
            forward = embedding.simple_forward_cuda
        else:
            forward = embedding.simple_forward_cpu
        score = forward(entity, relation, h_index, t_index, r_index)
        ctx.save_for_backward(entity, relation, h_index, t_index, r_index)
        return score

    @staticmethod
    def backward(ctx, score_grad):
        if score_grad.device.type == "cuda":
            backward = embedding.simple_backward_cuda
        else:
            backward = embedding.simple_backward_cpu
        entity_grad, relation_grad = backward(*ctx.saved_tensors, score_grad)
        return entity_grad, relation_grad, None, None, None


class RotatEFunction(autograd.Function):

    @staticmethod
    def forward(ctx, entity, relation, h_index, t_index, r_index):
        if entity.device.type == "cuda":
            forward = embedding.rotate_forward_cuda
        else:
            forward = embedding.rotate_forward_cpu
        score = forward(entity, relation, h_index, t_index, r_index)
        ctx.save_for_backward(entity, relation, h_index, t_index, r_index)
        return score

    @staticmethod
    def backward(ctx, score_grad):
        if score_grad.device.type == "cuda":
            backward = embedding.rotate_backward_cuda
        else:
            backward = embedding.rotate_backward_cpu
        entity_grad, relation_grad = backward(*ctx.saved_tensors, score_grad)
        return entity_grad, relation_grad, None, None, None


def transe_score(entity, relation, h_index, t_index, r_index):
    """
    TransE score function from `Translating Embeddings for Modeling Multi-relational Data`_.

    .. _Translating Embeddings for Modeling Multi-relational Data:
        https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf

    Parameters:
        entity (Tensor): entity embeddings of shape :math:`(|V|, d)`
        relation (Tensor): relation embeddings of shape :math:`(|R|, d)`
        h_index (LongTensor): index of head entities
        t_index (LongTensor): index of tail entities
        r_index (LongTensor): index of relations
    """
    if backend == "native":
        h = entity[h_index]
        r = relation[r_index]
        t = entity[t_index]
        score = (h + r - t).norm(p=1, dim=-1)
    elif backend == "fast":
        score = TransEFunction.apply(entity, relation, h_index, t_index, r_index)
    else:
        raise ValueError("Unknown embedding backend `%s`" % backend)
    return score


def distmult_score(entity, relation, h_index, t_index, r_index):
    """
    DistMult score function from `Embedding Entities and Relations for Learning and Inference in Knowledge Bases`_.

    .. _Embedding Entities and Relations for Learning and Inference in Knowledge Bases:
        https://arxiv.org/pdf/1412.6575.pdf

    Parameters:
        entity (Tensor): entity embeddings of shape :math:`(|V|, d)`
        relation (Tensor): relation embeddings of shape :math:`(|R|, d)`
        h_index (LongTensor): index of head entities
        t_index (LongTensor): index of tail entities
        r_index (LongTensor): index of relations
    """
    if backend == "native":
        h = entity[h_index]
        r = relation[r_index]
        t = entity[t_index]
        score = (h * r * t).sum(dim=-1)
    elif backend == "fast":
        score = DistMultFunction.apply(entity, relation, h_index, t_index, r_index)
    else:
        raise ValueError("Unknown embedding backend `%s`" % backend)
    return score


def complex_score(entity, relation, h_index, t_index, r_index):
    """
    ComplEx score function from `Complex Embeddings for Simple Link Prediction`_.

    .. _Complex Embeddings for Simple Link Prediction:
        http://proceedings.mlr.press/v48/trouillon16.pdf

    Parameters:
        entity (Tensor): entity embeddings of shape :math:`(|V|, 2d)`
        relation (Tensor): relation embeddings of shape :math:`(|R|, 2d)`
        h_index (LongTensor): index of head entities
        t_index (LongTensor): index of tail entities
        r_index (LongTensor): index of relations
    """
    if backend == "native":
        h = entity[h_index]
        r = relation[r_index]
        t = entity[t_index]

        h_re, h_im = h.chunk(2, dim=-1)
        r_re, r_im = r.chunk(2, dim=-1)
        t_re, t_im = t.chunk(2, dim=-1)

        x_re = h_re * r_re - h_im * r_im
        x_im = h_re * r_im + h_im * r_re
        x = x_re * t_re + x_im * t_im
        score = x.sum(dim=-1)
    elif backend == "fast":
        score = ComplExFunction.apply(entity, relation, h_index, t_index, r_index)
    else:
        raise ValueError("Unknown embedding backend `%s`" % backend)
    return score


def simple_score(entity, relation, h_index, t_index, r_index):
    """
    SimplE score function from `SimplE Embedding for Link Prediction in Knowledge Graphs`_.

    .. _SimplE Embedding for Link Prediction in Knowledge Graphs:
        https://papers.nips.cc/paper/2018/file/b2ab001909a8a6f04b51920306046ce5-Paper.pdf

    Parameters:
        entity (Tensor): entity embeddings of shape :math:`(|V|, 2d)`
        relation (Tensor): relation embeddings of shape :math:`(|R|, d)`
        h_index (LongTensor): index of head entities
        t_index (LongTensor): index of tail entities
        r_index (LongTensor): index of relations
    """
    if backend == "native":
        h = entity[h_index]
        r = relation[r_index]
        t = entity[t_index]
        t_flipped = torch.cat(t.chunk(2, dim=-1)[::-1], dim=-1)
        score = (h * r * t_flipped).sum(dim=-1)
    elif backend == "fast":
        score = SimplEFunction.apply(entity, relation, h_index, t_index, r_index)
    else:
        raise ValueError("Unknown embedding backend `%s`" % backend)
    return score


def rotate_score(entity, relation, h_index, t_index, r_index):
    """
    RotatE score function from `RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space`_.

    .. _RotatE\: Knowledge Graph Embedding by Relational Rotation in Complex Space:
        https://arxiv.org/pdf/1902.10197.pdf

    Parameters:
        entity (Tensor): entity embeddings of shape :math:`(|V|, 2d)`
        relation (Tensor): relation embeddings of shape :math:`(|R|, d)`
        h_index (LongTensor): index of head entities
        t_index (LongTensor): index of tail entities
        r_index (LongTensor): index of relations
    """
    if backend == "native":
        h = entity[h_index]
        r = relation[r_index]
        t = entity[t_index]

        h_re, h_im = h.chunk(2, dim=-1)
        r_re, r_im = torch.cos(r), torch.sin(r)
        t_re, t_im = t.chunk(2, dim=-1)

        x_re = h_re * r_re - h_im * r_im - t_re
        x_im = h_re * r_im + h_im * r_re - t_im
        x = torch.stack([x_re, x_im], dim=-1)
        score = x.norm(p=2, dim=-1).sum(dim=-1)
    elif backend == "fast":
        score = RotatEFunction.apply(entity, relation, h_index, t_index, r_index)
    else:
        raise ValueError("Unknown embedding backend `%s`" % backend)
    return score