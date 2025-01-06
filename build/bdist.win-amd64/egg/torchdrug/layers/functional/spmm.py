import os
import sys

import torch
from torch import autograd

from torchdrug import utils

module = sys.modules[__name__]

path = os.path.join(os.path.dirname(__file__), "extension")
spmm = utils.load_extension("spmm", [os.path.join(path, "spmm.cpp"), os.path.join(path, "rspmm.cpp"),
                                     os.path.join(path, "spmm.cu"), os.path.join(path, "rspmm.cu")])


class SPMMAddMulFunction(autograd.Function):

    @staticmethod
    def forward(ctx, sparse, input):
        assert sparse.is_coalesced()
        if input.device.type == "cuda":
            forward = spmm.spmm_add_mul_forward_cuda
        else:
            forward = spmm.spmm_add_mul_forward_cpu
        output = forward(sparse, input)
        ctx.save_for_backward(sparse, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = spmm.spmm_add_mul_backward_cuda
        else:
            backward = spmm.spmm_add_mul_backward_cpu
        sparse_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        if not ctx.saved_tensors[0].requires_grad:
            sparse_grad = None
        return sparse_grad, input_grad


class SPMMMinMulFunction(autograd.Function):

    @staticmethod
    def forward(ctx, sparse, input):
        assert sparse.is_coalesced()
        if input.device.type == "cuda":
            forward = spmm.spmm_min_mul_forward_cuda
        else:
            forward = spmm.spmm_min_mul_forward_cpu
        output = forward(sparse, input)
        ctx.save_for_backward(sparse, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = spmm.spmm_min_mul_backward_cuda
        else:
            backward = spmm.spmm_min_mul_backward_cpu
        sparse_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        if not ctx.saved_tensors[0].requires_grad:
            sparse_grad = None
        return sparse_grad, input_grad


class SPMMMaxMulFunction(autograd.Function):

    @staticmethod
    def forward(ctx, sparse, input):
        assert sparse.is_coalesced()
        if input.device.type == "cuda":
            forward = spmm.spmm_max_mul_forward_cuda
        else:
            forward = spmm.spmm_max_mul_forward_cpu
        output = forward(sparse, input)
        ctx.save_for_backward(sparse, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = spmm.spmm_max_mul_backward_cuda
        else:
            backward = spmm.spmm_max_mul_backward_cpu
        sparse_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        if not ctx.saved_tensors[0].requires_grad:
            sparse_grad = None
        return sparse_grad, input_grad


class SPMMAddAddFunction(autograd.Function):

    @staticmethod
    def forward(ctx, sparse, input):
        assert sparse.is_coalesced()
        if input.device.type == "cuda":
            forward = spmm.spmm_add_add_forward_cuda
        else:
            forward = spmm.spmm_add_add_forward_cpu
        output = forward(sparse, input)
        ctx.save_for_backward(sparse, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = spmm.spmm_add_add_backward_cuda
        else:
            backward = spmm.spmm_add_add_backward_cpu
        sparse_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        if not ctx.saved_tensors[0].requires_grad:
            sparse_grad = None
        return sparse_grad, input_grad


class SPMMMinAddFunction(autograd.Function):

    @staticmethod
    def forward(ctx, sparse, input):
        assert sparse.is_coalesced()
        if input.device.type == "cuda":
            forward = spmm.spmm_min_add_forward_cuda
        else:
            forward = spmm.spmm_min_add_forward_cpu
        output = forward(sparse, input)
        ctx.save_for_backward(sparse, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = spmm.spmm_min_add_backward_cuda
        else:
            backward = spmm.spmm_min_add_backward_cpu
        sparse_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        if not ctx.saved_tensors[0].requires_grad:
            sparse_grad = None
        return sparse_grad, input_grad


class SPMMMaxAddFunction(autograd.Function):

    @staticmethod
    def forward(ctx, sparse, input):
        assert sparse.is_coalesced()
        if input.device.type == "cuda":
            forward = spmm.spmm_max_add_forward_cuda
        else:
            forward = spmm.spmm_max_add_forward_cpu
        output = forward(sparse, input)
        ctx.save_for_backward(sparse, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = spmm.spmm_max_add_backward_cuda
        else:
            backward = spmm.spmm_max_add_backward_cpu
        sparse_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        if not ctx.saved_tensors[0].requires_grad:
            sparse_grad = None
        return sparse_grad, input_grad


class RSPMMAddMulFunction(autograd.Function):

    @staticmethod
    def forward(ctx, sparse, relation, input):
        assert sparse.is_coalesced()
        if input.device.type == "cuda":
            forward = spmm.rspmm_add_mul_forward_cuda
        else:
            forward = spmm.rspmm_add_mul_forward_cpu
        output = forward(sparse, relation, input)
        ctx.save_for_backward(sparse, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = spmm.rspmm_add_mul_backward_cuda
        else:
            backward = spmm.rspmm_add_mul_backward_cpu
        sparse_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        if not ctx.saved_tensors[0].requires_grad:
            sparse_grad = None
        return sparse_grad, relation_grad, input_grad


class RSPMMMinMulFunction(autograd.Function):

    @staticmethod
    def forward(ctx, sparse, relation, input):
        assert sparse.is_coalesced()
        if input.device.type == "cuda":
            forward = spmm.rspmm_min_mul_forward_cuda
        else:
            forward = spmm.rspmm_min_mul_forward_cpu
        output = forward(sparse, relation, input)
        ctx.save_for_backward(sparse, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = spmm.rspmm_min_mul_backward_cuda
        else:
            backward = spmm.rspmm_min_mul_backward_cpu
        sparse_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        if not ctx.saved_tensors[0].requires_grad:
            sparse_grad = None
        return sparse_grad, relation_grad, input_grad


class RSPMMMaxMulFunction(autograd.Function):

    @staticmethod
    def forward(ctx, sparse, relation, input):
        assert sparse.is_coalesced()
        if input.device.type == "cuda":
            forward = spmm.rspmm_max_mul_forward_cuda
        else:
            forward = spmm.rspmm_max_mul_forward_cpu
        output = forward(sparse, relation, input)
        ctx.save_for_backward(sparse, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = spmm.rspmm_max_mul_backward_cuda
        else:
            backward = spmm.rspmm_max_mul_backward_cpu
        sparse_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        if not ctx.saved_tensors[0].requires_grad:
            sparse_grad = None
        return sparse_grad, relation_grad, input_grad


class RSPMMAddAddFunction(autograd.Function):

    @staticmethod
    def forward(ctx, sparse, relation, input):
        assert sparse.is_coalesced()
        if input.device.type == "cuda":
            forward = spmm.rspmm_add_add_forward_cuda
        else:
            forward = spmm.rspmm_add_add_forward_cpu
        output = forward(sparse, relation, input)
        ctx.save_for_backward(sparse, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = spmm.rspmm_add_add_backward_cuda
        else:
            backward = spmm.rspmm_add_add_backward_cpu
        sparse_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        if not ctx.saved_tensors[0].requires_grad:
            sparse_grad = None
        return sparse_grad, relation_grad, input_grad


class RSPMMMinAddFunction(autograd.Function):

    @staticmethod
    def forward(ctx, sparse, relation, input):
        assert sparse.is_coalesced()
        if input.device.type == "cuda":
            forward = spmm.rspmm_min_add_forward_cuda
        else:
            forward = spmm.rspmm_min_add_forward_cpu
        output = forward(sparse, relation, input)
        ctx.save_for_backward(sparse, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = spmm.rspmm_min_add_backward_cuda
        else:
            backward = spmm.rspmm_min_add_backward_cpu
        sparse_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        if not ctx.saved_tensors[0].requires_grad:
            sparse_grad = None
        return sparse_grad, relation_grad, input_grad


class RSPMMMaxAddFunction(autograd.Function):

    @staticmethod
    def forward(ctx, sparse, relation, input):
        assert sparse.is_coalesced()
        if input.device.type == "cuda":
            forward = spmm.rspmm_max_add_forward_cuda
        else:
            forward = spmm.rspmm_max_add_forward_cpu
        output = forward(sparse, relation, input)
        ctx.save_for_backward(sparse, relation, input, output)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        if output_grad.device.type == "cuda":
            backward = spmm.rspmm_max_add_backward_cuda
        else:
            backward = spmm.rspmm_max_add_backward_cpu
        sparse_grad, relation_grad, input_grad = backward(*ctx.saved_tensors, output_grad)
        if not ctx.saved_tensors[0].requires_grad:
            sparse_grad = None
        return sparse_grad, relation_grad, input_grad


def generalized_spmm(sparse, input, sum="add", mul="mul"):
    r"""
    Generalized sparse-dense matrix multiplication.

    This function computes the matrix multiplication of a sparse matrix and a dense input matrix.
    The output dense matrix satisfies

    .. math::

        output_{i,k} = \bigoplus_{j: sparse_{i,j} \neq 0} sparse_{i,j} \otimes input_{j,k}

    where :math:`\oplus` and :math:`\otimes` are the summation and the multiplication operators respectively.

    .. warning::

        Gradient w.r.t. the sparse matrix is only computed for non-zero entries of the sparse matrix.
        This behaves differently from dense-dense matrix multiplication with zero entries.

    Parameters:
        sparse (SparseTensor): 2D sparse tensor
        input (Tensor): 2D dense tensor
        sum (str, optional): generalized summation operator. Available operators are ``add``, ``min`` and ``max``.
        mul (str, optional): generalized multiplication operator. Available operators are ``add`` and ``mul``.
    """
    name = "SPMM%s%sFunction" % (sum.capitalize(), mul.capitalize())
    if not hasattr(module, name):
        raise ValueError("No generalized spmm implementation found for summation `%s` and multiplication `%s`"
                         % (sum, mul))
    Function = getattr(module, name)
    return Function.apply(sparse.coalesce(), input)


def generalized_rspmm(sparse, relation, input, sum="add", mul="mul"):
    r"""
    Generalized relational sparse-dense matrix multiplication.

    This function computes the matrix multiplication of a sparse matrix, a dense relation matrix and
    a dense input matrix. The output dense matrix satisfies

    .. math::

        output_{i,l} = \bigoplus_{j,k: sparse_{i,j,k} \neq 0} sparse_{i, j, k} \times (relation_{k,l} \otimes input_{j,l})

    where :math:`\oplus` and :math:`\otimes` are the summation and the multiplication operators respectively.

    .. warning::

        Gradient w.r.t. the sparse matrix is only computed for non-zero entries of the sparse matrix.
        This behaves differently from dense-dense matrix multiplication with zero entries.

    Parameters:
        sparse (SparseTensor): 3D sparse tensor
        relation (Tensor): 2D dense tensor
        input (Tensor): 2D dense tensor
        sum (str, optional): generalized summation operator. Available operators are ``add``, ``min`` and ``max``.
        mul (str, optional): generalized multiplication operator. Available operators are ``add`` and ``mul``.
    """
    name = "RSPMM%s%sFunction" % (sum.capitalize(), mul.capitalize())
    if not hasattr(module, name):
        raise ValueError("No generalized rspmm implementation found for summation `%s` and multiplication `%s`"
                         % (sum, mul))
    Function = getattr(module, name)
    return Function.apply(sparse.coalesce(), relation, input)