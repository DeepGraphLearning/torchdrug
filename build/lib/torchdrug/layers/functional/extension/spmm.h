#pragma once

#include <tuple>

#include <torch/extension.h>
#include <ATen/SparseTensorUtils.h>

#include "rspmm.h"

namespace at {

using namespace at::sparse;

void spmm_forward_check(CheckedFrom c, const TensorArg &sparse_arg, const TensorArg &input_arg);

void spmm_backward_check(CheckedFrom c, const TensorArg &sparse_arg, const TensorArg &input_arg,
                         const TensorArg &output_arg, const TensorArg &output_grad_arg);

std::tuple<Tensor, Tensor, Tensor> coo2csr(const SparseTensor &sparse);

SparseTensor csr2coo(const Tensor &row_ptr_, const Tensor &col_ind, const Tensor &value, IntArrayRef size);

Tensor spmm_add_mul_forward_cpu(const SparseTensor &sparse, const Tensor &input);

std::tuple<SparseTensor, Tensor> spmm_add_mul_backward_cpu(
        const SparseTensor &sparse, const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor spmm_min_mul_forward_cpu(const SparseTensor &sparse, const Tensor &input);

std::tuple<SparseTensor, Tensor> spmm_min_mul_backward_cpu(
        const SparseTensor &sparse, const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor spmm_max_mul_forward_cpu(const SparseTensor &sparse, const Tensor &input);

std::tuple<SparseTensor, Tensor> spmm_max_mul_backward_cpu(
        const SparseTensor &sparse, const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor spmm_add_add_forward_cpu(const SparseTensor &sparse, const Tensor &input);

std::tuple<SparseTensor, Tensor> spmm_add_add_backward_cpu(
        const SparseTensor &sparse, const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor spmm_min_add_forward_cpu(const SparseTensor &sparse, const Tensor &input);

std::tuple<SparseTensor, Tensor> spmm_min_add_backward_cpu(
        const SparseTensor &sparse, const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor spmm_max_add_forward_cpu(const SparseTensor &sparse, const Tensor &input);

std::tuple<SparseTensor, Tensor> spmm_max_add_backward_cpu(
        const SparseTensor &sparse, const Tensor &input, const Tensor &output, const Tensor &output_grad);

#ifdef CUDA_OP
Tensor spmm_add_mul_forward_cuda(const SparseTensor &sparse, const Tensor &input);

std::tuple<SparseTensor, Tensor> spmm_add_mul_backward_cuda(
        const SparseTensor &sparse, const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor spmm_min_mul_forward_cuda(const SparseTensor &sparse, const Tensor &input);

std::tuple<SparseTensor, Tensor> spmm_min_mul_backward_cuda(
        const SparseTensor &sparse, const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor spmm_max_mul_forward_cuda(const SparseTensor &sparse, const Tensor &input);

std::tuple<SparseTensor, Tensor> spmm_max_mul_backward_cuda(
        const SparseTensor &sparse, const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor spmm_add_add_forward_cuda(const SparseTensor &sparse, const Tensor &input);

std::tuple<SparseTensor, Tensor> spmm_add_add_backward_cuda(
        const SparseTensor &sparse, const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor spmm_min_add_forward_cuda(const SparseTensor &sparse, const Tensor &input);

std::tuple<SparseTensor, Tensor> spmm_min_add_backward_cuda(
        const SparseTensor &sparse, const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor spmm_max_add_forward_cuda(const SparseTensor &sparse, const Tensor &input);

std::tuple<SparseTensor, Tensor> spmm_max_add_backward_cuda(
        const SparseTensor &sparse, const Tensor &input, const Tensor &output, const Tensor &output_grad);
#endif

} // namespace at