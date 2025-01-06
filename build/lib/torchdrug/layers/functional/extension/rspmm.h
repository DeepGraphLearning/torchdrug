#pragma once

#include <tuple>

#include <torch/extension.h>
#include <ATen/SparseTensorUtils.h>

namespace at {

using namespace at::sparse;

void rspmm_forward_check(CheckedFrom c, const TensorArg &sparse_arg, const TensorArg &relation_arg,
                         const TensorArg &input_arg);

void rspmm_backward_check(CheckedFrom c, const TensorArg &sparse_arg, const TensorArg &relation_arg,
                          const TensorArg &input_arg, const TensorArg &output_arg, const TensorArg &output_grad_arg);

std::tuple<Tensor, Tensor, Tensor, Tensor> coo2csr3d(const SparseTensor &sparse);

SparseTensor csr2coo3d(const Tensor &row_ptr, const Tensor &col_ind, const Tensor &layer_ind, const Tensor &value,
                       IntArrayRef size);

Tensor rspmm_add_mul_forward_cpu(const SparseTensor &sparse, const Tensor &relation, const Tensor &input);

std::tuple<SparseTensor, Tensor, Tensor> rspmm_add_mul_backward_cpu(const SparseTensor &sparse,
        const Tensor &relation, const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor rspmm_min_mul_forward_cpu(const SparseTensor &sparse, const Tensor &relation, const Tensor &input);

std::tuple<SparseTensor, Tensor, Tensor> rspmm_min_mul_backward_cpu(const SparseTensor &sparse,
        const Tensor &relation, const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor rspmm_max_mul_forward_cpu(const SparseTensor &sparse, const Tensor &relation, const Tensor &input);

std::tuple<SparseTensor, Tensor, Tensor> rspmm_max_mul_backward_cpu(const SparseTensor &sparse,
        const Tensor &relation, const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor rspmm_add_add_forward_cpu(const SparseTensor &sparse, const Tensor &relation, const Tensor &input);

std::tuple<SparseTensor, Tensor, Tensor> rspmm_add_add_backward_cpu(const SparseTensor &sparse,
        const Tensor &relation, const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor rspmm_min_add_forward_cpu(const SparseTensor &sparse, const Tensor &relation, const Tensor &input);

std::tuple<SparseTensor, Tensor, Tensor> rspmm_min_add_backward_cpu(const SparseTensor &sparse,
        const Tensor &relation, const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor rspmm_max_add_forward_cpu(const SparseTensor &sparse, const Tensor &relation, const Tensor &input);

std::tuple<SparseTensor, Tensor, Tensor> rspmm_max_add_backward_cpu(const SparseTensor &sparse,
        const Tensor &relation, const Tensor &input, const Tensor &output, const Tensor &output_grad);

#ifdef CUDA_OP
Tensor rspmm_add_mul_forward_cuda(const SparseTensor &sparse, const Tensor &relation, const Tensor &input);

std::tuple<SparseTensor, Tensor, Tensor> rspmm_add_mul_backward_cuda(const SparseTensor &sparse,
        const Tensor &relation, const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor rspmm_min_mul_forward_cuda(const SparseTensor &sparse, const Tensor &relation, const Tensor &input);

std::tuple<SparseTensor, Tensor, Tensor> rspmm_min_mul_backward_cuda(const SparseTensor &sparse,
        const Tensor &relation, const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor rspmm_max_mul_forward_cuda(const SparseTensor &sparse, const Tensor &relation, const Tensor &input);

std::tuple<SparseTensor, Tensor, Tensor> rspmm_max_mul_backward_cuda(const SparseTensor &sparse,
        const Tensor &relation, const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor rspmm_add_add_forward_cuda(const SparseTensor &sparse, const Tensor &relation, const Tensor &input);

std::tuple<SparseTensor, Tensor, Tensor> rspmm_add_add_backward_cuda(const SparseTensor &sparse,
        const Tensor &relation, const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor rspmm_min_add_forward_cuda(const SparseTensor &sparse, const Tensor &relation, const Tensor &input);

std::tuple<SparseTensor, Tensor, Tensor> rspmm_min_add_backward_cuda(const SparseTensor &sparse,
        const Tensor &relation, const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor rspmm_max_add_forward_cuda(const SparseTensor &sparse, const Tensor &relation, const Tensor &input);

std::tuple<SparseTensor, Tensor, Tensor> rspmm_max_add_backward_cuda(const SparseTensor &sparse,
        const Tensor &relation, const Tensor &input, const Tensor &output, const Tensor &output_grad);
#endif

} // namespace at