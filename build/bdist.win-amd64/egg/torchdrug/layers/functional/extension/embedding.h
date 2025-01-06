#pragma once

#include <tuple>

#include <torch/extension.h>

namespace at {

void embedding_forward_check(CheckedFrom c, const TensorArg &entity_arg, const TensorArg &relation_arg,
                             const TensorArg &h_index_arg, const TensorArg &t_index_arg, const TensorArg &r_index_arg);

void embedding_backward_check(CheckedFrom c, const TensorArg &entity_arg, const TensorArg &relation_arg,
                              const TensorArg &h_index_arg, const TensorArg &t_index_arg, const TensorArg &r_index_arg,
                              const TensorArg &score_grad_arg);

Tensor transe_forward_cpu(const Tensor &entity_, const Tensor &relation_,
                          const Tensor &h_index_, const Tensor &t_index_, const Tensor &r_index_);

std::tuple<Tensor, Tensor> transe_backward_cpu(
        const Tensor &entity_, const Tensor &relation_,
        const Tensor &h_index_, const Tensor &t_index_, const Tensor &r_index_, const Tensor &score_grad_);

Tensor distmult_forward_cpu(const Tensor &entity_, const Tensor &relation_,
                            const Tensor &h_index_, const Tensor &t_index_, const Tensor &r_index_);

std::tuple<Tensor, Tensor> distmult_backward_cpu(
        const Tensor &entity_, const Tensor &relation_,
        const Tensor &h_index_, const Tensor &t_index_, const Tensor &r_index_, const Tensor &score_grad_);

Tensor complex_forward_cpu(const Tensor &entity_, const Tensor &relation_,
                           const Tensor &h_index_, const Tensor &t_index_, const Tensor &r_index_);

std::tuple<Tensor, Tensor> complex_backward_cpu(
        const Tensor &entity_, const Tensor &relation_,
        const Tensor &h_index_, const Tensor &t_index_, const Tensor &r_index_, const Tensor &score_grad_);

Tensor rotate_forward_cpu(const Tensor &entity_, const Tensor &relation_, const Tensor &h_index_,
                          const Tensor &t_index_, const Tensor &r_index_);

std::tuple<Tensor, Tensor> rotate_backward_cpu(
        const Tensor &entity_, const Tensor &relation_,
        const Tensor &h_index_, const Tensor &t_index_, const Tensor &r_index_, const Tensor &score_grad_);

Tensor simple_forward_cpu(const Tensor &entity_, const Tensor &relation_, const Tensor &h_index_,
                          const Tensor &t_index_, const Tensor &r_index_);

std::tuple<Tensor, Tensor> simple_backward_cpu(
        const Tensor &entity_, const Tensor &relation_,
        const Tensor &h_index_, const Tensor &t_index_, const Tensor &r_index_, const Tensor &score_grad_);

#ifdef CUDA_OP
Tensor transe_forward_cuda(const Tensor &entity_, const Tensor &relation_,
                           const Tensor &h_index_, const Tensor &t_index_, const Tensor &r_index_);

std::tuple<Tensor, Tensor> transe_backward_cuda(
        const Tensor &entity, const Tensor &relation_,
        const Tensor &h_index, const Tensor &t_index_, const Tensor &r_index_, const Tensor &score_grad_);

Tensor distmult_forward_cuda(const Tensor &entity_, const Tensor &relation_,
                             const Tensor &h_index_, const Tensor &t_index_, const Tensor &r_index_);

std::tuple<Tensor, Tensor> distmult_backward_cuda(
        const Tensor &entity, const Tensor &relation_,
        const Tensor &h_index, const Tensor &t_index_, const Tensor &r_index_, const Tensor &score_grad_);

Tensor complex_forward_cuda(const Tensor &entity_, const Tensor &relation_,
                            const Tensor &h_index_, const Tensor &t_index_, const Tensor &r_index_);

std::tuple<Tensor, Tensor> complex_backward_cuda(
        const Tensor &entity, const Tensor &relation_,
        const Tensor &h_index, const Tensor &t_index_, const Tensor &r_index_, const Tensor &score_grad_);

Tensor rotate_forward_cuda(const Tensor &entity_, const Tensor &relation_,
                           const Tensor &h_index_, const Tensor &t_index_, const Tensor &r_index_);

std::tuple<Tensor, Tensor> rotate_backward_cuda(
        const Tensor &entity_, const Tensor &relation_,
        const Tensor &h_index_, const Tensor &t_index_, const Tensor &r_index_, const Tensor &score_grad_);

Tensor simple_forward_cuda(const Tensor &entity_, const Tensor &relation_,
                           const Tensor &h_index_, const Tensor &t_index_, const Tensor &r_index_);

std::tuple<Tensor, Tensor> simple_backward_cuda(
        const Tensor &entity_, const Tensor &relation_,
        const Tensor &h_index_, const Tensor &t_index_, const Tensor &r_index_, const Tensor &score_grad_);
#endif

} // namespace at