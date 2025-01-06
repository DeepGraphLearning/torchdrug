#include <mutex>

#include <ATen/Parallel.h>

#include "operator.cuh"
#include "rspmm.h"

namespace at {

// In PyTorch 1.4.0, parallel_for depends on some functions from at::internal in ATen/Parallel.h
// which are not explicitly included
// This is fixed in some new PyTorch release
using namespace at::internal;

void rspmm_forward_check(CheckedFrom c, const TensorArg &sparse_arg, const TensorArg &relation_arg,
                         const TensorArg &input_arg) {
    TORCH_CHECK(sparse_arg->sparse_dim() == 3,
        "Expected 3-dimensional sparse tensor, but got ", sparse_arg->sparse_dim(),
        "-dimensional tensor for ", sparse_arg," (while checking arguments for ", c, ")");
    TORCH_CHECK(sparse_arg->dense_dim() == 0,
        "Expected 3-dimensional sparse tensor, but got ", sparse_arg->dense_dim(),
        " dense dimensions for", sparse_arg," (while checking arguments for ", c, ")");
    checkDim(c, relation_arg, 2);
    checkDim(c, input_arg, 2);
    checkScalarType(c, input_arg, sparse_arg->scalar_type());
    checkSameType(c, relation_arg, input_arg);
    checkSize(c, input_arg, 0, sparse_arg->size(1));
    checkSize(c, relation_arg, {sparse_arg->size(2), input_arg->size(1)});
}

void rspmm_backward_check(CheckedFrom c, const TensorArg &sparse_arg, const TensorArg &relation_arg,
                          const TensorArg &input_arg, const TensorArg &output_arg, const TensorArg &output_grad_arg) {
    rspmm_forward_check(c, sparse_arg, relation_arg, input_arg);
    checkDim(c, output_arg, 2);
    checkSameSize(c, output_arg, output_grad_arg);
    checkSameType(c, input_arg, output_arg);
    checkSameType(c, input_arg, output_grad_arg);
    checkSize(c, output_arg, {sparse_arg->size(0), input_arg->size(1)});
}

std::tuple<Tensor, Tensor, Tensor, Tensor> coo2csr3d(const SparseTensor &sparse) {
    TORCH_CHECK(sparse.is_coalesced(), "Expect coalesced sparse tensor");
    Tensor index = sparse.indices();
    Tensor row_ind = index.select(0, 0);
    Tensor col_ind = index.select(0, 1);
    Tensor layer_ind = index.select(0, 2);
    Tensor value = sparse.values();
    // scatter_add is super slow for int64, due to non-hardware atomic operations
    // use int32 instead
    Tensor nnz_per_row = at::zeros({sparse.size(0)}, row_ind.options().dtype(at::ScalarType::Int));
    nnz_per_row.scatter_add_(0, row_ind, at::ones(row_ind.sizes(), nnz_per_row.options()));
    nnz_per_row = nnz_per_row.toType(at::ScalarType::Long);
    Tensor row_ptr = nnz_per_row.cumsum(0) - nnz_per_row;
    return std::make_tuple(row_ptr, col_ind, layer_ind, value);
}

SparseTensor csr2coo3d(const Tensor &row_ptr_, const Tensor &col_ind, const Tensor &layer_ind, const Tensor &value,
                       IntArrayRef size) {
    Tensor row_ptr = row_ptr_.masked_select(row_ptr_ < col_ind.size(0));
    // scatter_add is super slow for int64, due to non-hardware atomic operations
    // use int32 instead
    Tensor row_ind = at::zeros(col_ind.sizes(), col_ind.options().dtype(at::ScalarType::Int));
    row_ind.scatter_add_(0, row_ptr, at::ones(row_ptr.sizes(), row_ind.options()));
    row_ind = row_ind.toType(at::ScalarType::Long);
    row_ind = row_ind.cumsum(0) - 1;
    Tensor index = at::stack({row_ind, col_ind, layer_ind}, 0);
    return at::_sparse_coo_tensor_unsafe(index, value, size, value.options().layout(kSparse));
}

template <class scalar_t, class NaryOp, class BinaryOp>
void rspmm_forward_out_cpu(const int64_t *row_ptr, const int64_t *col_ind, const int64_t *layer_ind,
                           const scalar_t *value, const scalar_t *relation, const scalar_t *input,
                           scalar_t *output,
                           int64_t num_row, int64_t nnz, int64_t dim) {
    parallel_for(0, num_row, 0, [&](int64_t row_start, int64_t row_end) {
        for (int64_t row = row_start; row < row_end; row++) {
            for (int64_t d = 0; d < dim; d++)
                output[row * dim + d] = NaryOp::zero;

            int64_t ptr_start = row_ptr[row];
            int64_t ptr_end = row + 1 < num_row ? row_ptr[row + 1] : nnz;
            for (int64_t ptr = ptr_start; ptr < ptr_end; ptr++) {
                int64_t col = col_ind[ptr];
                int64_t layer = layer_ind[ptr];
                scalar_t val = value[ptr];
                for (int64_t d = 0; d < dim; d++) {
                    scalar_t x = BinaryOp::forward(relation[layer * dim + d], input[col * dim + d]);
                    scalar_t y = val * x;
                    scalar_t &out = output[row * dim + d];
                    out = NaryOp::forward(out, y);
                }
            }
        }
    });
}

template <class scalar_t, class NaryOp, class BinaryOp>
void rspmm_backward_out_cpu(const int64_t *row_ptr, const int64_t *col_ind, const int64_t *layer_ind,
                            const scalar_t *value, const scalar_t *relation, const scalar_t *input,
                            const scalar_t *output, const scalar_t *output_grad,
                            scalar_t *value_grad, scalar_t *relation_grad, scalar_t *input_grad,
                            int64_t num_row, int64_t nnz, int64_t dim,
                            std::vector<std::mutex> &relation_mutex, std::vector<std::mutex> &input_mutex) {
    parallel_for(0, num_row, 0, [&](int64_t row_start, int64_t row_end) {
        for (int64_t row = row_start; row < row_end; row++) {
            int64_t ptr_start = row_ptr[row];
            int64_t ptr_end = row + 1 < num_row ? row_ptr[row + 1] : nnz;
            for (int64_t ptr = ptr_start; ptr < ptr_end; ptr++) {
                int64_t col = col_ind[ptr];
                int64_t layer = layer_ind[ptr];
                scalar_t val = value[ptr];
                scalar_t val_grad = 0;
                for (int64_t d = 0; d < dim; d++) {
                    scalar_t rel = relation[layer * dim + d];
                    scalar_t in = input[col * dim + d];
                    scalar_t out = output[row * dim + d];
                    scalar_t out_grad = output_grad[row * dim + d];
                    scalar_t x = BinaryOp::forward(rel, in);
                    scalar_t y = val * x;
                    scalar_t dx_drel = BinaryOp::backward_lhs(rel, in);
                    scalar_t dx_din = BinaryOp::backward_rhs(rel, in);
                    scalar_t dout_dy = NaryOp::backward(out, y);
                    scalar_t dy_dval = x;
                    scalar_t dy_dx = val;
                    val_grad += out_grad * dout_dy * dy_dval;
                    {
                        std::lock_guard<std::mutex> lock(relation_mutex[layer * dim + d]);
                        relation_grad[layer * dim + d] += out_grad * dout_dy * dy_dx * dx_drel;
                    }
                    {
                        std::lock_guard<std::mutex> lock(input_mutex[col * dim + d]);
                        input_grad[col * dim + d] += out_grad * dout_dy * dy_dx * dx_din;
                    }
                }
                value_grad[ptr] = val_grad;
            }
        }
    });
}

template <template<class> class NaryOp, template<class> class BinaryOp>
Tensor rspmm_forward_cpu(const SparseTensor &sparse, const Tensor &relation_, const Tensor &input_) {
    constexpr const char *fn_name = "rspmm_forward_cpu";
    TensorArg sparse_arg(sparse, "sparse", 1), relation_arg(relation_, "relation", 2), input_arg(input_, "input", 3);

    rspmm_forward_check(fn_name, sparse_arg, relation_arg, input_arg);
    checkDeviceType(fn_name, {sparse, relation_, input_}, kCPU);

    const Tensor relation = relation_.contiguous();
    const Tensor input = input_.contiguous();

    int64_t nnz = sparse._nnz();
    int64_t dim = input.size(1);
    int64_t num_row = sparse.size(0);
    Tensor output = at::empty({num_row, dim}, input.options());

    auto csr = coo2csr3d(sparse);
    Tensor row_ptr = std::get<0>(csr);
    Tensor col_ind = std::get<1>(csr);
    Tensor layer_ind = std::get<2>(csr);
    Tensor value = std::get<3>(csr);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rspmm_forward_cpu", [&] {
        rspmm_forward_out_cpu<scalar_t, NaryOp<scalar_t>, BinaryOp<scalar_t>>(
            row_ptr.data_ptr<int64_t>(),
            col_ind.data_ptr<int64_t>(),
            layer_ind.data_ptr<int64_t>(),
            value.data_ptr<scalar_t>(),
            relation.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_row, nnz, dim
        );
    });

    return output;
}

template <template<class> class NaryOp, template<class> class BinaryOp>
std::tuple<SparseTensor, Tensor, Tensor> rspmm_backward_cpu(
        const SparseTensor &sparse, const Tensor &relation_, const Tensor &input_, const Tensor &output_,
        const Tensor &output_grad_) {
    constexpr const char *fn_name = "rspmm_backward_cpu";
    TensorArg sparse_arg(sparse, "sparse", 1), relation_arg(relation_, "relation", 2), input_arg(input_, "input", 3),
              output_arg(output_, "output", 4), output_grad_arg(output_grad_, "output_grad", 5);

    rspmm_backward_check(fn_name, sparse_arg, relation_arg, input_arg, output_arg, output_grad_arg);
    checkDeviceType(fn_name, {sparse, relation_, input_, output_, output_grad_}, kCPU);

    const Tensor relation = relation_.contiguous();
    const Tensor input = input_.contiguous();
    const Tensor output = output_.contiguous();
    const Tensor output_grad = output_grad_.contiguous();

    int64_t nnz = sparse._nnz();
    int64_t dim = input.size(1);
    int64_t num_row = sparse.size(0);
    Tensor value_grad = at::zeros_like(sparse.values());
    Tensor relation_grad = at::zeros_like(relation);
    Tensor input_grad = at::zeros_like(input);
    SparseTensor sparse_grad = at::_sparse_coo_tensor_unsafe(sparse.indices(), value_grad, sparse.sizes());

    auto csr = coo2csr3d(sparse);
    Tensor row_ptr = std::get<0>(csr);
    Tensor col_ind = std::get<1>(csr);
    Tensor layer_ind = std::get<2>(csr);
    Tensor value = std::get<3>(csr);
    std::vector<std::mutex> relation_mutex(relation.numel());
    std::vector<std::mutex> input_mutex(input.numel());

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rspmm_backward_cpu", [&] {
        rspmm_backward_out_cpu<scalar_t, NaryOp<scalar_t>, BinaryOp<scalar_t>>(
            row_ptr.data_ptr<int64_t>(),
            col_ind.data_ptr<int64_t>(),
            layer_ind.data_ptr<int64_t>(),
            value.data_ptr<scalar_t>(),
            relation.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            output_grad.data_ptr<scalar_t>(),
            value_grad.data_ptr<scalar_t>(),
            relation_grad.data_ptr<scalar_t>(),
            input_grad.data_ptr<scalar_t>(),
            num_row, nnz, dim,
            relation_mutex, input_mutex
        );
    });

    return std::make_tuple(sparse_grad, relation_grad, input_grad);
}

#define DECLARE_FORWARD_IMPL(ADD, MUL, NARYOP, BINARYOP) \
    Tensor rspmm_##ADD##_##MUL##_forward_cpu(                                          \
            const SparseTensor &sparse, const Tensor &relation, const Tensor &input) { \
        return rspmm_forward_cpu<NARYOP, BINARYOP>(sparse, relation, input);           \
    }

#define DECLARE_BACKWARD_IMPL(ADD, MUL, NARYOP, BINARYOP) \
    std::tuple<SparseTensor, Tensor, Tensor> rspmm_##ADD##_##MUL##_backward_cpu(                                   \
            const SparseTensor &sparse, const Tensor &relation, const Tensor &input, const Tensor &output, \
            const Tensor &output_grad) {                                                                   \
        return rspmm_backward_cpu<NARYOP, BINARYOP>(sparse, relation, input, output, output_grad);         \
    }

DECLARE_FORWARD_IMPL(add, mul, NaryAdd, BinaryMul)
DECLARE_BACKWARD_IMPL(add, mul, NaryAdd, BinaryMul)

DECLARE_FORWARD_IMPL(min, mul, NaryMin, BinaryMul)
DECLARE_BACKWARD_IMPL(min, mul, NaryMin, BinaryMul)

DECLARE_FORWARD_IMPL(max, mul, NaryMax, BinaryMul)
DECLARE_BACKWARD_IMPL(max, mul, NaryMax, BinaryMul)

DECLARE_FORWARD_IMPL(add, add, NaryAdd, BinaryAdd)
DECLARE_BACKWARD_IMPL(add, add, NaryAdd, BinaryAdd)

DECLARE_FORWARD_IMPL(min, add, NaryMin, BinaryAdd)
DECLARE_BACKWARD_IMPL(min, add, NaryMin, BinaryAdd)

DECLARE_FORWARD_IMPL(max, add, NaryMax, BinaryAdd)
DECLARE_BACKWARD_IMPL(max, add, NaryMax, BinaryAdd)

} // namespace at