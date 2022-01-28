#include <mutex>

#include <ATen/Parallel.h>

#include "operator.cuh"
#include "spmm.h"

namespace at {

// In PyTorch 1.4.0, parallel_for depends on some functions from at::internal in ATen/Parallel.h
// which are not explicitly included
// This is fixed in some new PyTorch release
using namespace at::internal;

void spmm_forward_check(CheckedFrom c, const TensorArg &sparse_arg, const TensorArg &input_arg) {
    TORCH_CHECK(sparse_arg->sparse_dim() == 2,
        "Expected 2-dimensional sparse tensor, but got ", sparse_arg->sparse_dim(),
        "-dimensional tensor for ", sparse_arg," (while checking arguments for ", c, ")");
    TORCH_CHECK(sparse_arg->dense_dim() == 0,
        "Expected 2-dimensional sparse tensor, but got ", sparse_arg->dense_dim(),
        " dense dimensions for", sparse_arg," (while checking arguments for ", c, ")");
    checkDim(c, input_arg, 2);
    checkScalarType(c, input_arg, sparse_arg->scalar_type());
    checkSize(c, input_arg, 0, sparse_arg->size(1));
}

void spmm_backward_check(CheckedFrom c, const TensorArg &sparse_arg, const TensorArg &input_arg,
                         const TensorArg &output_arg, const TensorArg &output_grad_arg) {
    spmm_forward_check(c, sparse_arg, input_arg);
    checkDim(c, output_arg, 2);
    checkSameSize(c, output_arg, output_grad_arg);
    checkSameType(c, input_arg, output_arg);
    checkSameType(c, input_arg, output_grad_arg);
    checkSize(c, output_arg, {sparse_arg->size(0), input_arg->size(1)});
}

std::tuple<Tensor, Tensor, Tensor> coo2csr(const SparseTensor &sparse) {
    TORCH_CHECK(sparse.is_coalesced(), "Expect coalesced sparse tensor");
    Tensor index = sparse.indices();
    Tensor row_ind = index.select(0, 0);
    Tensor col_ind = index.select(0, 1);
    Tensor value = sparse.values();
    // scatter_add is super slow for int64, due to non-hardware atomic operations
    // use int32 instead
    Tensor nnz_per_row = at::zeros({sparse.size(0)}, row_ind.options().dtype(at::ScalarType::Int));
    nnz_per_row.scatter_add_(0, row_ind, at::ones(row_ind.sizes(), nnz_per_row.options()));
    nnz_per_row = nnz_per_row.toType(at::ScalarType::Long);
    Tensor row_ptr = nnz_per_row.cumsum(0) - nnz_per_row;
    return std::make_tuple(row_ptr, col_ind, value);
}

SparseTensor csr2coo(const Tensor &row_ptr_, const Tensor &col_ind, const Tensor &value, IntArrayRef size) {
    Tensor row_ptr = row_ptr_.masked_select(row_ptr_ < col_ind.size(0));
    // scatter_add is super slow for int64, due to non-hardware atomic operations
    // use int32 instead
    Tensor row_ind = at::zeros(col_ind.sizes(), col_ind.options().dtype(at::ScalarType::Int));
    row_ind.scatter_add_(0, row_ptr, at::ones(row_ptr.sizes(), row_ind.options()));
    row_ind = row_ind.toType(at::ScalarType::Long);
    row_ind = row_ind.cumsum(0) - 1;
    Tensor index = at::stack({row_ind, col_ind}, 0);
    return at::_sparse_coo_tensor_unsafe(index, value, size, value.options().layout(kSparse));
}

template <class scalar_t, class NaryOp, class BinaryOp>
void spmm_forward_out_cpu(const int64_t *row_ptr, const int64_t *col_ind, const scalar_t *value,
                          const scalar_t *input, scalar_t *output,
                          int64_t num_row, int64_t nnz, int64_t dim) {
    parallel_for(0, num_row, 0, [&](int64_t row_start, int64_t row_end) {
        for (int64_t row = row_start; row < row_end; row++) {
            for (int64_t d = 0; d < dim; d++)
                output[row * dim + d] = NaryOp::zero;

            int64_t ptr_start = row_ptr[row];
            int64_t ptr_end = row + 1 < num_row ? row_ptr[row + 1] : nnz;
            for (int64_t ptr = ptr_start; ptr < ptr_end; ptr++) {
                int64_t col = col_ind[ptr];
                scalar_t val = value[ptr];
                for (int64_t d = 0; d < dim; d++) {
                    scalar_t x = BinaryOp::forward(val, input[col * dim + d]);
                    scalar_t &out = output[row * dim + d];
                    out = NaryOp::forward(out, x);
                }
            }
        }
    });
}

template <class scalar_t, class NaryOp, class BinaryOp>
void spmm_backward_out_cpu(const int64_t *row_ptr, const int64_t *col_ind, const scalar_t *value,
                           const scalar_t *input, const scalar_t *output, const scalar_t *output_grad,
                           scalar_t *value_grad, scalar_t *input_grad,
                           int64_t num_row, int64_t nnz, int64_t dim,
                           std::vector<std::mutex> &mutex) {
    parallel_for(0, num_row, 0, [&](int64_t row_start, int64_t row_end) {
        for (int64_t row = row_start; row < row_end; row++) {
            int64_t ptr_start = row_ptr[row];
            int64_t ptr_end = row + 1 < num_row ? row_ptr[row + 1] : nnz;
            for (int64_t ptr = ptr_start; ptr < ptr_end; ptr++) {
                int64_t col = col_ind[ptr];
                scalar_t val = value[ptr];
                scalar_t val_grad = 0;
                for (int64_t d = 0; d < dim; d++) {
                    scalar_t in = input[col * dim + d];
                    scalar_t out = output[row * dim + d];
                    scalar_t out_grad = output_grad[row * dim + d];
                    scalar_t x = BinaryOp::forward(val, in);
                    scalar_t dx_dval = BinaryOp::backward_lhs(val, in);
                    scalar_t dx_din = BinaryOp::backward_rhs(val, in);
                    scalar_t dout_dx = NaryOp::backward(out, x);
                    val_grad += out_grad * dout_dx * dx_dval;
                    {
                        std::lock_guard<std::mutex> lock(mutex[col * dim + d]);
                        input_grad[col * dim + d] += out_grad * dout_dx * dx_din;
                    }
                }
                value_grad[ptr] = val_grad;
            }
        }
    });
}

template <template<class> class NaryOp, template<class> class BinaryOp>
Tensor spmm_forward_cpu(const SparseTensor &sparse, const Tensor &input_) {
    constexpr const char *fn_name = "spmm_forward_cpu";
    TensorArg sparse_arg(sparse, "sparse", 1), input_arg(input_, "input", 2);

    spmm_forward_check(fn_name, sparse_arg, input_arg);
    checkDeviceType(fn_name, {sparse, input_}, kCPU);

    const Tensor input = input_.contiguous();

    int64_t nnz = sparse._nnz();
    int64_t dim = input.size(1);
    int64_t num_row = sparse.size(0);
    Tensor output = at::empty({num_row, dim}, input.options());

    auto csr = coo2csr(sparse);
    Tensor row_ptr = std::get<0>(csr);
    Tensor col_ind = std::get<1>(csr);
    Tensor value = std::get<2>(csr);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "spmm_forward_cpu", [&] {
        spmm_forward_out_cpu<scalar_t, NaryOp<scalar_t>, BinaryOp<scalar_t>>(
            row_ptr.data_ptr<int64_t>(),
            col_ind.data_ptr<int64_t>(),
            value.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_row, nnz, dim
        );
    });

    return output;
}

template <template<class> class NaryOp, template<class> class BinaryOp>
std::tuple<SparseTensor, Tensor> spmm_backward_cpu(
        const SparseTensor &sparse, const Tensor &input_, const Tensor &output_, const Tensor &output_grad_) {
    constexpr const char *fn_name = "spmm_backward_cpu";
    TensorArg sparse_arg(sparse, "sparse", 1), input_arg(input_, "input", 2), output_arg(output_, "output", 3),
              output_grad_arg(output_grad_, "output_grad", 4);

    spmm_backward_check(fn_name, sparse_arg, input_arg, output_arg, output_grad_arg);
    checkDeviceType(fn_name, {sparse, input_, output_, output_grad_}, kCPU);

    const Tensor input = input_.contiguous();
    const Tensor output = output_.contiguous();
    const Tensor output_grad = output_grad_.contiguous();

    int64_t nnz = sparse._nnz();
    int64_t dim = input.size(1);
    int64_t num_row = sparse.size(0);
    Tensor value_grad = at::zeros_like(sparse.values());
    Tensor input_grad = at::zeros_like(input);
    SparseTensor sparse_grad = at::_sparse_coo_tensor_unsafe(sparse.indices(), value_grad, sparse.sizes());

    auto csr = coo2csr(sparse);
    Tensor row_ptr = std::get<0>(csr).contiguous();
    Tensor col_ind = std::get<1>(csr).contiguous();
    Tensor value = std::get<2>(csr).contiguous();
    std::vector<std::mutex> mutex(input.numel());

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "spmm_backward_cpu", [&] {
        spmm_backward_out_cpu<scalar_t, NaryOp<scalar_t>, BinaryOp<scalar_t>>(
            row_ptr.data_ptr<int64_t>(),
            col_ind.data_ptr<int64_t>(),
            value.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            output_grad.data_ptr<scalar_t>(),
            value_grad.data_ptr<scalar_t>(),
            input_grad.data_ptr<scalar_t>(),
            num_row, nnz, dim,
            mutex
        );
    });

    return std::make_tuple(sparse_grad, input_grad);
}

#define DECLARE_FORWARD_IMPL(ADD, MUL, NARYOP, BINARYOP) \
    Tensor spmm_##ADD##_##MUL##_forward_cpu(const SparseTensor &sparse, const Tensor &input) { \
        return spmm_forward_cpu<NARYOP, BINARYOP>(sparse, input);                              \
    }

#define DECLARE_BACKWARD_IMPL(ADD, MUL, NARYOP, BINARYOP) \
    std::tuple<SparseTensor, Tensor> spmm_##ADD##_##MUL##_backward_cpu(                                         \
            const SparseTensor &sparse, const Tensor &input, const Tensor &output, const Tensor &output_grad) { \
        return spmm_backward_cpu<NARYOP, BINARYOP>(sparse, input, output, output_grad);                         \
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spmm_add_mul_forward_cpu", &at::spmm_add_mul_forward_cpu);
    m.def("spmm_add_mul_backward_cpu", &at::spmm_add_mul_backward_cpu);
    m.def("spmm_min_mul_forward_cpu", &at::spmm_min_mul_forward_cpu);
    m.def("spmm_min_mul_backward_cpu", &at::spmm_min_mul_backward_cpu);
    m.def("spmm_max_mul_forward_cpu", &at::spmm_max_mul_forward_cpu);
    m.def("spmm_max_mul_backward_cpu", &at::spmm_max_mul_backward_cpu);
    m.def("spmm_add_add_forward_cpu", &at::spmm_add_add_forward_cpu);
    m.def("spmm_add_add_backward_cpu", &at::spmm_add_add_backward_cpu);
    m.def("spmm_min_add_forward_cpu", &at::spmm_min_add_forward_cpu);
    m.def("spmm_min_add_backward_cpu", &at::spmm_min_add_backward_cpu);
    m.def("spmm_max_add_forward_cpu", &at::spmm_max_add_forward_cpu);
    m.def("spmm_max_add_backward_cpu", &at::spmm_max_add_backward_cpu);
    m.def("rspmm_add_mul_forward_cpu", &at::rspmm_add_mul_forward_cpu);
    m.def("rspmm_add_mul_backward_cpu", &at::rspmm_add_mul_backward_cpu);
    m.def("rspmm_min_mul_forward_cpu", &at::rspmm_min_mul_forward_cpu);
    m.def("rspmm_min_mul_backward_cpu", &at::rspmm_min_mul_backward_cpu);
    m.def("rspmm_max_mul_forward_cpu", &at::rspmm_max_mul_forward_cpu);
    m.def("rspmm_max_mul_backward_cpu", &at::rspmm_max_mul_backward_cpu);
    m.def("rspmm_add_add_forward_cpu", &at::rspmm_add_add_forward_cpu);
    m.def("rspmm_add_add_backward_cpu", &at::rspmm_add_add_backward_cpu);
    m.def("rspmm_min_add_forward_cpu", &at::rspmm_min_add_forward_cpu);
    m.def("rspmm_min_add_backward_cpu", &at::rspmm_min_add_backward_cpu);
    m.def("rspmm_max_add_forward_cpu", &at::rspmm_max_add_forward_cpu);
    m.def("rspmm_max_add_backward_cpu", &at::rspmm_max_add_backward_cpu);
#ifdef CUDA_OP
    m.def("spmm_add_mul_forward_cuda", &at::spmm_add_mul_forward_cuda);
    m.def("spmm_add_mul_backward_cuda", &at::spmm_add_mul_backward_cuda);
    m.def("spmm_min_mul_forward_cuda", &at::spmm_min_mul_forward_cuda);
    m.def("spmm_min_mul_backward_cuda", &at::spmm_min_mul_backward_cuda);
    m.def("spmm_max_mul_forward_cuda", &at::spmm_max_mul_forward_cuda);
    m.def("spmm_max_mul_backward_cuda", &at::spmm_max_mul_backward_cuda);
    m.def("spmm_add_add_forward_cuda", &at::spmm_add_add_forward_cuda);
    m.def("spmm_add_add_backward_cuda", &at::spmm_add_add_backward_cuda);
    m.def("spmm_min_add_forward_cuda", &at::spmm_min_add_forward_cuda);
    m.def("spmm_min_add_backward_cuda", &at::spmm_min_add_backward_cuda);
    m.def("spmm_max_add_forward_cuda", &at::spmm_max_add_forward_cuda);
    m.def("spmm_max_add_backward_cuda", &at::spmm_max_add_backward_cuda);
    m.def("rspmm_add_mul_forward_cuda", &at::rspmm_add_mul_forward_cuda);
    m.def("rspmm_add_mul_backward_cuda", &at::rspmm_add_mul_backward_cuda);
    m.def("rspmm_min_mul_forward_cuda", &at::rspmm_min_mul_forward_cuda);
    m.def("rspmm_min_mul_backward_cuda", &at::rspmm_min_mul_backward_cuda);
    m.def("rspmm_max_mul_forward_cuda", &at::rspmm_max_mul_forward_cuda);
    m.def("rspmm_max_mul_backward_cuda", &at::rspmm_max_mul_backward_cuda);
    m.def("rspmm_add_add_forward_cuda", &at::rspmm_add_add_forward_cuda);
    m.def("rspmm_add_add_backward_cuda", &at::rspmm_add_add_backward_cuda);
    m.def("rspmm_min_add_forward_cuda", &at::rspmm_min_add_forward_cuda);
    m.def("rspmm_min_add_backward_cuda", &at::rspmm_min_add_backward_cuda);
    m.def("rspmm_max_add_forward_cuda", &at::rspmm_max_add_forward_cuda);
    m.def("rspmm_max_add_backward_cuda", &at::rspmm_max_add_backward_cuda);
#endif
}