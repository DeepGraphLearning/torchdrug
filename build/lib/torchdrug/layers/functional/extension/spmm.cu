#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>

#include "util.cuh"
#include "operator.cuh"
#include "spmm.h"

// Memory & time efficient implementation of generalized spmm
// Much of the code is inspired by GE-SpMM
// https://github.com/hgyhungry/ge-spmm

namespace at {

namespace {

const int kCoarseningFactor = 2;
const int kThreadPerBlock = 256;

} // namespace anonymous

template <class scalar_t, class NaryOp, class BinaryOp>
__global__
void spmm_forward_out_cuda(const int64_t *row_ptr, const int64_t *col_ind, const scalar_t *value,
                           const scalar_t *input, scalar_t *output,
                           int64_t num_row, int64_t nnz, int64_t dim) {
    // for best optimization, the following code is compiled with constant warpSize
    assert(blockDim.x == warpSize);

    extern __shared__ int64_t buffer[];
    int64_t *col_ind_buf = buffer;
    scalar_t *value_buf = reinterpret_cast<scalar_t *>(col_ind_buf + blockDim.y * warpSize);
    col_ind_buf += threadIdx.y * warpSize;
    value_buf += threadIdx.y * warpSize;

    int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= num_row)
        return;
    int64_t d_start = blockIdx.y * warpSize * kCoarseningFactor + threadIdx.x;
    int64_t ptr_start = row_ptr[row];
    int64_t ptr_end = row + 1 < num_row ? row_ptr[row + 1] : nnz;
    scalar_t out[kCoarseningFactor];
#pragma unroll
    for (int64_t i = 0; i < kCoarseningFactor; i++)
        out[i] = NaryOp::zero;

    for (int64_t block_ptr = ptr_start; block_ptr < ptr_end; block_ptr += warpSize) {
        int64_t ptr = block_ptr + threadIdx.x;
        if (ptr < ptr_end) {
            col_ind_buf[threadIdx.x] = col_ind[ptr];
            value_buf[threadIdx.x] = value[ptr];
        }
        __syncwarp();

        int64_t max_offset = warpSize < ptr_end - block_ptr ? warpSize : ptr_end - block_ptr;
        for (int64_t offset_ptr = 0; offset_ptr < max_offset; offset_ptr++) {
            int64_t col = col_ind_buf[offset_ptr];
            scalar_t val = value_buf[offset_ptr];
#pragma unroll
            for (int64_t i = 0; i < kCoarseningFactor; i++) {
                int64_t d = d_start + i * warpSize;
                if (d >= dim)
                    break;
                scalar_t x = BinaryOp::forward(val, input[col * dim + d]);
                out[i] = NaryOp::forward(out[i], x);
            }
        }
        __syncwarp();
    }

#pragma unroll
    for (int64_t i = 0; i < kCoarseningFactor; i++) {
        int64_t d = d_start + i * warpSize;
        if (d >= dim)
            break;
        output[row * dim + d] = out[i];
    }
}

// both sparse and input require gradients
template <class scalar_t, class NaryOp, class BinaryOp>
__global__
void spmm_backward_out_cuda(const int64_t *row_ptr, const int64_t *col_ind, const scalar_t *value,
                            const scalar_t *input, const scalar_t *output, const scalar_t *output_grad,
                            scalar_t *value_grad, scalar_t *input_grad,
                            int64_t num_row, int64_t nnz, int64_t dim) {
    // for best optimization, the following code is compiled with constant warpSize
    assert(blockDim.x == warpSize);

    extern __shared__ int64_t buffer[];
    int64_t *col_ind_buf = buffer;
    scalar_t *value_buf = reinterpret_cast<scalar_t *>(col_ind_buf + blockDim.y * warpSize);
    col_ind_buf += threadIdx.y * warpSize;
    value_buf += threadIdx.y * warpSize;

    int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= num_row)
        return;
    int64_t d_start = blockIdx.y * warpSize * kCoarseningFactor + threadIdx.x;
    int64_t ptr_start = row_ptr[row];
    int64_t ptr_end = row + 1 < num_row ? row_ptr[row + 1] : nnz;

    for (int64_t block_ptr = ptr_start; block_ptr < ptr_end; block_ptr += warpSize) {
        int64_t ptr = block_ptr + threadIdx.x;
        if (ptr < ptr_end) {
            col_ind_buf[threadIdx.x] = col_ind[ptr];
            value_buf[threadIdx.x] = value[ptr];
        }
        __syncwarp();

        int64_t max_offset = warpSize < ptr_end - block_ptr ? warpSize : ptr_end - block_ptr;
        for (int64_t offset_ptr = 0; offset_ptr < max_offset; offset_ptr++) {
            int64_t col = col_ind_buf[offset_ptr];
            scalar_t val = value_buf[offset_ptr];
            scalar_t val_grad = 0;
#pragma unroll
            for (int64_t i = 0; i < kCoarseningFactor; i++) {
                int64_t d = d_start + i * warpSize;
                if (d >= dim)
                    break;
                scalar_t in = input[col * dim + d];
                scalar_t out = output[row * dim + d];
                scalar_t out_grad = output_grad[row * dim + d];
                scalar_t x = BinaryOp::forward(val, in);
                scalar_t dx_dval = BinaryOp::backward_lhs(val, in);
                scalar_t dx_din = BinaryOp::backward_rhs(val, in);
                scalar_t dout_dx = NaryOp::backward(out, x);
                val_grad += out_grad * dout_dx * dx_dval;
                atomicAdd(&input_grad[col * dim + d], out_grad * dout_dx * dx_din);
            }
            val_grad = warp_reduce(val_grad);
            if (threadIdx.x == 0)
                atomicAdd(&value_grad[block_ptr + offset_ptr], val_grad);
        }
        __syncwarp();
    }
}

// only input requires gradients
template <class scalar_t, class NaryOp, class BinaryOp>
__global__
void spmm_backward_out_cuda(const int64_t *row_ptr, const int64_t *col_ind, const scalar_t *value,
                            const scalar_t *input, const scalar_t *output, const scalar_t *output_grad,
                            scalar_t *input_grad,
                            int64_t num_row, int64_t nnz, int64_t dim) {
    // for best optimization, the following code is compiled with constant warpSize
    assert(blockDim.x == warpSize);

    extern __shared__ int64_t buffer[];
    int64_t *col_ind_buf = buffer;
    scalar_t *value_buf = reinterpret_cast<scalar_t *>(col_ind_buf + blockDim.y * warpSize);
    col_ind_buf += threadIdx.y * warpSize;
    value_buf += threadIdx.y * warpSize;

    int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= num_row)
        return;
    int64_t d_start = blockIdx.y * warpSize * kCoarseningFactor + threadIdx.x;
    int64_t ptr_start = row_ptr[row];
    int64_t ptr_end = row + 1 < num_row ? row_ptr[row + 1] : nnz;

    for (int64_t block_ptr = ptr_start; block_ptr < ptr_end; block_ptr += warpSize) {
        int64_t ptr = block_ptr + threadIdx.x;
        if (ptr < ptr_end) {
            col_ind_buf[threadIdx.x] = col_ind[ptr];
            value_buf[threadIdx.x] = value[ptr];
        }
        __syncwarp();

        int64_t max_offset = warpSize < ptr_end - block_ptr ? warpSize : ptr_end - block_ptr;
        for (int64_t offset_ptr = 0; offset_ptr < max_offset; offset_ptr++) {
            int64_t col = col_ind_buf[offset_ptr];
            scalar_t val = value_buf[offset_ptr];
#pragma unroll
            for (int64_t i = 0; i < kCoarseningFactor; i++) {
                int64_t d = d_start + i * warpSize;
                if (d >= dim)
                    break;
                scalar_t in = input[col * dim + d];
                scalar_t out = output[row * dim + d];
                scalar_t out_grad = output_grad[row * dim + d];
                scalar_t x = BinaryOp::forward(val, in);
                scalar_t dx_din = BinaryOp::backward_rhs(val, in);
                scalar_t dout_dx = NaryOp::backward(out, x);
                atomicAdd(&input_grad[col * dim + d], out_grad * dout_dx * dx_din);
            }
        }
        __syncwarp();
    }
}

template <template<class> class NaryOp, template<class> class BinaryOp>
Tensor spmm_forward_cuda(const SparseTensor &sparse, const Tensor &input_) {
    constexpr const char *fn_name = "spmm_forward_cuda";
    TensorArg sparse_arg(sparse, "sparse", 1), input_arg(input_, "input", 2);

    spmm_forward_check(fn_name, sparse_arg, input_arg);
    checkAllSameGPU(fn_name, {sparse_arg, input_arg});

    const Tensor input = input_.contiguous();

    int64_t nnz = sparse._nnz();
    int64_t dim = input.size(1);
    int64_t num_row = sparse.size(0);
    Tensor output = at::empty({num_row, dim}, input.options());

    auto csr = coo2csr(sparse);
    Tensor row_ptr = std::get<0>(csr);
    Tensor col_ind = std::get<1>(csr);
    Tensor value = std::get<2>(csr);

    cudaSetDevice(input.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int dim_per_block = 32; // warpSize
    const int num_dim_block = (dim + dim_per_block * kCoarseningFactor - 1) / (dim_per_block * kCoarseningFactor);
    const int row_per_block = kThreadPerBlock / dim_per_block;
    const int num_row_block = (num_row + row_per_block - 1) / row_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "spmm_forward_cuda", [&] {
        const int memory_size = kThreadPerBlock * (sizeof(int64_t) + sizeof(scalar_t));
        spmm_forward_out_cuda<scalar_t, NaryOp<scalar_t>, BinaryOp<scalar_t>>
            <<<dim3(num_row_block, num_dim_block), dim3(dim_per_block, row_per_block), memory_size, stream>>>(
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
std::tuple<SparseTensor, Tensor> spmm_backward_cuda(
        const SparseTensor &sparse, const Tensor &input_, const Tensor &output_, const Tensor &output_grad_) {
    constexpr const char *fn_name = "spmm_backward_cuda";
    TensorArg sparse_arg(sparse, "sparse", 1), input_arg(input_, "input", 2), output_arg(output_, "output", 3),
              output_grad_arg(output_grad_, "output_grad", 4);

    spmm_backward_check(fn_name, sparse_arg, input_arg, output_arg, output_grad_arg);
    checkAllSameGPU(fn_name, {sparse_arg, input_arg, output_arg, output_grad_arg});

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

    cudaSetDevice(input.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int dim_per_block = 32; // warpSize
    const int num_dim_block = (dim + dim_per_block * kCoarseningFactor - 1) / (dim_per_block * kCoarseningFactor);
    const int row_per_block = kThreadPerBlock / dim_per_block;
    const int num_row_block = (num_row + row_per_block - 1) / row_per_block;

    if (sparse.requires_grad())
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "spmm_backward_cuda", [&] {
            const int memory_size = kThreadPerBlock * (sizeof(int64_t) + sizeof(scalar_t));
            spmm_backward_out_cuda<scalar_t, NaryOp<scalar_t>, BinaryOp<scalar_t>>
                <<<dim3(num_row_block, num_dim_block), dim3(dim_per_block, row_per_block), memory_size, stream>>>(
                row_ptr.data_ptr<int64_t>(),
                col_ind.data_ptr<int64_t>(),
                value.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                output_grad.data_ptr<scalar_t>(),
                value_grad.data_ptr<scalar_t>(),
                input_grad.data_ptr<scalar_t>(),
                num_row, nnz, dim
            );
        });
    else
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "spmm_backward_cuda", [&] {
            const int memory_size = kThreadPerBlock * (sizeof(int64_t) + sizeof(scalar_t));
            spmm_backward_out_cuda<scalar_t, NaryOp<scalar_t>, BinaryOp<scalar_t>>
                <<<dim3(num_row_block, num_dim_block), dim3(dim_per_block, row_per_block), memory_size, stream>>>(
                row_ptr.data_ptr<int64_t>(),
                col_ind.data_ptr<int64_t>(),
                value.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                output_grad.data_ptr<scalar_t>(),
                input_grad.data_ptr<scalar_t>(),
                num_row, nnz, dim
            );
        });

    return std::make_tuple(sparse_grad, input_grad);
}

#define DECLARE_FORWARD_IMPL(ADD, MUL, NARYOP, BINARYOP) \
    Tensor spmm_##ADD##_##MUL##_forward_cuda(const SparseTensor &sparse, const Tensor &input) { \
        return spmm_forward_cuda<NARYOP, BINARYOP>(sparse, input);                              \
    }

#define DECLARE_BACKWARD_IMPL(ADD, MUL, NARYOP, BINARYOP) \
    std::tuple<SparseTensor, Tensor> spmm_##ADD##_##MUL##_backward_cuda(                                        \
            const SparseTensor &sparse, const Tensor &input, const Tensor &output, const Tensor &output_grad) { \
        return spmm_backward_cuda<NARYOP, BINARYOP>(sparse, input, output, output_grad);                        \
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