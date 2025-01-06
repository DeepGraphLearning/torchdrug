#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>

#include "util.cuh"
#include "embedding.h"

// Memory & time efficient implementation of embedding score functions
// Much of the code is adapted from GraphVite
// https://github.com/DeepGraphLearning/graphvite

namespace at {

template <class scalar_t>
__global__
void transe_forward_out_cuda(const scalar_t *entity, const scalar_t *relation, const int64_t *h_index,
                             const int64_t *t_index, const int64_t *r_index, scalar_t *score,
                             int64_t num_entity, int64_t num_relation, int64_t embedding_dim, int64_t num_sample) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = thread_id % warpSize;
    const int num_thread = gridDim.x * blockDim.x;

    for (int64_t sample_id = thread_id / warpSize; sample_id < num_sample; sample_id += num_thread / warpSize) {
        const scalar_t *h = entity + h_index[sample_id] * embedding_dim;
        const scalar_t *r = relation + r_index[sample_id] * embedding_dim;
        const scalar_t *t = entity + t_index[sample_id] * embedding_dim;
        scalar_t x = 0;
        for (int64_t i = lane_id; i < embedding_dim; i += warpSize)
            x += ::abs(h[i] + r[i] - t[i]);
        x = warp_broadcast(warp_reduce(x), 0);

        if (lane_id == 0)
            score[sample_id] = x;
    }
}

template <class scalar_t>
__global__
void transe_backward_out_cuda(const scalar_t *entity, const scalar_t *relation,
                              const int64_t *h_index, const int64_t *t_index, const int64_t *r_index,
                              const scalar_t *score_grad, scalar_t *entity_grad, scalar_t *relation_grad,
                              int64_t num_entity, int64_t num_relation, int64_t embedding_dim, int64_t num_sample) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = thread_id % warpSize;
    const int num_thread = gridDim.x * blockDim.x;

    for (int64_t sample_id = thread_id / warpSize; sample_id < num_sample; sample_id += num_thread / warpSize) {
        int64_t h_sample = h_index[sample_id];
        int64_t r_sample = r_index[sample_id];
        int64_t t_sample = t_index[sample_id];
        const scalar_t *h = entity + h_sample * embedding_dim;
        const scalar_t *r = relation + r_sample * embedding_dim;
        const scalar_t *t = entity + t_sample * embedding_dim;
        scalar_t *h_grad = entity_grad + h_sample * embedding_dim;
        scalar_t *r_grad = relation_grad + r_sample * embedding_dim;
        scalar_t *t_grad = entity_grad + t_sample * embedding_dim;
        scalar_t grad = score_grad[sample_id];

        for (int64_t i = lane_id; i < embedding_dim; i += warpSize) {
            scalar_t s = h[i] + r[i] - t[i] > 0 ? 1 : -1;
            atomicAdd(&h_grad[i], grad * s);
            atomicAdd(&r_grad[i], grad * s);
            atomicAdd(&t_grad[i], -grad * s);
        }
    }
}

template <class scalar_t>
__global__
void distmult_forward_out_cuda(const scalar_t *entity, const scalar_t *relation, const int64_t *h_index,
                               const int64_t *t_index, const int64_t *r_index, scalar_t *score,
                               int64_t num_entity, int64_t num_relation, int64_t embedding_dim, int64_t num_sample) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = thread_id % warpSize;
    const int num_thread = gridDim.x * blockDim.x;

    for (int64_t sample_id = thread_id / warpSize; sample_id < num_sample; sample_id += num_thread / warpSize) {
        const scalar_t *h = entity + h_index[sample_id] * embedding_dim;
        const scalar_t *r = relation + r_index[sample_id] * embedding_dim;
        const scalar_t *t = entity + t_index[sample_id] * embedding_dim;
        scalar_t x = 0;
        for (int64_t i = lane_id; i < embedding_dim; i += warpSize)
            x += h[i] * r[i] * t[i];
        x = warp_broadcast(warp_reduce(x), 0);

        if (lane_id == 0)
            score[sample_id] = x;
    }
}

template <class scalar_t>
__global__
void distmult_backward_out_cuda(const scalar_t *entity, const scalar_t *relation,
                                const int64_t *h_index, const int64_t *t_index, const int64_t *r_index,
                                const scalar_t *score_grad, scalar_t *entity_grad, scalar_t *relation_grad,
                                int64_t num_entity, int64_t num_relation, int64_t embedding_dim, int64_t num_sample) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = thread_id % warpSize;
    const int num_thread = gridDim.x * blockDim.x;

    for (int64_t sample_id = thread_id / warpSize; sample_id < num_sample; sample_id += num_thread / warpSize) {
        int64_t h_sample = h_index[sample_id];
        int64_t r_sample = r_index[sample_id];
        int64_t t_sample = t_index[sample_id];
        const scalar_t *h = entity + h_sample * embedding_dim;
        const scalar_t *r = relation + r_sample * embedding_dim;
        const scalar_t *t = entity + t_sample * embedding_dim;
        scalar_t *h_grad = entity_grad + h_sample * embedding_dim;
        scalar_t *r_grad = relation_grad + r_sample * embedding_dim;
        scalar_t *t_grad = entity_grad + t_sample * embedding_dim;
        scalar_t grad = score_grad[sample_id];

        for (int64_t i = lane_id; i < embedding_dim; i += warpSize) {
            scalar_t h_i = h[i], r_i = r[i], t_i = t[i];
            atomicAdd(&h_grad[i], grad * r_i * t_i);
            atomicAdd(&r_grad[i], grad * h_i * t_i);
            atomicAdd(&t_grad[i], grad * h_i * r_i);
        }
    }
}

template <class scalar_t>
__global__
void complex_forward_out_cuda(const scalar_t *entity, const scalar_t *relation, const int64_t *h_index,
                              const int64_t *t_index, const int64_t *r_index, scalar_t *score,
                              int64_t num_entity, int64_t num_relation, int64_t embedding_dim, int64_t num_sample) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = thread_id % warpSize;
    const int num_thread = gridDim.x * blockDim.x;

    for (int64_t sample_id = thread_id / warpSize; sample_id < num_sample; sample_id += num_thread / warpSize) {
        const scalar_t *h = entity + h_index[sample_id] * embedding_dim;
        const scalar_t *r = relation + r_index[sample_id] * embedding_dim;
        const scalar_t *t = entity + t_index[sample_id] * embedding_dim;
        scalar_t x = 0;
        for (int64_t i = lane_id; i < embedding_dim / 2; i += warpSize) {
            scalar_t h_re = h[i], h_im = h[i + embedding_dim / 2];
            scalar_t r_re = r[i], r_im = r[i + embedding_dim / 2];
            scalar_t t_re = t[i], t_im = t[i + embedding_dim / 2];
            scalar_t product_re = h_re * r_re - h_im * r_im;
            scalar_t product_im = h_re * r_im + h_im * r_re;
            x += product_re * t_re + product_im * t_im;
        }
        x = warp_broadcast(warp_reduce(x), 0);

        if (lane_id == 0)
            score[sample_id] = x;
    }
}

template <class scalar_t>
__global__
void complex_backward_out_cuda(const scalar_t *entity, const scalar_t *relation,
                               const int64_t *h_index, const int64_t *t_index, const int64_t *r_index,
                               const scalar_t *score_grad, scalar_t *entity_grad, scalar_t *relation_grad,
                               int64_t num_entity, int64_t num_relation, int64_t embedding_dim, int64_t num_sample) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = thread_id % warpSize;
    const int num_thread = gridDim.x * blockDim.x;

    for (int64_t sample_id = thread_id / warpSize; sample_id < num_sample; sample_id += num_thread / warpSize) {
        int64_t h_sample = h_index[sample_id];
        int64_t r_sample = r_index[sample_id];
        int64_t t_sample = t_index[sample_id];
        const scalar_t *h = entity + h_sample * embedding_dim;
        const scalar_t *r = relation + r_sample * embedding_dim;
        const scalar_t *t = entity + t_sample * embedding_dim;
        scalar_t *h_grad = entity_grad + h_sample * embedding_dim;
        scalar_t *r_grad = relation_grad + r_sample * embedding_dim;
        scalar_t *t_grad = entity_grad + t_sample * embedding_dim;
        scalar_t grad = score_grad[sample_id];

        for (int64_t i = lane_id; i < embedding_dim / 2; i += warpSize) {
            scalar_t h_re = h[i], h_im = h[i + embedding_dim / 2];
            scalar_t r_re = r[i], r_im = r[i + embedding_dim / 2];
            scalar_t t_re = t[i], t_im = t[i + embedding_dim / 2];
            atomicAdd(&h_grad[i], grad * (r_re * t_re + r_im * t_im));
            atomicAdd(&h_grad[i + embedding_dim / 2], grad * (-r_im * t_re + r_re * t_im));
            atomicAdd(&r_grad[i], grad * (h_re * t_re + h_im * t_im));
            atomicAdd(&r_grad[i + embedding_dim / 2], grad * (-h_im * t_re + h_re * t_im));
            atomicAdd(&t_grad[i], grad * (h_re * r_re - h_im * r_im));
            atomicAdd(&t_grad[i + embedding_dim / 2], grad * (h_re * r_im + h_im * r_re));
        }
    }
}

template <class scalar_t>
__global__
void rotate_forward_out_cuda(const scalar_t *entity, const scalar_t *relation, const int64_t *h_index,
                             const int64_t *t_index, const int64_t *r_index, scalar_t *score,
                             int64_t num_entity, int64_t num_relation, int64_t embedding_dim, int64_t num_sample) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = thread_id % warpSize;
    const int num_thread = gridDim.x * blockDim.x;

    for (int64_t sample_id = thread_id / warpSize; sample_id < num_sample; sample_id += num_thread / warpSize) {
        const scalar_t *h = entity + h_index[sample_id] * embedding_dim;
        const scalar_t *r = relation + r_index[sample_id] * embedding_dim / 2;
        const scalar_t *t = entity + t_index[sample_id] * embedding_dim;
        scalar_t x = 0;
        for (int64_t i = lane_id; i < embedding_dim / 2; i += warpSize) {
            scalar_t h_re = h[i], h_im = h[i + embedding_dim / 2];
            scalar_t r_re = ::cos(r[i]), r_im = ::sin(r[i]);
            scalar_t t_re = t[i], t_im = t[i + embedding_dim / 2];
            scalar_t distance_re = h_re * r_re - h_im * r_im - t_re;
            scalar_t distance_im = h_re * r_im + h_im * r_re - t_im;
            x += ::sqrt(distance_re * distance_re + distance_im * distance_im);
        }
        x = warp_broadcast(warp_reduce(x), 0);

        if (lane_id == 0)
            score[sample_id] = x;
    }
}

template <class scalar_t>
__global__
void rotate_backward_out_cuda(const scalar_t *entity, const scalar_t *relation,
                              const int64_t *h_index, const int64_t *t_index, const int64_t *r_index,
                              const scalar_t *score_grad, scalar_t *entity_grad, scalar_t *relation_grad,
                              int64_t num_entity, int64_t num_relation, int64_t embedding_dim, int64_t num_sample) {
    const float kEpsilon = 1e-15; // 1e-15 from GraphVite
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = thread_id % warpSize;
    const int num_thread = gridDim.x * blockDim.x;

    for (int64_t sample_id = thread_id / warpSize; sample_id < num_sample; sample_id += num_thread / warpSize) {
        int64_t h_sample = h_index[sample_id];
        int64_t r_sample = r_index[sample_id];
        int64_t t_sample = t_index[sample_id];
        const scalar_t *h = entity + h_sample * embedding_dim;
        const scalar_t *r = relation + r_sample * embedding_dim / 2;
        const scalar_t *t = entity + t_sample * embedding_dim;
        scalar_t *h_grad = entity_grad + h_sample * embedding_dim;
        scalar_t *r_grad = relation_grad + r_sample * embedding_dim / 2;
        scalar_t *t_grad = entity_grad + t_sample * embedding_dim;
        scalar_t grad = score_grad[sample_id];

        for (int64_t i = lane_id; i < embedding_dim / 2; i += warpSize) {
            scalar_t h_re = h[i], h_im = h[i + embedding_dim / 2];
            scalar_t r_re = ::cos(r[i]), r_im = ::sin(r[i]);
            scalar_t t_re = t[i], t_im = t[i + embedding_dim / 2];
            scalar_t distance_re = h_re * r_re - h_im * r_im - t_re;
            scalar_t distance_im = h_re * r_im + h_im * r_re - t_im;
            scalar_t g = grad / (::sqrt(distance_re * distance_re + distance_im * distance_im) + kEpsilon);
            atomicAdd(&h_grad[i], g * (distance_re * r_re + distance_im * r_im));
            atomicAdd(&h_grad[i + embedding_dim / 2], g * (-distance_re * r_im + distance_im * r_re));
            atomicAdd(&r_grad[i], g * (-distance_re * (h_re * r_im + h_im * r_re)
                                      + distance_im * (h_re * r_re - h_im * r_im)));
            atomicAdd(&t_grad[i], -g * distance_re);
            atomicAdd(&t_grad[i + embedding_dim / 2], -g * distance_im);
        }
    }
}

template <class scalar_t>
__global__
void simple_forward_out_cuda(const scalar_t *entity, const scalar_t *relation, const int64_t *h_index,
                             const int64_t *t_index, const int64_t *r_index, scalar_t *score,
                             int64_t num_entity, int64_t num_relation, int64_t embedding_dim, int64_t num_sample) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = thread_id % warpSize;
    const int num_thread = gridDim.x * blockDim.x;

    for (int64_t sample_id = thread_id / warpSize; sample_id < num_sample; sample_id += num_thread / warpSize) {
        const scalar_t *h = entity + h_index[sample_id] * embedding_dim;
        const scalar_t *r = relation + r_index[sample_id] * embedding_dim;
        const scalar_t *t = entity + t_index[sample_id] * embedding_dim;
        scalar_t x = 0;
        for (int64_t i = lane_id; i < embedding_dim; i += warpSize) {
            int64_t j = (i + embedding_dim / 2) % embedding_dim;
            x += h[i] * r[i] * t[j];
        }
        x = warp_broadcast(warp_reduce(x), 0);

        if (lane_id == 0)
            score[sample_id] = x;
    }
}

template <class scalar_t>
__global__
void simple_backward_out_cuda(const scalar_t *entity, const scalar_t *relation,
                              const int64_t *h_index, const int64_t *t_index, const int64_t *r_index,
                              const scalar_t *score_grad, scalar_t *entity_grad, scalar_t *relation_grad,
                              int64_t num_entity, int64_t num_relation, int64_t embedding_dim, int64_t num_sample) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = thread_id % warpSize;
    const int num_thread = gridDim.x * blockDim.x;

    for (int64_t sample_id = thread_id / warpSize; sample_id < num_sample; sample_id += num_thread / warpSize) {
        int64_t h_sample = h_index[sample_id];
        int64_t r_sample = r_index[sample_id];
        int64_t t_sample = t_index[sample_id];
        const scalar_t *h = entity + h_sample * embedding_dim;
        const scalar_t *r = relation + r_sample * embedding_dim;
        const scalar_t *t = entity + t_sample * embedding_dim;
        scalar_t *h_grad = entity_grad + h_sample * embedding_dim;
        scalar_t *r_grad = relation_grad + r_sample * embedding_dim;
        scalar_t *t_grad = entity_grad + t_sample * embedding_dim;
        scalar_t grad = score_grad[sample_id];

        for (int64_t i = lane_id; i < embedding_dim; i += warpSize) {
            int64_t j = (i + embedding_dim / 2) % embedding_dim;
            scalar_t h_i = h[i], r_i = r[i], t_j = t[j];
            atomicAdd(&h_grad[i], grad * r_i * t_j);
            atomicAdd(&r_grad[i], grad * h_i * t_j);
            atomicAdd(&t_grad[j], grad * h_i * r_i);
        }
    }
}

// If written in templates, the partial instantiation of template template parameters can't be resolved
// Therefore we opt for a macro implementation
#define DECLARE_FORWARD_IMPL(NAME)                                                                         \
    Tensor NAME##_forward_cuda(const Tensor &entity_, const Tensor &relation_, const Tensor &h_index_,     \
			const Tensor &t_index_, const Tensor &r_index_) {                                              \
		constexpr const char *fn_name = #NAME"_forward_cuda";                                              \
		TensorArg entity_arg(entity_, "entity", 1), relation_arg(relation_, "relation", 2),                \
				  h_index_arg(h_index_, "h_index", 3), r_index_arg(r_index_, "r_index", 4),                \
				  t_index_arg(t_index_, "t_index", 5);                                                     \
                                                                                                           \
		embedding_forward_check(fn_name, entity_arg, relation_arg, h_index_arg, r_index_arg, t_index_arg); \
		checkAllSameGPU(fn_name, {entity_arg, relation_arg, h_index_arg, r_index_arg, t_index_arg});       \
                                                                                                           \
		const Tensor entity = entity_.contiguous();                                                        \
		const Tensor relation = relation_.contiguous();                                                    \
		const Tensor h_index = h_index_.contiguous();                                                      \
		const Tensor r_index = r_index_.contiguous();                                                      \
		const Tensor t_index = t_index_.contiguous();                                                      \
                                                                                                           \
		int64_t num_entity = entity.size(0);                                                               \
		int64_t num_relation = relation.size(0);                                                           \
		int64_t embedding_dim = entity.size(-1);                                                           \
		int64_t num_sample = h_index.numel();                                                              \
                                                                                                           \
		Tensor score = at::empty(h_index.sizes(), entity.options());                                       \
                                                                                                           \
		cudaSetDevice(entity.get_device());                                                                \
		auto stream = at::cuda::getCurrentCUDAStream();                                                    \
                                                                                                           \
        AT_DISPATCH_FLOATING_TYPES(entity.scalar_type(), fn_name, [&] {                                    \
			NAME##_forward_out_cuda<scalar_t><<<4096, 512, 0, stream>>>(                                   \
				entity.data_ptr<scalar_t>(), relation.data_ptr<scalar_t>(),                                \
				h_index.data_ptr<int64_t>(), t_index.data_ptr<int64_t>(), r_index.data_ptr<int64_t>(),     \
				score.data_ptr<scalar_t>(),                                                                \
				num_entity, num_relation, embedding_dim, num_sample                                        \
			);                                                                                             \
		});                                                                                                \
                                                                                                           \
		return score;                                                                                      \
	}                                                                                                      \

#define DECLARE_BACKWARD_IMPL(NAME)                                                                                  \
    std::tuple<Tensor, Tensor> NAME##_backward_cuda(                                                                 \
            const Tensor &entity_, const Tensor &relation_, const Tensor &h_index_,                                  \
            const Tensor &t_index_, const Tensor &r_index_, const Tensor &score_grad_) {                             \
        constexpr const char *fn_name = #NAME"_backward_cuda";                                                       \
        TensorArg entity_arg(entity_, "entity", 1), relation_arg(relation_, "relation", 2),                          \
                  h_index_arg(h_index_, "h_index", 3), r_index_arg(r_index_, "r_index", 4),                          \
                  t_index_arg(t_index_, "t_index", 5), score_grad_arg(score_grad_, "score_grad", 6);                 \
                                                                                                                     \
        embedding_backward_check(fn_name, entity_arg, relation_arg, h_index_arg, r_index_arg, t_index_arg,           \
                                 score_grad_arg);                                                                    \
        checkAllSameGPU(fn_name, {entity_arg, relation_arg, h_index_arg, r_index_arg, t_index_arg, score_grad_arg}); \
                                                                                                                     \
        const Tensor entity = entity_.contiguous();                                                                  \
        const Tensor relation = relation_.contiguous();                                                              \
        const Tensor h_index = h_index_.contiguous();                                                                \
        const Tensor r_index = r_index_.contiguous();                                                                \
        const Tensor t_index = t_index_.contiguous();                                                                \
        const Tensor score_grad = score_grad_.contiguous();                                                          \
                                                                                                                     \
        int64_t num_entity = entity.size(0);                                                                         \
        int64_t num_relation = relation.size(0);                                                                     \
        int64_t embedding_dim = entity.size(-1);                                                                     \
        int64_t num_sample = h_index.numel();                                                                        \
                                                                                                                     \
        Tensor entity_grad = at::zeros_like(entity);                                                                 \
        Tensor relation_grad = at::zeros_like(relation);                                                             \
                                                                                                                     \
        cudaSetDevice(entity.get_device());                                                                          \
        auto stream = at::cuda::getCurrentCUDAStream();                                                              \
                                                                                                                     \
        AT_DISPATCH_FLOATING_TYPES(entity.scalar_type(), fn_name, [&] {                                              \
            NAME##_backward_out_cuda<scalar_t><<<4096, 512, 0, stream>>>(                                            \
                entity.data_ptr<scalar_t>(), relation.data_ptr<scalar_t>(),                                          \
                h_index.data_ptr<int64_t>(), t_index.data_ptr<int64_t>(), r_index.data_ptr<int64_t>(),               \
                score_grad.data_ptr<scalar_t>(),                                                                     \
                entity_grad.data_ptr<scalar_t>(), relation_grad.data_ptr<scalar_t>(),                                \
                num_entity, num_relation, embedding_dim, num_sample                                                  \
            );                                                                                                       \
        });                                                                                                          \
                                                                                                                     \
        return std::make_tuple(entity_grad, relation_grad);                                                          \
    }

DECLARE_FORWARD_IMPL(transe)
DECLARE_BACKWARD_IMPL(transe)

DECLARE_FORWARD_IMPL(distmult)
DECLARE_BACKWARD_IMPL(distmult)

DECLARE_FORWARD_IMPL(complex)
DECLARE_BACKWARD_IMPL(complex)

DECLARE_FORWARD_IMPL(rotate)
DECLARE_BACKWARD_IMPL(rotate)

DECLARE_FORWARD_IMPL(simple)
DECLARE_BACKWARD_IMPL(simple)

} // namespace at