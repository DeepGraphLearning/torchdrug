#include <ATen/Parallel.h>

#include "embedding.h"

namespace at {

// In PyTorch 1.4.0, parallel_for depends on some functions from at::internal in ATen/Parallel.h
// which are not explicitly included
// This is fixed in some new PyTorch release
using namespace at::internal;

void embedding_forward_check(CheckedFrom c, const TensorArg &entity_arg, const TensorArg &relation_arg,
                             const TensorArg &h_index_arg, const TensorArg &t_index_arg, const TensorArg &r_index_arg) {
    checkDim(c, entity_arg, 2);
    checkDim(c, relation_arg, 2);
    checkAllSameNumel(c, {h_index_arg, r_index_arg, t_index_arg});
    checkScalarType(c, h_index_arg, kLong);
    checkScalarType(c, t_index_arg, kLong);
    checkScalarType(c, r_index_arg, kLong);
}

void embedding_backward_check(CheckedFrom c, const TensorArg &entity_arg, const TensorArg &relation_arg,
                              const TensorArg &h_index_arg, const TensorArg &t_index_arg, const TensorArg &r_index_arg,
                              const TensorArg &score_grad_arg) {
    embedding_forward_check(c, entity_arg, relation_arg, h_index_arg, t_index_arg, r_index_arg);
    checkSameSize(c, h_index_arg, score_grad_arg);
}

template <class scalar_t>
void transe_forward_out_cpu(const scalar_t *entity, const scalar_t *relation, const int64_t *h_index,
                            const int64_t *t_index, const int64_t *r_index, scalar_t *score,
                            int64_t num_entity, int64_t num_relation, int64_t embedding_dim, int64_t num_sample) {
    parallel_for(0, num_sample, 0, [&](int64_t start, int64_t end) {
        for (int64_t sample_id = start; sample_id < end; sample_id++) {
            const scalar_t *h = entity + h_index[sample_id] * embedding_dim;
            const scalar_t *r = relation + r_index[sample_id] * embedding_dim;
            const scalar_t *t = entity + t_index[sample_id] * embedding_dim;
            scalar_t x = 0;
            for (int64_t i = 0; i < embedding_dim; i++)
                x += ::abs(h[i] + r[i] - t[i]);

            score[sample_id] = x;
        }
    });
}

template <class scalar_t>
void transe_backward_out_cpu(const scalar_t *entity, const scalar_t *relation,
                             const int64_t *h_index, const int64_t *t_index, const int64_t *r_index,
                             const scalar_t *score_grad, scalar_t *entity_grad, scalar_t *relation_grad,
                             int64_t num_entity, int64_t num_relation, int64_t embedding_dim, int64_t num_sample) {
    // since #CPU thread < embedding_dim
    // we can parallel over embedding_dim to avoid atomic operations
    parallel_for(0, embedding_dim, 0, [&](int64_t start, int64_t end) {
        for (int64_t sample_id = 0; sample_id < num_sample; sample_id++) {
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

            for (int64_t i = start; i < end; i++) {
                scalar_t s = h[i] + r[i] - t[i] > 0 ? 1 : -1;
                h_grad[i] += grad * s;
                r_grad[i] += grad * s;
                t_grad[i] += -grad * s;
            }
        }
    });
}

template <class scalar_t>
void distmult_forward_out_cpu(const scalar_t *entity, const scalar_t *relation, const int64_t *h_index,
                              const int64_t *t_index, const int64_t *r_index, scalar_t *score,
                              int64_t num_entity, int64_t num_relation, int64_t embedding_dim, int64_t num_sample) {
    parallel_for(0, num_sample, 0, [&](int64_t start, int64_t end) {
        for (int64_t sample_id = start; sample_id < end; sample_id++) {
            const scalar_t *h = entity + h_index[sample_id] * embedding_dim;
            const scalar_t *r = relation + r_index[sample_id] * embedding_dim;
            const scalar_t *t = entity + t_index[sample_id] * embedding_dim;
            scalar_t x = 0;
            for (int64_t i = 0; i < embedding_dim; i++)
                x += h[i] * r[i] * t[i];

            score[sample_id] = x;
        }
    });
}

template <class scalar_t>
void distmult_backward_out_cpu(const scalar_t *entity, const scalar_t *relation,
                               const int64_t *h_index, const int64_t *t_index, const int64_t *r_index,
                               const scalar_t *score_grad, scalar_t *entity_grad, scalar_t *relation_grad,
                               int64_t num_entity, int64_t num_relation, int64_t embedding_dim, int64_t num_sample) {
    // since #CPU thread < embedding_dim
    // we can parallel over embedding_dim to avoid atomic operations
    parallel_for(0, embedding_dim, 0, [&](int64_t start, int64_t end) {
        for (int64_t sample_id = 0; sample_id < num_sample; sample_id++) {
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

            for (int64_t i = start; i < end; i++) {
                scalar_t h_i = h[i], r_i = r[i], t_i = t[i];
                h_grad[i] += grad * r_i * t_i;
                r_grad[i] += grad * h_i * t_i;
                t_grad[i] += grad * h_i * r_i;
            }
        }
    });
}

template <class scalar_t>
void complex_forward_out_cpu(const scalar_t *entity, const scalar_t *relation, const int64_t *h_index,
                             const int64_t *t_index, const int64_t *r_index, scalar_t *score,
                             int64_t num_entity, int64_t num_relation, int64_t embedding_dim, int64_t num_sample) {
    parallel_for(0, num_sample, 0, [&](int64_t start, int64_t end) {
        for (int64_t sample_id = start; sample_id < end; sample_id++) {
            const scalar_t *h = entity + h_index[sample_id] * embedding_dim;
            const scalar_t *r = relation + r_index[sample_id] * embedding_dim;
            const scalar_t *t = entity + t_index[sample_id] * embedding_dim;
            scalar_t x = 0;
            for (int64_t i = 0; i < embedding_dim / 2; i++) {
                scalar_t h_re = h[i], h_im = h[i + embedding_dim / 2];
                scalar_t r_re = r[i], r_im = r[i + embedding_dim / 2];
                scalar_t t_re = t[i], t_im = t[i + embedding_dim / 2];
                scalar_t product_re = h_re * r_re - h_im * r_im;
                scalar_t product_im = h_re * r_im + h_im * r_re;
                x += product_re * t_re + product_im * t_im;
            }

            score[sample_id] = x;
        }
    });
}

template <class scalar_t>
void complex_backward_out_cpu(const scalar_t *entity, const scalar_t *relation,
                              const int64_t *h_index, const int64_t *t_index, const int64_t *r_index,
                              const scalar_t *score_grad, scalar_t *entity_grad, scalar_t *relation_grad,
                              int64_t num_entity, int64_t num_relation, int64_t embedding_dim, int64_t num_sample) {
    // since #CPU thread < embedding_dim
    // we can parallel over embedding_dim to avoid atomic operations
    parallel_for(0, embedding_dim / 2, 0, [&](int64_t start, int64_t end) {
        for (int64_t sample_id = 0; sample_id < num_sample; sample_id++) {
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

            for (int64_t i = start; i < end; i++) {
                scalar_t h_re = h[i], h_im = h[i + embedding_dim / 2];
                scalar_t r_re = r[i], r_im = r[i + embedding_dim / 2];
                scalar_t t_re = t[i], t_im = t[i + embedding_dim / 2];
                h_grad[i] = grad * (r_re * t_re + r_im * t_im);
                h_grad[i + embedding_dim / 2] = grad * (-r_im * t_re + r_re * t_im);
                r_grad[i] = grad * (h_re * t_re + h_im * t_im);
                r_grad[i + embedding_dim / 2] = grad * (-h_im * t_re + h_re * t_im);
                t_grad[i] = grad * (h_re * r_re - h_im * r_im);
                t_grad[i + embedding_dim / 2] = grad * (h_re * r_im + h_im * r_re);
            }
        }
    });
}

template <class scalar_t>
void rotate_forward_out_cpu(const scalar_t *entity, const scalar_t *relation, const int64_t *h_index,
                            const int64_t *t_index, const int64_t *r_index, scalar_t *score,
                            int64_t num_entity, int64_t num_relation, int64_t embedding_dim, int64_t num_sample) {
    parallel_for(0, num_sample, 0, [&](int64_t start, int64_t end) {
        for (int64_t sample_id = start; sample_id < end; sample_id++) {
            const scalar_t *h = entity + h_index[sample_id] * embedding_dim;
            const scalar_t *r = relation + r_index[sample_id] * embedding_dim / 2;
            const scalar_t *t = entity + t_index[sample_id] * embedding_dim;
            scalar_t x = 0;
            for (int64_t i = 0; i < embedding_dim / 2; i++) {
                scalar_t h_re = h[i], h_im = h[i + embedding_dim / 2];
                scalar_t r_re = ::cos(r[i]), r_im = ::sin(r[i]);
                scalar_t t_re = t[i], t_im = t[i + embedding_dim / 2];
                scalar_t distance_re = h_re * r_re - h_im * r_im - t_re;
                scalar_t distance_im = h_re * r_im + h_im * r_re - t_im;
                x += ::sqrt(distance_re * distance_re + distance_im * distance_im);
            }

            score[sample_id] = x;
        }
    });
}

template <class scalar_t>
void rotate_backward_out_cpu(const scalar_t *entity, const scalar_t *relation,
                             const int64_t *h_index, const int64_t *t_index, const int64_t *r_index,
                             const scalar_t *score_grad, scalar_t *entity_grad, scalar_t *relation_grad,
                             int64_t num_entity, int64_t num_relation, int64_t embedding_dim, int64_t num_sample) {
    const float kEpsilon = 1e-15; // 1e-15 from GraphVite
    // since #CPU thread < embedding_dim / 2
    // we can parallel over embedding_dim to avoid atomic operations
    parallel_for(0, embedding_dim / 2, 0, [&](int64_t start, int64_t end) {
        for (int64_t sample_id = 0; sample_id < num_sample; sample_id++) {
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

            for (int64_t i = start; i < end; i++) {
                scalar_t h_re = h[i], h_im = h[i + embedding_dim / 2];
                scalar_t r_re = ::cos(r[i]), r_im = ::sin(r[i]);
                scalar_t t_re = t[i], t_im = t[i + embedding_dim / 2];
                scalar_t distance_re = h_re * r_re - h_im * r_im - t_re;
                scalar_t distance_im = h_re * r_im + h_im * r_re - t_im;
                scalar_t g = grad / (::sqrt(distance_re * distance_re + distance_im * distance_im) + kEpsilon);
                h_grad[i] += g * (distance_re * r_re + distance_im * r_im);
                h_grad[i + embedding_dim / 2] += g * (-distance_re * r_im + distance_im * r_re);
                r_grad[i] += g * (-distance_re * (h_re * r_im + h_im * r_re)
                                    + distance_im * (h_re * r_re - h_im * r_im));
                t_grad[i] += -g * distance_re;
                t_grad[i + embedding_dim / 2] += -g * distance_im;
            }
        }
    });
}

template <class scalar_t>
void simple_forward_out_cpu(const scalar_t *entity, const scalar_t *relation, const int64_t *h_index,
                            const int64_t *t_index, const int64_t *r_index, scalar_t *score,
                            int64_t num_entity, int64_t num_relation, int64_t embedding_dim, int64_t num_sample) {
    parallel_for(0, num_sample, 0, [&](int64_t start, int64_t end) {
        for (int64_t sample_id = start; sample_id < end; sample_id++) {
            const scalar_t *h = entity + h_index[sample_id] * embedding_dim;
            const scalar_t *r = relation + r_index[sample_id] * embedding_dim;
            const scalar_t *t = entity + t_index[sample_id] * embedding_dim;
            scalar_t x = 0;
            for (int64_t i = 0; i < embedding_dim; i++) {
                int64_t j = (i + embedding_dim / 2) % embedding_dim;
                x += h[i] * r[i] * t[j];
            }
            score[sample_id] = x;
        }
    });
}

template <class scalar_t>
void simple_backward_out_cpu(const scalar_t *entity, const scalar_t *relation,
                             const int64_t *h_index, const int64_t *t_index, const int64_t *r_index,
                             const scalar_t *score_grad, scalar_t *entity_grad, scalar_t *relation_grad,
                             int64_t num_entity, int64_t num_relation, int64_t embedding_dim, int64_t num_sample) {
    // since #CPU thread < embedding_dim
    // we can parallel over embedding_dim to avoid atomic operations
    parallel_for(0, embedding_dim, 0, [&](int64_t start, int64_t end) {
        for (int64_t sample_id = 0; sample_id < num_sample; sample_id++) {
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

            for (int64_t i = start; i < end; i++) {
                int64_t j = (i + embedding_dim / 2) % embedding_dim;
                scalar_t h_i = h[i], r_i = r[i], t_j = t[j];
                h_grad[i] += grad * r_i * t_j;
                r_grad[i] += grad * h_i * t_j;
                t_grad[j] += grad * h_i * r_i;
            }
        }
    });
}

// If written in templates, the partial instantiation of template template parameters can't be resolved
// Therefore we opt for a macro implementation
#define DECLARE_FORWARD_IMPL(NAME)                                                                         \
	Tensor NAME##_forward_cpu(const Tensor &entity_, const Tensor &relation_,                              \
							  const Tensor &h_index_, const Tensor &t_index_, const Tensor &r_index_) {    \
		constexpr const char *fn_name = #NAME"_forward_cpu";                                               \
		TensorArg entity_arg(entity_, "entity", 1), relation_arg(relation_, "relation", 2),                \
				  h_index_arg(h_index_, "h_index", 3), r_index_arg(r_index_, "r_index", 4),                \
				  t_index_arg(t_index_, "t_index", 5);                                                     \
                                                                                                           \
		embedding_forward_check(fn_name, entity_arg, relation_arg, h_index_arg, r_index_arg, t_index_arg); \
		checkDeviceType(fn_name, {entity_, relation_, h_index_, r_index_, t_index_}, kCPU);                \
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
		AT_DISPATCH_FLOATING_TYPES(entity.scalar_type(), #NAME"_forward_cpu", [&] {                        \
			NAME##_forward_out_cpu<scalar_t>(                                                              \
				entity.data_ptr<scalar_t>(), relation.data_ptr<scalar_t>(),                                \
				h_index.data_ptr<int64_t>(), t_index.data_ptr<int64_t>(), r_index.data_ptr<int64_t>(),     \
				score.data_ptr<scalar_t>(),                                                                \
				num_entity, num_relation, embedding_dim, num_sample                                        \
			);                                                                                             \
		});                                                                                                \
                                                                                                           \
		return score;                                                                                      \
	}

#define DECLARE_BACKWARD_IMPL(NAME)                                                                                  \
	std::tuple<Tensor, Tensor> NAME##_backward_cpu(                                                                  \
			const Tensor &entity_, const Tensor &relation_, const Tensor &h_index_,                                  \
            const Tensor &t_index_, const Tensor &r_index_, const Tensor &score_grad_) {                             \
		constexpr const char *fn_name = #NAME"_backward_cpu";                                                                  \
		TensorArg entity_arg(entity_, "entity", 1), relation_arg(relation_, "relation", 2),                          \
				  h_index_arg(h_index_, "h_index", 3), r_index_arg(r_index_, "r_index", 4),                          \
				  t_index_arg(t_index_, "t_index", 5), score_grad_arg(score_grad_, "score_grad", 6);                 \
                                                                                                                     \
		embedding_backward_check(fn_name, entity_arg, relation_arg, h_index_arg, r_index_arg, t_index_arg,           \
		                         score_grad_arg);                                                                    \
		checkDeviceType(fn_name, {entity_, relation_, h_index_, r_index_, t_index_, score_grad_}, kCPU);             \
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
		int64_t embedding_dim = entity.size(1);                                                                      \
		int64_t num_sample = h_index.numel();                                                                        \
                                                                                                                     \
		Tensor entity_grad = at::zeros_like(entity);                                                                 \
		Tensor relation_grad = at::zeros_like(relation);                                                             \
                                                                                                                     \
		AT_DISPATCH_FLOATING_TYPES(entity.scalar_type(), #NAME"_backward_cpu", [&] {                                 \
			NAME##_backward_out_cpu<scalar_t>(                                                                       \
				entity.data_ptr<scalar_t>(), relation.data_ptr<scalar_t>(),                                          \
				h_index.data_ptr<int64_t>(), t_index.data_ptr<int64_t>(), r_index.data_ptr<int64_t>(),               \
				score_grad.data_ptr<scalar_t>(),																	 \
				entity_grad.data_ptr<scalar_t>(), relation_grad.data_ptr<scalar_t>(),      							 \
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transe_forward_cpu", &at::transe_forward_cpu);
    m.def("transe_backward_cpu", &at::transe_backward_cpu);
    m.def("distmult_forward_cpu", &at::distmult_forward_cpu);
    m.def("distmult_backward_cpu", &at::distmult_backward_cpu);
    m.def("complex_forward_cpu", &at::complex_forward_cpu);
    m.def("complex_backward_cpu", &at::complex_backward_cpu);
    m.def("rotate_forward_cpu", &at::rotate_forward_cpu);
    m.def("rotate_backward_cpu", &at::rotate_backward_cpu);
    m.def("simple_forward_cpu", &at::simple_forward_cpu);
    m.def("simple_backward_cpu", &at::simple_backward_cpu);
#ifdef CUDA_OP
    m.def("transe_forward_cuda", &at::transe_forward_cuda);
    m.def("transe_backward_cuda", &at::transe_backward_cuda);
    m.def("distmult_forward_cuda", &at::distmult_forward_cuda);
    m.def("distmult_backward_cuda", &at::distmult_backward_cuda);
    m.def("complex_forward_cuda", &at::complex_forward_cuda);
    m.def("complex_backward_cuda", &at::complex_backward_cuda);
    m.def("rotate_forward_cuda", &at::rotate_forward_cuda);
    m.def("rotate_backward_cuda", &at::rotate_backward_cuda);
    m.def("simple_forward_cuda", &at::simple_forward_cuda);
    m.def("simple_backward_cuda", &at::simple_backward_cuda);
#endif
}