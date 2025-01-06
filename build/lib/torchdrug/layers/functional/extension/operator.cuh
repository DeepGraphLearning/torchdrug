#pragma once

#include <limits>

#ifdef __CUDA_ARCH__
    #define HOST_DEVICE __host__ __device__
#else
    #define HOST_DEVICE
#endif

namespace at {

template <class scalar_t>
struct BinaryAdd {
    HOST_DEVICE static scalar_t forward(scalar_t x, scalar_t y) {
        return x + y;
    }

    HOST_DEVICE static scalar_t backward_lhs(scalar_t x, scalar_t y) {
        return 1;
    }

    HOST_DEVICE static scalar_t backward_rhs(scalar_t x, scalar_t y) {
        return 1;
    }
};

template <class scalar_t>
struct BinaryMul {
    HOST_DEVICE static scalar_t forward(scalar_t x, scalar_t y) {
        return x * y;
    }

    HOST_DEVICE static scalar_t backward_lhs(scalar_t x, scalar_t y) {
        return y;
    }

    HOST_DEVICE static scalar_t backward_rhs(scalar_t x, scalar_t y) {
        return x;
    }
};

template <class scalar_t>
struct NaryAdd {
    HOST_DEVICE static scalar_t forward(scalar_t result, scalar_t x) {
        return result + x;
    }

    HOST_DEVICE static scalar_t backward(scalar_t result, scalar_t x) {
        return 1;
    }

    static constexpr scalar_t zero = 0;
};

template <class scalar_t>
struct NaryMin {
    HOST_DEVICE static scalar_t forward(scalar_t result, scalar_t x) {
        return result < x ? result : x;
    }

    HOST_DEVICE static scalar_t backward(scalar_t result, scalar_t x) {
        return result == x ? 1 : 0;
    }

    static constexpr scalar_t zero = std::numeric_limits<scalar_t>::max();
};

template <class scalar_t>
struct NaryMax {
    HOST_DEVICE static scalar_t forward(scalar_t result, scalar_t x) {
        return result > x ? result : x;
    }

    HOST_DEVICE static scalar_t backward(scalar_t result, scalar_t x) {
        return result == x ? 1 : 0;
    }

    static constexpr scalar_t zero = std::numeric_limits<scalar_t>::lowest();
};

} // namespace at