#pragma once

namespace at {

const unsigned kFullMask = 0xFFFFFFFF;

template <class scalar_t>
__device__ scalar_t warp_reduce(scalar_t value) {
#pragma unroll
    for (int delta = 1; delta < warpSize; delta *= 2)
#if __CUDACC_VER_MAJOR__ >= 9
        value += __shfl_down_sync(kFullMask, value, delta);
#else
        value += __shfl_down(value, delta);
#endif
    return value;
}

template<class scalar_t>
__device__ scalar_t warp_broadcast(scalar_t value, int lane_id) {
#if __CUDACC_VER_MAJOR__ >= 9
    return __shfl_sync(kFullMask, value, lane_id);
#else
    return __shfl(value, lane_id);
#endif
}

} // namespace at