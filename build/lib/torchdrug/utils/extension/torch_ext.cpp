#include <torch/extension.h>

namespace at {

Tensor sparse_coo_tensor_unsafe(const Tensor &indices, const Tensor &values, IntArrayRef size) {
    return _sparse_coo_tensor_unsafe(indices, values, size, values.options().layout(kSparse));
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_coo_tensor_unsafe", &at::sparse_coo_tensor_unsafe,
          "Construct sparse COO tensor without index check");
}