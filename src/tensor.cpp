#include "ccptorch/tensor.hpp"

namespace ccptorch {

Tensor zeros(const Shape& shape, Device device) {
    Tensor tensor{shape, device};
    auto* data_ptr = tensor.data();
    const std::size_t total_size = tensor.size();

    for (std::size_t i = 0; i < total_size; ++i) {
        data_ptr[i] = Tensor::Scalar{0.0};
    }

    return tensor;
}

Tensor ones(const Shape& shape, Device device) {
    Tensor tensor{shape, device};
    auto* data_ptr = tensor.data();
    const std::size_t total_size = tensor.size();

    for (std::size_t i = 0; i < total_size; ++i) {
        data_ptr[i] = Tensor::Scalar{1.0};
    }

    return tensor;
}

} // namespace ccptorch
