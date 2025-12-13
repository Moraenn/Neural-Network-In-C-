#include "ccptorch/tensor.hpp"
#include <format>
#include <random>

namespace ccptorch {

Tensor Tensor::clone() const {

}

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

Tensor full(const Shape& shape, Tensor::Scalar num, Device device = Device::cpu()){
    Tensor tensor{shape, device};
    auto* data_ptr = tensor.data();
    const std::size_t total_size = tensor.size();

    for (std::size_t i = 0; i < total_size; ++i) {
        data_ptr[i] = num;
    }

    return tensor;
}

Tensor rand(
    const Shape& shape, 
    Tensor::Scalar lbound = Tensor::Scalar{-1}, 
    Tensor::Scalar ubound=Tensor::Scalar{1}, 
    Device device = Device::cpu()
){
    Tensor tensor{shape, device};
    auto* data_ptr = tensor.data();
    const std::size_t total_size = tensor.size();

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist6(lbound, ubound);

    for (std::size_t i = 0; i < total_size; ++i) {
        data_ptr[i] = dist6(rng);
    }

    return tensor;
}

// Element-wise operations
Tensor add(const Tensor& a, const Tensor& b){
    if (a.device() != b.device()){
        throw std::logic_error(std::format("Tensor::add: tensors must be on the same device. Got {} and {}", a.device(), b.device())); 
    }
    if (a.shape() != b.shape()){
        throw std::logic_error(std::format("Tensor::add: tensors must be on of the same shape. Got {} and {}", a.shape(), b.shape())); 
    }
    Tensor tensor{a.shape(), a.device()};
    Shape shape = a.shape();
    Tensor::Scalar* data =  tensor.data();
    const Shape& strides = tensor.strides();
    for (std::size_t dim : shape){
        
    }
}
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

// Scalar operations
Tensor add(const Tensor& a, Tensor::Scalar scalar);
Tensor mul(const Tensor& a, Tensor::Scalar scalar);

// Matrix operations
Tensor matmul(const Tensor& a, const Tensor& b);

// Activation functions (forward only for now)
Tensor relu(const Tensor& x);

// Utility
Tensor reshape(const Tensor& x, const Shape& new_shape);

} // namespace ccptorch
