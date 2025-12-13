#ifndef CCPTORCH_TENSOR_HPP
#define CCPTORCH_TENSOR_HPP

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <format>

#include "device.hpp"

namespace ccptorch {

using Shape = std::vector<std::size_t>;

class TensorStorage {
public:
    using Scalar = double;

    TensorStorage() = default;

    explicit TensorStorage(std::size_t size)
        : data_(size, Scalar{}) {}

    std::size_t size() const noexcept {
        return data_.size();
    }

    Scalar* data() noexcept {
        return data_.data();
    }

    const Scalar* data() const noexcept {
        return data_.data();
    }

private:
    std::vector<Scalar> data_;
    // TODO: рассмотреть использование std::pmr::vector для более гибкого управления памятью
};

class Tensor {
public:
    using Scalar = double;

    Tensor() = default;

    explicit Tensor(Shape shape, Device device = Device::cpu())
        : storage_{std::make_shared<TensorStorage>(compute_numel(shape))}
        , shape_{std::move(shape)}
        , strides_(compute_default_strides(shape_))
        , device_{device} {}

    Tensor clone() const;

    // Element-wise operations
    Tensor add(const Tensor& a, const Tensor& b);
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

    const Shape& shape() const noexcept {
        return shape_;
    }

    std::size_t ndim() const noexcept {
        return shape_.size();
    }

    std::size_t size() const noexcept {
        return storage_ ? storage_->size() : 0;
    }

    Device device() const noexcept {
        return device_;
    }

    std::size_t numel() const noexcept {
        return size();
    }

    const Shape& strides() const noexcept {
        return strides_;
    }

    Scalar* data() noexcept {
        return storage_ ? storage_->data() : nullptr;
    }

    const Scalar* data() const noexcept {
        return storage_ ? storage_->data() : nullptr;
    }

    Scalar& operator()(std::initializer_list<std::size_t> indices) {
        const std::size_t flat_index = compute_flat_index(indices);
        return storage_->data()[flat_index];
    }

    const Scalar& operator()(std::initializer_list<std::size_t> indices) const {
        const std::size_t flat_index = compute_flat_index(indices);
        return storage_->data()[flat_index];
    }

    Scalar& operator[](std::size_t index) {
        if (shape_.size() != 1) {
            throw std::logic_error("Tensor::operator[]: tensor must be one dimensional");
        }
        if (index >= shape_[0]) {
            throw std::out_of_range("Tensor::operator[]: index is out of range");
        }

        return storage_->data()[index];
    }

    const Scalar& operator[](std::size_t index) const {
        if (shape_.size() != 1) {
            throw std::logic_error("Tensor::operator[]: tensor must be one dimensional");
        }
        if (index >= shape_[0]) {
            throw std::out_of_range("Tensor::operator[]: index is out of range");
        }

        return storage_->data()[index];
    }

    bool empty() const noexcept {
        return size() == 0;
    }

private:
    std::shared_ptr<TensorStorage> storage_;
    Shape shape_;
    Shape strides_;
    Device device_{Device::cpu()};

    static std::size_t compute_numel(const Shape& shape) {
        std::size_t result = 1;
        for (std::size_t dim : shape) {
            if (dim == 0) {
                return 0;
            }
            result *= dim;
        }
        return result;
    }
    
    friend std::ostream& operator<<(std::ostream& stream, const Tensor& tensor){
        stream << std::format("Tensor shape: [{}, {}] on device: {}", tensor.shape_[0], tensor.shape_[1], tensor.device_.to_string());
        return stream;
    }

    static Shape compute_default_strides(const Shape& shape) {
        Shape strides(shape.size(), 0);
        if (shape.empty()) {
            return strides;
        }

        std::size_t stride = 1;
        for (std::size_t i = shape.size(); i-- > 0;) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    std::size_t compute_flat_index(std::initializer_list<std::size_t> indices) const {
        if (indices.size() != shape_.size()) {
            throw std::out_of_range("Tensor: incorrect number of indices");
        }

        std::size_t flat_index = 0;
        std::size_t axis = 0;
        for (std::size_t index : indices) {
            if (index >= shape_[axis]) {
                throw std::out_of_range("Tensor: index out of bounds");
            }
            flat_index += index * strides_[axis];
            ++axis;
        }
        return flat_index;
    }
};

Tensor zeros(const Shape& shape, Device device = Device::cpu());
Tensor ones(const Shape& shape, Device device = Device::cpu());

Tensor full(const Shape& shape, Tensor::Scalar num, Device device = Device::cpu());
Tensor rand(const Shape& shape, Tensor::Scalar lbound = Tensor::Scalar{-1}, Tensor::Scalar ubound=Tensor::Scalar{1}, Device device = Device::cpu());

// TODO: добавить фабрики вроде full, arange, rand, randn

} // namespace ccptorch

#endif // CCPTORCH_TENSOR_HPP
