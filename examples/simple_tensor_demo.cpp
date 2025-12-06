#include <ccptorch/tensor.hpp>

#include <cstddef>
#include <print>

int main() {
    using namespace ccptorch;

    Shape shape{2, 3};
    Tensor a = zeros(shape);
    Tensor b = ones(shape);

    for (std::size_t i = 0; i < shape[0]; ++i) {
        for (std::size_t j = 0; j < shape[1]; ++j) {
            a({i, j}) = static_cast<double>(i * 10 + j);
        }
    }

    std::println("Tensor a on device {} with shape [{} x {}]",
                 a.device().to_string(),
                 shape[0],
                 shape[1]);

    for (std::size_t i = 0; i < shape[0]; ++i) {
        for (std::size_t j = 0; j < shape[1]; ++j) {
            std::print("{:6.1f}", a({i, j}));
        }
        std::println("");
    }

    std::println("Tensor b (ones):");
    for (std::size_t i = 0; i < shape[0]; ++i) {
        for (std::size_t j = 0; j < shape[1]; ++j) {
            std::print("{:6.1f}", b({i, j}));
        }
        std::println("");
    }

    return 0;
}
