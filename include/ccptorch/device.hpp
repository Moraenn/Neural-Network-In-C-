#ifndef CCPTORCH_DEVICE_HPP
#define CCPTORCH_DEVICE_HPP

#include <string>
#include <string_view>

namespace ccptorch {

enum class DeviceType {
    CPU
    // TODO: добавить CUDA, ROCm, OpenCL и другие типы устройств
};

class Device {
public:
    constexpr Device() noexcept = default;

    constexpr Device(DeviceType type, int index = 0) noexcept
        : type_{type}
        , index_{index} {}

    static constexpr Device cpu(int index = 0) noexcept {
        return Device{DeviceType::CPU, index};
    }

    constexpr DeviceType type() const noexcept {
        return type_;
    }

    constexpr int index() const noexcept {
        return index_;
    }

    [[nodiscard]] std::string to_string() const {
        return std::string{type_to_string_view(type_)} + ":" + std::to_string(index_);
    }

    friend constexpr bool operator==(const Device& lhs, const Device& rhs) noexcept {
        return lhs.type_ == rhs.type_ && lhs.index_ == rhs.index_;
    }

    friend constexpr bool operator!=(const Device& lhs, const Device& rhs) noexcept {
        return !(lhs == rhs);
    }

private:
    DeviceType type_{DeviceType::CPU};
    int index_{0};

    static constexpr std::string_view type_to_string_view(DeviceType type) noexcept {
        switch (type) {
        case DeviceType::CPU:
            return "cpu";
        }
        return "unknown";
    }
};

} // namespace ccptorch

#endif // CCPTORCH_DEVICE_HPP
