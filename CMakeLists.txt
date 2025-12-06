cmake_minimum_required(VERSION 3.20)

project(CcpTorch
    VERSION 1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(ccptorch
    src/tensor.cpp
)

target_include_directories(ccptorch
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(ccptorch
    PUBLIC
        cxx_std_23
)

# TODO: добавить более строгие warning-флаги с учетом платформы (MSVC/Clang/GCC)
# TODO: добавить опцию CCPTORCH_BUILD_TESTS и подключить тестовый фреймворк

add_executable(ccptorch_tensor_demo
    examples/simple_tensor_demo.cpp
)

target_link_libraries(ccptorch_tensor_demo
    PRIVATE
        ccptorch
)
