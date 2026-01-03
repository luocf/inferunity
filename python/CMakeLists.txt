# Python绑定（使用pybind11）
# 参考ONNX Runtime的Python绑定实现

cmake_minimum_required(VERSION 3.15)

# 查找Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 使用FetchContent获取pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Python模块
pybind11_add_module(inferunity_py
    inferunity_bindings.cpp
)

# 链接库
target_link_libraries(inferunity_py PRIVATE
    inferunity
    pybind11::module
)

# 包含目录
target_include_directories(inferunity_py PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# 设置输出目录
set_target_properties(inferunity_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
)

# 安装Python模块
install(TARGETS inferunity_py
    LIBRARY DESTINATION python
)

