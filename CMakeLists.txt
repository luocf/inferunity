cmake_minimum_required(VERSION 3.15)
project(InferUnity VERSION 1.0.0 LANGUAGES CXX)

# C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译选项和警告设置
if(MSVC)
    add_compile_options(/W4 /WX-)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0)
    else()
        add_compile_options(-O3 -DNDEBUG)
    endif()
endif()

# 设置调试符号
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
endif()

# 编译选项
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_TOOLS "Build tools" ON)
option(ENABLE_CUDA "Enable CUDA backend" OFF)
option(ENABLE_TENSORRT "Enable TensorRT backend" OFF)
option(ENABLE_VULKAN "Enable Vulkan backend" OFF)
option(ENABLE_METAL "Enable Metal backend (macOS)" OFF)
option(ENABLE_SNPE "Enable SNPE backend" OFF)
option(ENABLE_ARMNN "Enable ARM NN backend" OFF)
option(ENABLE_ONNXRUNTIME "Enable ONNX Runtime backend" OFF)
option(USE_SYSTEM_PROTOBUF "Use system protobuf" ON)

# 输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# Protobuf依赖 (用于ONNX解析)
# 参考ONNX Runtime的protobuf集成方式
if(USE_SYSTEM_PROTOBUF)
    find_package(Protobuf REQUIRED)
    set(PROTOBUF_LIBRARIES ${Protobuf_LIBRARIES})
    set(PROTOBUF_INCLUDE_DIRS ${Protobuf_INCLUDE_DIRS})
else()
    # 使用FetchContent下载protobuf (简化依赖管理)
    include(FetchContent)
    FetchContent_Declare(
        protobuf
        GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
        GIT_TAG v21.12
    )
    set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(protobuf)
    set(PROTOBUF_LIBRARIES protobuf::libprotobuf)
    set(PROTOBUF_INCLUDE_DIRS ${protobuf_SOURCE_DIR}/src)
endif()

# ONNX protobuf定义
# 注意：这里我们直接使用ONNX的proto文件，或者使用预生成的pb文件
# 为了简化，我们创建一个简化的ONNX proto定义
# 实际项目中应该使用ONNX官方的proto文件
set(ONNX_PROTO_DIR ${CMAKE_SOURCE_DIR}/third_party/onnx)
if(EXISTS ${ONNX_PROTO_DIR}/onnx/onnx.proto)
    # 使用ONNX官方proto文件
    set(ONNX_PROTO_FILE ${ONNX_PROTO_DIR}/onnx/onnx.proto)
    set(ONNX_PROTOBUF_OUT_DIR ${CMAKE_BINARY_DIR}/generated)
    file(MAKE_DIRECTORY ${ONNX_PROTOBUF_OUT_DIR})
    
    # 生成protobuf代码
    set(ONNX_PROTOBUF_SRCS ${ONNX_PROTOBUF_OUT_DIR}/onnx.pb.cc)
    set(ONNX_PROTOBUF_HDRS ${ONNX_PROTOBUF_OUT_DIR}/onnx.pb.h)
    
    # 获取 protoc 可执行文件路径
    if(USE_SYSTEM_PROTOBUF)
        set(PROTOC_EXECUTABLE ${Protobuf_PROTOC_EXECUTABLE})
    else()
        # 使用 FetchContent 时，使用 generator expression 获取 protoc 路径
        set(PROTOC_EXECUTABLE $<TARGET_FILE:protobuf::protoc>)
    endif()
    
    add_custom_command(
        OUTPUT ${ONNX_PROTOBUF_SRCS} ${ONNX_PROTOBUF_HDRS}
        COMMAND ${PROTOC_EXECUTABLE}
        ARGS --cpp_out=${ONNX_PROTOBUF_OUT_DIR}
             --proto_path=${ONNX_PROTO_DIR}
             ${ONNX_PROTO_FILE}
        DEPENDS ${ONNX_PROTO_FILE}
        COMMENT "Generating ONNX protobuf files"
    )
    
    include_directories(${ONNX_PROTOBUF_OUT_DIR})
    add_definitions(-DINFERUNITY_USE_ONNX_PROTOBUF)
else()
    # 如果没有ONNX proto文件，创建一个简化的定义
    # 实际使用时应该下载ONNX的proto文件
    message(WARNING "ONNX proto files not found. ONNX parsing will be limited.")
    message(STATUS "To enable full ONNX support, clone ONNX repo and set ONNX_PROTO_DIR")
endif()

# ============================================================================
# 核心库（必须先定义，因为其他库依赖它）
# ============================================================================
add_library(inferunity_core STATIC
    src/core/tensor.cpp
    src/core/memory.cpp
    src/core/memory_lifetime.cpp
    src/core/memory_pool.cpp
    src/core/tensor_lifetime_optimizer.cpp
    src/core/graph.cpp
    src/core/engine.cpp
    src/core/shape_inference.cpp
)

target_include_directories(inferunity_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

# ============================================================================
# 前端解析器库
# ============================================================================
add_library(inferunity_frontend STATIC
    src/frontend/onnx_parser.cpp
    src/frontend/onnx_parser_impl.cpp
)

target_include_directories(inferunity_frontend PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
    ${PROTOBUF_INCLUDE_DIRS}
)

target_link_libraries(inferunity_frontend PUBLIC 
    inferunity_core
    ${PROTOBUF_LIBRARIES}
)

if(EXISTS ${ONNX_PROTOBUF_SRCS})
    target_sources(inferunity_frontend PRIVATE ${ONNX_PROTOBUF_SRCS})
    target_include_directories(inferunity_frontend PRIVATE ${ONNX_PROTOBUF_OUT_DIR})
    target_compile_definitions(inferunity_frontend PRIVATE INFERUNITY_USE_ONNX_PROTOBUF)
endif()

# ============================================================================
# 算子实现库
# ============================================================================
add_library(inferunity_operators STATIC
    src/operators/conv.cpp
    src/operators/activation.cpp
    src/operators/math.cpp
    src/operators/pooling.cpp
    src/operators/normalization.cpp
    src/operators/softmax.cpp
    src/operators/fused_ops.cpp
    src/operators/simd_utils.cpp
    src/operators/shape.cpp
    src/operators/operator_init.cpp
)

target_include_directories(inferunity_operators PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(inferunity_operators PUBLIC inferunity_core)

# 启用SIMD优化
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(inferunity_operators PRIVATE
        -march=native  # 自动检测CPU特性
    )
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_compile_options(inferunity_operators PRIVATE
            -mavx
            -mavx2
        )
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
        target_compile_options(inferunity_operators PRIVATE
            -mfpu=neon
        )
    endif()
endif()

# ============================================================================
# 优化器库
# ============================================================================
add_library(inferunity_optimizers STATIC
    src/optimizers/optimizer.cpp
    src/optimizers/constant_folding.cpp
    src/optimizers/operator_fusion.cpp
)

target_include_directories(inferunity_optimizers PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(inferunity_optimizers PUBLIC inferunity_core)

# ============================================================================
# 运行时库
# ============================================================================
add_library(inferunity_runtime STATIC
    src/runtime/runtime.cpp
    src/runtime/parallel_executor.cpp
    src/runtime/thread_pool.cpp
)

target_include_directories(inferunity_runtime PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(inferunity_runtime PUBLIC inferunity_core)

# ============================================================================
# 后端库
# ============================================================================
add_library(inferunity_backends STATIC
    src/backends/cpu_backend.cpp
)

target_include_directories(inferunity_backends PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(inferunity_backends PUBLIC inferunity_core inferunity_runtime inferunity_operators)

# ONNX Runtime后端（可选）
if(ENABLE_ONNXRUNTIME)
    # 查找ONNX Runtime
    find_package(onnxruntime QUIET)
    
    if(onnxruntime_FOUND)
        message(STATUS "ONNX Runtime found, enabling ONNX Runtime backend")
        
        # 定义宏以启用ONNX Runtime后端代码
        target_compile_definitions(inferunity_backends PRIVATE INFERUNITY_USE_ONNXRUNTIME)
        
        # 将ONNX Runtime后端源文件添加到主后端库
        target_sources(inferunity_backends PRIVATE
            src/backends/onnxruntime_backend.cpp
        )
        
        target_link_libraries(inferunity_backends PUBLIC
            ${onnxruntime_LIBRARIES}
        )
        
        target_include_directories(inferunity_backends PUBLIC
            ${onnxruntime_INCLUDE_DIRS}
        )
    else()
        message(WARNING "ONNX Runtime not found. Install it or set onnxruntime_DIR to enable ONNX Runtime backend.")
    endif()
endif()

# ============================================================================
# 可选后端
# ============================================================================

# CUDA后端（可选）
if(ENABLE_CUDA)
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
    
    add_library(inferunity_cuda_backend STATIC
        src/backends/cuda_backend.cpp
    )
    
    target_include_directories(inferunity_cuda_backend PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    
    target_link_libraries(inferunity_cuda_backend PUBLIC
        inferunity_core
        inferunity_runtime
        ${CUDA_LIBRARIES}
    )
    
    target_compile_options(inferunity_cuda_backend PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75>
    )
endif()

# TensorRT后端（可选）
if(ENABLE_TENSORRT)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        PATHS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include)
    
    find_library(TENSORRT_LIBRARY nvinfer
        PATHS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
    
    if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
        add_library(inferunity_tensorrt_backend STATIC
            src/backends/tensorrt_backend.cpp
        )
        
        target_include_directories(inferunity_tensorrt_backend PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
            ${TENSORRT_INCLUDE_DIR}
        )
        
        target_link_libraries(inferunity_tensorrt_backend PUBLIC
            inferunity_core
            inferunity_runtime
            ${TENSORRT_LIBRARY}
        )
    endif()
endif()

# Vulkan后端（可选）
if(ENABLE_VULKAN)
    find_package(Vulkan REQUIRED)
    
    add_library(inferunity_vulkan_backend STATIC
        src/backends/vulkan_backend.cpp
    )
    
    target_include_directories(inferunity_vulkan_backend PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${Vulkan_INCLUDE_DIRS}
    )
    
    target_link_libraries(inferunity_vulkan_backend PUBLIC
        inferunity_core
        inferunity_runtime
        ${Vulkan_LIBRARIES}
    )
endif()

# Metal后端（macOS可选）
if(ENABLE_METAL AND APPLE)
    add_library(inferunity_metal_backend STATIC
        src/backends/metal_backend.cpp
    )
    
    target_include_directories(inferunity_metal_backend PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    
    target_link_libraries(inferunity_metal_backend PUBLIC
        inferunity_core
        inferunity_runtime
        "-framework Metal"
        "-framework MetalKit"
        "-framework MetalPerformanceShaders"
    )
endif()

# 主库（组合所有组件）
# 注意：这是一个接口库，只链接其他库，不包含源文件
add_library(inferunity INTERFACE)

target_link_libraries(inferunity INTERFACE
    inferunity_core
    inferunity_frontend
    inferunity_optimizers
    inferunity_runtime
    inferunity_backends
)

# 对于算子库，使用-force_load确保所有符号（包括静态初始化）被包含
# macOS使用-force_load，Linux使用--whole-archive
if(APPLE)
    # macOS: 使用-force_load
    target_link_libraries(inferunity INTERFACE
        -Wl,-force_load,$<TARGET_FILE:inferunity_operators>
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # Linux: 使用--whole-archive
    target_link_libraries(inferunity INTERFACE
        -Wl,--whole-archive
        $<TARGET_FILE:inferunity_operators>
        -Wl,--no-whole-archive
    )
else()
    target_link_libraries(inferunity INTERFACE inferunity_operators)
endif()

# 如果启用了可选后端，链接它们
if(ENABLE_CUDA)
    target_link_libraries(inferunity PUBLIC inferunity_cuda_backend)
endif()

if(ENABLE_TENSORRT AND TENSORRT_LIBRARY)
    target_link_libraries(inferunity PUBLIC inferunity_tensorrt_backend)
endif()

if(ENABLE_VULKAN)
    target_link_libraries(inferunity PUBLIC inferunity_vulkan_backend)
endif()

if(ENABLE_METAL AND APPLE)
    target_link_libraries(inferunity PUBLIC inferunity_metal_backend)
endif()

# 工具
if(BUILD_TOOLS)
    add_subdirectory(tools)
endif()

# ============================================================================
# Python绑定（可选，非核心功能）
# ============================================================================
option(BUILD_PYTHON "Build Python bindings (optional, not required for core engine)" OFF)
if(BUILD_PYTHON)
    add_subdirectory(python)
endif()

# ============================================================================
# 示例程序
# ============================================================================
option(BUILD_EXAMPLES "Build example programs" ON)
if(BUILD_EXAMPLES)
    add_executable(basic_usage
        examples/basic_usage.cpp
    )
    
    add_executable(simple_test
        examples/simple_test.cpp
    )
    target_link_libraries(basic_usage PRIVATE inferunity)
    set_target_properties(basic_usage PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    
    target_link_libraries(simple_test PRIVATE inferunity)
    set_target_properties(simple_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    
    add_executable(inference_example
        examples/inference_example.cpp
    )
    target_link_libraries(inference_example PRIVATE inferunity)
    set_target_properties(inference_example PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    
    add_executable(test_qwen
        examples/test_qwen.cpp
    )
    target_link_libraries(test_qwen PRIVATE inferunity)
    set_target_properties(test_qwen PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    
    add_executable(test_provider
        examples/test_provider.cpp
    )
    target_link_libraries(test_provider PRIVATE inferunity)
    set_target_properties(test_provider PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    
    add_executable(test_operator_registry
        examples/test_operator_registry.cpp
    )
    target_link_libraries(test_operator_registry PRIVATE inferunity)
    set_target_properties(test_operator_registry PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    
    add_executable(test_cpu_backend
        examples/test_cpu_backend.cpp
    )
    target_link_libraries(test_cpu_backend PRIVATE inferunity)
    set_target_properties(test_cpu_backend PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()

# ============================================================================
# 测试
# ============================================================================
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================================
# 安装
# ============================================================================
install(TARGETS inferunity
    inferunity_core
    inferunity_frontend
    inferunity_operators
    inferunity_optimizers
    inferunity_runtime
    inferunity_backends
    EXPORT InferUnityTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

install(EXPORT InferUnityTargets
    FILE InferUnityTargets.cmake
    NAMESPACE InferUnity::
    DESTINATION lib/cmake/InferUnity
)

# ============================================================================
# 交叉编译和平台特定配置
# ============================================================================

# 交叉编译支持
if(CMAKE_CROSSCOMPILING)
    message(STATUS "Cross-compiling for ${CMAKE_SYSTEM_PROCESSOR}")
    # 可以添加交叉编译特定配置
endif()

# 平台特定优化
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    # ARM平台优化
    if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(inferunity_core PRIVATE -march=native)
    endif()
    if(ENABLE_ARMNN)
        # ARM NN集成
        message(STATUS "ARM NN support enabled")
    endif()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    # x86优化
    if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(inferunity_core PRIVATE -march=native -mavx2)
    endif()
endif()

