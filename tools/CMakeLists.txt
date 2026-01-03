# 模型转换工具
add_executable(inferunity_convert
    convert.cpp
)

target_link_libraries(inferunity_convert PRIVATE inferunity)

# 性能分析工具
add_executable(inferunity_profiler
    profiler.cpp
)

target_link_libraries(inferunity_profiler PRIVATE inferunity)

# 性能基准测试工具
add_executable(inferunity_benchmark
    benchmark.cpp
)

target_link_libraries(inferunity_benchmark PRIVATE inferunity)
set_target_properties(inferunity_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 基准测试套件
add_executable(inferunity_benchmark_suite
    benchmark_suite.cpp
)

target_link_libraries(inferunity_benchmark_suite PRIVATE inferunity)
set_target_properties(inferunity_benchmark_suite PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

