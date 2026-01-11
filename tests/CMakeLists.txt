# 使用Google Test（如果可用）
# 尝试多种方式查找GTest
# 注意：先尝试手动查找静态库，避免找到动态库版本
set(GTest_FOUND FALSE)
unset(GTest_LIBRARIES CACHE)
unset(GTest_MAIN_LIBRARIES CACHE)
unset(GTest_INCLUDE_DIRS CACHE)

# 先尝试手动查找静态库（Homebrew安装的googletest）
file(GLOB GTEST_VERSIONS "/usr/local/Cellar/googletest/*")
if(GTEST_VERSIONS)
    list(SORT GTEST_VERSIONS)
    list(REVERSE GTEST_VERSIONS)
    list(GET GTEST_VERSIONS 0 GTEST_ROOT)
    
    find_path(GTest_INCLUDE_DIRS
        NAMES gtest/gtest.h
        PATHS "${GTEST_ROOT}"
        PATH_SUFFIXES include
        NO_DEFAULT_PATH
    )
    # 强制查找静态库（.a文件）
    find_library(GTest_LIBRARIES
        NAMES libgtest.a
        PATHS "${GTEST_ROOT}"
        PATH_SUFFIXES lib
        NO_DEFAULT_PATH
    )
    find_library(GTest_MAIN_LIBRARIES
        NAMES libgtest_main.a
        PATHS "${GTEST_ROOT}"
        PATH_SUFFIXES lib
        NO_DEFAULT_PATH
    )
    if(GTest_INCLUDE_DIRS AND GTest_LIBRARIES)
        set(GTest_FOUND TRUE)
        message(STATUS "Found GTest (static): ${GTEST_ROOT}")
        message(STATUS "  Include: ${GTest_INCLUDE_DIRS}")
        message(STATUS "  Libraries: ${GTest_LIBRARIES} ${GTest_MAIN_LIBRARIES}")
    endif()
endif()

# 如果手动查找失败，再尝试CMake的find_package
if(NOT GTest_FOUND)
    find_package(GTest QUIET)
endif()

if(NOT GTest_FOUND)
    # 尝试通过pkg-config查找
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(GTEST QUIET gtest)
        if(GTEST_FOUND)
            set(GTest_FOUND TRUE)
            set(GTest_LIBRARIES ${GTEST_LIBRARIES})
            set(GTest_INCLUDE_DIRS ${GTEST_INCLUDE_DIRS})
        endif()
    endif()
endif()


if(GTest_FOUND)
    enable_testing()
    
    # 设置GTest库变量（如果使用手动查找的方式）
    if(NOT TARGET GTest::GTest AND GTest_LIBRARIES)
        add_library(GTest::GTest STATIC IMPORTED)
        set_target_properties(GTest::GTest PROPERTIES
            IMPORTED_LOCATION "${GTest_LIBRARIES}"
            INTERFACE_INCLUDE_DIRECTORIES "${GTest_INCLUDE_DIRS}"
        )
    endif()
    if(NOT TARGET GTest::Main AND GTest_MAIN_LIBRARIES)
        add_library(GTest::Main STATIC IMPORTED)
        set_target_properties(GTest::Main PROPERTIES
            IMPORTED_LOCATION "${GTest_MAIN_LIBRARIES}"
            INTERFACE_LINK_LIBRARIES "GTest::GTest"
        )
    endif()
    
    # 辅助函数：链接测试目标
    # 对于算子库，使用-force_load确保所有符号（包括静态初始化）被包含
    function(link_test_target target_name)
        # 先链接核心库
        target_link_libraries(${target_name} PRIVATE 
            inferunity_core 
            inferunity_optimizers
            inferunity_runtime
            inferunity_backends)
        
        # 对于算子库，使用-force_load确保所有符号（包括静态初始化）被包含
        # macOS使用-force_load，Linux使用--whole-archive
        if(APPLE)
            # macOS: 使用-force_load
            target_link_libraries(${target_name} PRIVATE 
                -Wl,-force_load,$<TARGET_FILE:inferunity_operators>)
        else()
            # Linux: 使用--whole-archive
            target_link_libraries(${target_name} PRIVATE 
                -Wl,--whole-archive
                $<TARGET_FILE:inferunity_operators>
                -Wl,--no-whole-archive)
        endif()
        
        # 链接GTest库
        if(TARGET GTest::GTest)
            target_link_libraries(${target_name} PRIVATE 
                GTest::GTest GTest::Main)
        else()
            target_link_libraries(${target_name} PRIVATE 
                ${GTest_LIBRARIES} ${GTest_MAIN_LIBRARIES})
            target_include_directories(${target_name} PRIVATE ${GTest_INCLUDE_DIRS})
        endif()
    endfunction()
    
    # 核心测试
    add_executable(test_core
        test_tensor.cpp
        test_graph.cpp
        test_memory.cpp
    )
    link_test_target(test_core)
    add_test(NAME CoreTests COMMAND test_core)
    
    # 算子测试
    add_executable(test_operators
        test_operators.cpp
    )
    link_test_target(test_operators)
    add_test(NAME OperatorsTests COMMAND test_operators)
    
    # 集成测试
    add_executable(test_integration
        test_integration.cpp
    )
    link_test_target(test_integration)
    target_link_libraries(test_integration PRIVATE inferunity_frontend)
    add_test(NAME IntegrationTests COMMAND test_integration)
    
    # 运行时测试
    add_executable(test_runtime
        test_runtime.cpp
    )
    link_test_target(test_runtime)
    target_link_libraries(test_runtime PRIVATE inferunity_frontend)
    add_test(NAME RuntimeTests COMMAND test_runtime)
    
    # 融合算子测试
    add_executable(test_fused_operators
        test_fused_operators.cpp
    )
    link_test_target(test_fused_operators)
    add_test(NAME FusedOperatorsTests COMMAND test_fused_operators)
    
    # 算子融合Pass测试
    add_executable(test_operator_fusion
        test_operator_fusion.cpp
    )
    link_test_target(test_operator_fusion)
    add_test(NAME OperatorFusionTests COMMAND test_operator_fusion)
    
    # ONNX解析器测试
    add_executable(test_onnx_parser
        test_onnx_parser.cpp
    )
    link_test_target(test_onnx_parser)
    target_link_libraries(test_onnx_parser PRIVATE inferunity_frontend)
    add_test(NAME ONNXParserTests COMMAND test_onnx_parser)
    
    # 数学算子测试
    add_executable(test_math_operators
        test_math_operators.cpp
    )
    link_test_target(test_math_operators)
    add_test(NAME MathOperatorsTests COMMAND test_math_operators)
    
    # 激活函数测试
    add_executable(test_activation_operators
        test_activation_operators.cpp
    )
    link_test_target(test_activation_operators)
    add_test(NAME ActivationOperatorsTests COMMAND test_activation_operators)
    
    # 归一化算子测试
    add_executable(test_normalization_operators
        test_normalization_operators.cpp
    )
    link_test_target(test_normalization_operators)
    add_test(NAME NormalizationOperatorsTests COMMAND test_normalization_operators)
    
    # Conv调试测试
    add_executable(test_conv_debug
        test_conv_debug.cpp
    )
    link_test_target(test_conv_debug)
    add_test(NAME ConvDebugTests COMMAND test_conv_debug)
    
    # 形状操作算子测试
    add_executable(test_shape_operators
        test_shape_operators.cpp
    )
    link_test_target(test_shape_operators)
    add_test(NAME ShapeOperatorsTests COMMAND test_shape_operators)
    
    # 性能测试
    add_executable(test_performance
        test_performance.cpp
    )
    link_test_target(test_performance)
    add_test(NAME PerformanceTests COMMAND test_performance)
    
    # 错误处理测试
    add_executable(test_error_handling
        test_error_handling.cpp
    )
    link_test_target(test_error_handling)
    add_test(NAME ErrorHandlingTests COMMAND test_error_handling)
    
    # 内存优化测试
    add_executable(test_memory_optimization
        test_memory_optimization.cpp
    )
    link_test_target(test_memory_optimization)
    add_test(NAME MemoryOptimizationTests COMMAND test_memory_optimization)
    
else()
    message(WARNING "Google Test not found, tests will not be built")
endif()
