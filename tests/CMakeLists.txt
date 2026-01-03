# 使用Google Test（如果可用）
find_package(GTest QUIET)

if(GTest_FOUND)
    enable_testing()
    
    # 核心测试
    add_executable(test_core
        test_tensor.cpp
        test_graph.cpp
        test_memory.cpp
    )
    
    target_link_libraries(test_core PRIVATE 
        inferunity_core 
        inferunity_operators
        inferunity_optimizers
        inferunity_runtime
        inferunity_backends
        GTest::GTest GTest::Main)
    add_test(NAME CoreTests COMMAND test_core)
    
    # 算子测试
    add_executable(test_operators
        test_operators.cpp
    )
    
    target_link_libraries(test_operators PRIVATE 
        inferunity_core 
        inferunity_operators
        inferunity_optimizers
        inferunity_runtime
        inferunity_backends
        GTest::GTest GTest::Main)
    add_test(NAME OperatorsTests COMMAND test_operators)
    
    # 集成测试
    add_executable(test_integration
        test_integration.cpp
    )
    
    target_link_libraries(test_integration PRIVATE 
        inferunity_core 
        inferunity_frontend
        inferunity_operators
        inferunity_optimizers
        inferunity_runtime
        inferunity_backends
        GTest::GTest GTest::Main)
    add_test(NAME IntegrationTests COMMAND test_integration)
    
    # 运行时测试
    add_executable(test_runtime
        test_runtime.cpp
    )
    
    target_link_libraries(test_runtime PRIVATE 
        inferunity_core 
        inferunity_frontend
        inferunity_operators
        inferunity_optimizers
        inferunity_runtime
        inferunity_backends
        GTest::GTest GTest::Main)
    add_test(NAME RuntimeTests COMMAND test_runtime)
    
    # 融合算子测试
    add_executable(test_fused_operators
        test_fused_operators.cpp
    )
    
    target_link_libraries(test_fused_operators PRIVATE 
        inferunity_core 
        inferunity_operators
        inferunity_optimizers
        inferunity_runtime
        inferunity_backends
        GTest::GTest GTest::Main)
    add_test(NAME FusedOperatorsTests COMMAND test_fused_operators)
    
    # 算子融合Pass测试
    add_executable(test_operator_fusion
        test_operator_fusion.cpp
    )
    
    target_link_libraries(test_operator_fusion PRIVATE 
        inferunity_core 
        inferunity_operators
        inferunity_optimizers
        inferunity_runtime
        inferunity_backends
        GTest::GTest GTest::Main)
    add_test(NAME OperatorFusionTests COMMAND test_operator_fusion)
    
    # ONNX解析器测试
    add_executable(test_onnx_parser
        test_onnx_parser.cpp
    )
    
    target_link_libraries(test_onnx_parser PRIVATE 
        inferunity_core 
        inferunity_frontend
        inferunity_operators
        inferunity_optimizers
        inferunity_runtime
        inferunity_backends
        GTest::GTest GTest::Main)
    add_test(NAME ONNXParserTests COMMAND test_onnx_parser)
    
else()
    message(WARNING "Google Test not found, tests will not be built")
endif()

