# 后端库（执行提供者实现）

# CPU后端（基于内部算子实现，过渡方案）
# 注意：这个库在CMakeLists.txt中已经定义，这里只处理ONNX Runtime后端

# ONNX Runtime后端（可选）
if(ENABLE_ONNXRUNTIME)
    # 查找ONNX Runtime
    find_package(onnxruntime QUIET)
    
    if(onnxruntime_FOUND)
        message(STATUS "ONNX Runtime found, enabling ONNX Runtime backend")
        
        add_library(inferunity_backends_onnxruntime STATIC
            onnxruntime_backend.cpp
        )
        
        target_link_libraries(inferunity_backends_onnxruntime PUBLIC
            inferunity_core
            ${onnxruntime_LIBRARIES}
        )
        
        target_include_directories(inferunity_backends_onnxruntime PUBLIC
            ${onnxruntime_INCLUDE_DIRS}
        )
        
        # 将ONNX Runtime后端添加到主后端库
        target_link_libraries(inferunity_backends PUBLIC
            inferunity_backends_onnxruntime
        )
    else()
        message(WARNING "ONNX Runtime not found. Install it or set onnxruntime_DIR to enable ONNX Runtime backend.")
    endif()
endif()

