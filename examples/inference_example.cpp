// InferUnity 推理示例
// 演示如何加载ONNX模型并执行推理

#include "inferunity/engine.h"
#include "inferunity/tensor.h"
#include "inferunity/logger.h"
#include <iostream>
#include <vector>
#include <chrono>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> [input_data_file]" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    
    // 设置日志级别
    inferunity::Logger::Instance().SetLevel(inferunity::LogLevel::INFO);
    
    std::cout << "=== InferUnity 推理引擎示例 ===" << std::endl;
    std::cout << "模型路径: " << model_path << std::endl;
    
    // 1. 创建推理会话
    inferunity::SessionOptions options;
    options.execution_providers = {"CPUExecutionProvider"};
    options.graph_optimization_level = inferunity::SessionOptions::GraphOptimizationLevel::ALL;
    options.enable_profiling = true;  // 启用性能分析
    
    auto session = inferunity::InferenceSession::Create(options);
    if (!session) {
        std::cerr << "❌ 创建推理会话失败" << std::endl;
        return 1;
    }
    std::cout << "✅ 推理会话创建成功" << std::endl;
    
    // 2. 加载模型
    std::cout << "\n📥 加载模型..." << std::endl;
    auto status = session->LoadModel(model_path);
    if (!status.IsOk()) {
        std::cerr << "❌ 模型加载失败: " << status.Message() << std::endl;
        return 1;
    }
    std::cout << "✅ 模型加载成功" << std::endl;
    
    // 3. 获取模型信息
    std::cout << "\n📊 模型信息:" << std::endl;
    auto input_names = session->GetInputNames();
    auto output_names = session->GetOutputNames();
    auto input_shapes = session->GetInputShapes();
    auto output_shapes = session->GetOutputShapes();
    
    std::cout << "  输入数量: " << input_names.size() << std::endl;
    for (size_t i = 0; i < input_names.size(); ++i) {
        std::cout << "    输入[" << i << "]: " << input_names[i] << " 形状: (";
        for (size_t j = 0; j < input_shapes[i].dims.size(); ++j) {
            std::cout << input_shapes[i].dims[j];
            if (j < input_shapes[i].dims.size() - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
    }
    
    std::cout << "  输出数量: " << output_names.size() << std::endl;
    for (size_t i = 0; i < output_names.size(); ++i) {
        std::cout << "    输出[" << i << "]: " << output_names[i] << " 形状: (";
        for (size_t j = 0; j < output_shapes[i].dims.size(); ++j) {
            std::cout << output_shapes[i].dims[j];
            if (j < output_shapes[i].dims.size() - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
    }
    
    // 4. 准备输入
    std::cout << "\n🔧 准备输入数据..." << std::endl;
    std::vector<inferunity::Tensor*> inputs;
    std::vector<std::shared_ptr<inferunity::Tensor>> input_storage;
    
    for (size_t i = 0; i < input_names.size(); ++i) {
        auto input_tensor = session->CreateInputTensor(i);
        if (!input_tensor) {
            std::cerr << "❌ 创建输入张量失败: " << input_names[i] << std::endl;
            return 1;
        }
        
        // 填充输入数据（示例：填充1.0）
        // 实际使用时应该从文件或数据源加载真实数据
        float* input_data = static_cast<float*>(input_tensor->GetData());
        size_t element_count = input_tensor->GetElementCount();
        for (size_t j = 0; j < element_count; ++j) {
            input_data[j] = 1.0f;  // 示例数据
        }
        
        input_storage.push_back(input_tensor);
        inputs.push_back(input_tensor.get());
        std::cout << "  ✅ 输入[" << i << "] 准备完成，元素数量: " << element_count << std::endl;
    }
    
    // 5. 执行推理
    std::cout << "\n🚀 执行推理..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<inferunity::Tensor*> outputs;
    status = session->Run(inputs, outputs);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (!status.IsOk()) {
        std::cerr << "❌ 推理失败: " << status.Message() << std::endl;
        return 1;
    }
    
    std::cout << "✅ 推理成功，耗时: " << duration.count() << " ms" << std::endl;
    
    // 6. 获取输出
    std::cout << "\n📤 输出结果:" << std::endl;
    for (size_t i = 0; i < output_names.size(); ++i) {
        auto output_tensor = session->GetOutputTensor(i);
        if (output_tensor) {
            const auto& shape = output_tensor->GetShape();
            size_t element_count = output_tensor->GetElementCount();
            
            std::cout << "  输出[" << i << "] " << output_names[i] << ":" << std::endl;
            std::cout << "    形状: (";
            for (size_t j = 0; j < shape.dims.size(); ++j) {
                std::cout << shape.dims[j];
                if (j < shape.dims.size() - 1) std::cout << ", ";
            }
            std::cout << ")" << std::endl;
            std::cout << "    元素数量: " << element_count << std::endl;
            
            // 打印前几个输出值
            const float* output_data = static_cast<const float*>(output_tensor->GetData());
            size_t print_count = std::min(static_cast<size_t>(10), element_count);
            std::cout << "    前" << print_count << "个值: ";
            for (size_t j = 0; j < print_count; ++j) {
                std::cout << output_data[j] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    // 7. 性能分析
    if (options.enable_profiling) {
        std::cout << "\n⏱️  性能分析:" << std::endl;
        inferunity::ProfilingResult result;
        status = session->Profile(result);
        if (status.IsOk()) {
            std::cout << "  总执行时间: " << result.total_time_ms << " ms" << std::endl;
            std::cout << "  峰值内存: " << result.peak_memory_bytes / 1024.0 / 1024.0 << " MB" << std::endl;
            std::cout << "  节点性能 (前10个):" << std::endl;
            size_t print_count = std::min(static_cast<size_t>(10), result.node_profiles.size());
            for (size_t i = 0; i < print_count; ++i) {
                const auto& profile = result.node_profiles[i];
                std::cout << "    " << profile.node_name << " [" << profile.op_type << "]: "
                          << profile.execution_time_ms << " ms" << std::endl;
            }
        }
    }
    
    std::cout << "\n✅ 推理完成！" << std::endl;
    return 0;
}

