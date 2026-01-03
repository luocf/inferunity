// InferUnity 基础使用示例
// 参考ONNX Runtime的API设计

#include "inferunity/engine.h"
#include "inferunity/tensor.h"
#include <iostream>
#include <vector>

int main() {
    // 1. 创建推理会话 (参考ONNX Runtime的InferenceSession)
    inferunity::SessionOptions options;
    options.execution_providers = {"CPUExecutionProvider"};  // 参考ONNX Runtime的命名
    options.graph_optimization_level = inferunity::SessionOptions::GraphOptimizationLevel::ALL;
    
    auto session = inferunity::InferenceSession::Create(options);
    if (!session) {
        std::cerr << "Failed to create inference session" << std::endl;
        return 1;
    }
    
    // 2. 加载模型 (参考ONNX Runtime的模型加载)
    inferunity::Status status = session->LoadModel("model.onnx");
    if (!status.IsOk()) {
        std::cerr << "Failed to load model: " << status.Message() << std::endl;
        return 1;
    }
    
    // 3. 获取模型信息 (参考ONNX Runtime的输入输出查询)
    auto input_shapes = session->GetInputShapes();
    auto output_shapes = session->GetOutputShapes();
    
    std::cout << "Input shapes: ";
    for (const auto& shape : input_shapes) {
        std::cout << "(";
        for (size_t i = 0; i < shape.dims.size(); ++i) {
            std::cout << shape.dims[i];
            if (i < shape.dims.size() - 1) std::cout << ", ";
        }
        std::cout << ") ";
    }
    std::cout << std::endl;
    
    // 4. 准备输入 (参考ONNX Runtime的IOBinding)
    auto input = session->CreateInputTensor(0);
    if (!input) {
        std::cerr << "Failed to create input tensor" << std::endl;
        return 1;
    }
    
    // 填充输入数据（示例）
    float* input_data = static_cast<float*>(input->GetData());
    size_t element_count = input->GetElementCount();
    for (size_t i = 0; i < element_count; ++i) {
        input_data[i] = 1.0f;  // 示例数据
    }
    
    // 5. 执行推理 (参考ONNX Runtime的Run方法)
    std::vector<inferunity::Tensor*> inputs = {input.get()};
    std::vector<inferunity::Tensor*> outputs;
    
    status = session->Run(inputs, outputs);
    if (!status.IsOk()) {
        std::cerr << "Inference failed: " << status.Message() << std::endl;
        return 1;
    }
    
    // 6. 获取输出
    auto output = session->GetOutputTensor(0);
    if (output) {
        const float* output_data = static_cast<const float*>(output->GetData());
        std::cout << "Output shape: ";
        const auto& shape = output->GetShape();
        for (size_t i = 0; i < shape.dims.size(); ++i) {
            std::cout << shape.dims[i];
            if (i < shape.dims.size() - 1) std::cout << "x";
        }
        std::cout << std::endl;
        
        // 打印前几个输出值
        size_t print_count = std::min(static_cast<size_t>(10), output->GetElementCount());
        std::cout << "First " << print_count << " output values: ";
        for (size_t i = 0; i < print_count; ++i) {
            std::cout << output_data[i] << " ";
        }
        std::cout << std::endl;
    }
    
    // 7. 性能分析（可选）(参考ONNX Runtime的性能分析)
    if (session->GetOptions().enable_profiling) {
        inferunity::ProfilingResult result;
        status = session->Profile(result);
        if (status.IsOk()) {
            std::cout << "Total execution time: " << result.total_time_ms << " ms" << std::endl;
            for (const auto& node_profile : result.node_profiles) {
                std::cout << "  " << node_profile.node_name << ": "
                          << node_profile.execution_time_ms << " ms" << std::endl;
            }
        }
    }
    
    return 0;
}

