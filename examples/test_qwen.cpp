// Qwen2.5-0.5B 模型测试程序
// 用于测试InferUnity推理引擎对Qwen2.5模型的支持

#include "inferunity/engine.h"
#include "inferunity/logger.h"
#include "inferunity/graph.h"
#include <iostream>
#include <vector>
#include <unordered_set>
#include <chrono>
#include <iomanip>
#include <algorithm>

void PrintModelInfo(inferunity::InferenceSession* session) {
    std::cout << "\n=== 模型信息 ===" << std::endl;
    
    auto input_names = session->GetInputNames();
    auto output_names = session->GetOutputNames();
    auto input_shapes = session->GetInputShapes();
    auto output_shapes = session->GetOutputShapes();
    
    std::cout << "输入 (" << input_names.size() << "):" << std::endl;
    for (size_t i = 0; i < input_names.size(); ++i) {
        std::cout << "  [" << i << "] " << input_names[i] << " : (";
        for (size_t j = 0; j < input_shapes[i].dims.size(); ++j) {
            std::cout << input_shapes[i].dims[j];
            if (j < input_shapes[i].dims.size() - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
    }
    
    std::cout << "输出 (" << output_names.size() << "):" << std::endl;
    for (size_t i = 0; i < output_names.size(); ++i) {
        std::cout << "  [" << i << "] " << output_names[i] << " : (";
        for (size_t j = 0; j < output_shapes[i].dims.size(); ++j) {
            std::cout << output_shapes[i].dims[j];
            if (j < output_shapes[i].dims.size() - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
    }
}

void CheckOperators(inferunity::InferenceSession* session) {
    std::cout << "\n=== 检查算子支持 ===" << std::endl;
    
    const auto* graph = session->GetGraph();
    if (!graph) {
        std::cerr << "  无法获取图信息" << std::endl;
        return;
    }
    
    std::unordered_set<std::string> found_operators;
    
    // 收集模型使用的算子
    for (const auto& node : graph->GetNodes()) {
        found_operators.insert(node->GetOpType());
    }
    
    std::cout << "模型使用的算子 (" << found_operators.size() << "个):" << std::endl;
    for (const auto& op : found_operators) {
        std::cout << "  ✅ " << op << std::endl;
    }
    
    // 检查Qwen2.5-0.5B特性支持情况
    std::cout << "\nQwen2.5-0.5B 特性检查:" << std::endl;
    
    // 检查RoPE
    bool has_rope = false;
    for (const auto& op : found_operators) {
        if (op.find("RoPE") != std::string::npos || 
            op.find("Rotary") != std::string::npos ||
            op.find("rope") != std::string::npos) {
            has_rope = true;
            break;
        }
    }
    std::cout << "  RoPE (旋转位置编码): " 
              << (has_rope ? "✅ 支持" : "⚠️  可能由多个算子组合实现") << std::endl;
    
    // 检查SwiGLU
    bool has_swiglu = found_operators.find("SwiGLU") != found_operators.end() ||
                      (found_operators.find("SiLU") != found_operators.end() && 
                       found_operators.find("Mul") != found_operators.end());
    std::cout << "  SwiGLU (激活函数): " 
              << (has_swiglu ? "✅ 支持" : "⚠️  可能由SiLU+Mul组合实现") << std::endl;
    
    // 检查RMSNorm
    std::cout << "  RMSNorm: " 
              << (found_operators.find("RMSNorm") != found_operators.end() ? "✅ 支持" : "❌ 缺失") << std::endl;
    
    // 检查Embedding
    std::cout << "  Embedding: " 
              << (found_operators.find("Embedding") != found_operators.end() ? "✅ 支持" : "⚠️  可能由Gather实现") << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <qwen2.5-0.5b-model.onnx>" << std::endl;
        std::cerr << "示例: " << argv[0] << " qwen2.5-0.5b-instruct.onnx" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    
    // 设置日志
    inferunity::Logger::Instance().SetLevel(inferunity::LogLevel::INFO);
    
    std::cout << "========================================" << std::endl;
    std::cout << "  Qwen2.5-0.5B 模型测试" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "模型路径: " << model_path << std::endl;
    std::cout << "模型特性:" << std::endl;
    std::cout << "  - 参数量: 0.5B" << std::endl;
    std::cout << "  - 架构: Transformer (RoPE, SwiGLU, RMSNorm)" << std::endl;
    std::cout << "  - 层数: 24" << std::endl;
    std::cout << "  - 上下文长度: 32,768 tokens" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 1. 创建会话
    std::cout << "\n[1/5] 创建推理会话..." << std::endl;
    inferunity::SessionOptions options;
    options.execution_providers = {"CPUExecutionProvider"};
    options.graph_optimization_level = inferunity::SessionOptions::GraphOptimizationLevel::ALL;
    options.enable_profiling = true;
    options.num_threads = 0;  // 自动检测
    
    auto session = inferunity::InferenceSession::Create(options);
    if (!session) {
        std::cerr << "❌ 创建会话失败" << std::endl;
        return 1;
    }
    std::cout << "✅ 会话创建成功" << std::endl;
    
    // 2. 加载模型
    std::cout << "\n[2/5] 加载模型..." << std::endl;
    auto start_load = std::chrono::high_resolution_clock::now();
    auto status = session->LoadModel(model_path);
    auto end_load = std::chrono::high_resolution_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_load - start_load);
    
    if (!status.IsOk()) {
        std::cerr << "❌ 模型加载失败: " << status.Message() << std::endl;
        return 1;
    }
    std::cout << "✅ 模型加载成功 (耗时: " << load_time.count() << " ms)" << std::endl;
    
    // 3. 打印模型信息
    PrintModelInfo(session.get());
    
    // 4. 检查算子支持
    CheckOperators(session.get());
    
    // 5. 准备输入（示例：使用小batch size）
    std::cout << "\n[3/5] 准备输入数据..." << std::endl;
    auto input_names = session->GetInputNames();
    auto input_shapes = session->GetInputShapes();
    
    std::vector<inferunity::Tensor*> inputs;
    std::vector<std::shared_ptr<inferunity::Tensor>> input_storage;
    
    for (size_t i = 0; i < input_names.size(); ++i) {
        auto input_tensor = session->CreateInputTensor(i);
        if (!input_tensor) {
            std::cerr << "❌ 创建输入张量失败: " << input_names[i] << std::endl;
            return 1;
        }
        
        // 对于Qwen模型，输入通常是token IDs (INT64)
        // 这里简化处理，实际应该从tokenizer获取
        if (input_tensor->GetDataType() == inferunity::DataType::INT64) {
            int64_t* data = static_cast<int64_t*>(input_tensor->GetData());
            size_t count = input_tensor->GetElementCount();
            // 示例：填充一些token IDs（实际应该从tokenizer获取）
            for (size_t j = 0; j < count; ++j) {
                data[j] = 1;  // 示例token ID
            }
        } else {
            // FLOAT类型输入
            float* data = static_cast<float*>(input_tensor->GetData());
            size_t count = input_tensor->GetElementCount();
            for (size_t j = 0; j < count; ++j) {
                data[j] = 1.0f;
            }
        }
        
        input_storage.push_back(input_tensor);
        inputs.push_back(input_tensor.get());
        std::cout << "  ✅ " << input_names[i] << " 形状: (";
        for (size_t j = 0; j < input_shapes[i].dims.size(); ++j) {
            std::cout << input_shapes[i].dims[j];
            if (j < input_shapes[i].dims.size() - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
    }
    
    // 6. 执行推理
    std::cout << "\n[4/5] 执行推理..." << std::endl;
    auto start_infer = std::chrono::high_resolution_clock::now();
    
    std::vector<inferunity::Tensor*> outputs;
    status = session->Run(inputs, outputs);
    
    auto end_infer = std::chrono::high_resolution_clock::now();
    auto infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_infer - start_infer);
    
    if (!status.IsOk()) {
        std::cerr << "❌ 推理失败: " << status.Message() << std::endl;
        std::cerr << "\n可能的原因:" << std::endl;
        std::cerr << "  1. 模型使用了不支持的算子" << std::endl;
        std::cerr << "  2. 输入形状不匹配" << std::endl;
        std::cerr << "  3. 算子实现有bug" << std::endl;
        std::cerr << "\n建议:" << std::endl;
        std::cerr << "  - 检查上面的算子列表，确认缺失的算子" << std::endl;
        std::cerr << "  - 查看日志获取详细错误信息" << std::endl;
        return 1;
    }
    
    std::cout << "✅ 推理成功 (耗时: " << infer_time.count() << " ms)" << std::endl;
    
    // 7. 获取输出
    std::cout << "\n[5/5] 输出结果:" << std::endl;
    auto output_names = session->GetOutputNames();
    for (size_t i = 0; i < output_names.size(); ++i) {
        auto output_tensor = session->GetOutputTensor(i);
        if (output_tensor) {
            const auto& shape = output_tensor->GetShape();
            size_t element_count = output_tensor->GetElementCount();
            
            std::cout << "  " << output_names[i] << ":" << std::endl;
            std::cout << "    形状: (";
            for (size_t j = 0; j < shape.dims.size(); ++j) {
                std::cout << shape.dims[j];
                if (j < shape.dims.size() - 1) std::cout << ", ";
            }
            std::cout << ")" << std::endl;
            std::cout << "    元素数量: " << element_count << std::endl;
            
            // 打印前几个值
            if (output_tensor->GetDataType() == inferunity::DataType::FLOAT32) {
                const float* data = static_cast<const float*>(output_tensor->GetData());
                size_t print_count = std::min(static_cast<size_t>(10), element_count);
                std::cout << "    前" << print_count << "个值: ";
                for (size_t j = 0; j < print_count; ++j) {
                    std::cout << std::fixed << std::setprecision(6) << data[j] << " ";
                }
                std::cout << std::endl;
            } else if (output_tensor->GetDataType() == inferunity::DataType::INT64) {
                const int64_t* data = static_cast<const int64_t*>(output_tensor->GetData());
                size_t print_count = std::min(static_cast<size_t>(10), element_count);
                std::cout << "    前" << print_count << "个值: ";
                for (size_t j = 0; j < print_count; ++j) {
                    std::cout << data[j] << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    
    // 8. 性能分析
    if (options.enable_profiling) {
        std::cout << "\n=== 性能分析 ===" << std::endl;
        inferunity::ProfilingResult result;
        status = session->Profile(result);
        if (status.IsOk()) {
            std::cout << "总执行时间: " << result.total_time_ms << " ms" << std::endl;
            std::cout << "峰值内存: " << result.peak_memory_bytes / 1024.0 / 1024.0 << " MB" << std::endl;
            
            // 按执行时间排序
            std::vector<inferunity::ProfilingResult::NodeProfile> sorted_profiles = result.node_profiles;
            std::sort(sorted_profiles.begin(), sorted_profiles.end(),
                     [](const auto& a, const auto& b) {
                         return a.execution_time_ms > b.execution_time_ms;
                     });
            
            std::cout << "\n最耗时的算子 (前10个):" << std::endl;
            size_t print_count = std::min(static_cast<size_t>(10), sorted_profiles.size());
            for (size_t i = 0; i < print_count; ++i) {
                const auto& profile = sorted_profiles[i];
                std::cout << "  " << std::setw(6) << std::fixed << std::setprecision(2) 
                          << profile.execution_time_ms << " ms - " 
                          << profile.node_name << " [" << profile.op_type << "]" << std::endl;
            }
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "✅ 测试完成！" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}

