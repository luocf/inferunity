// 性能分析工具
// 参考ONNX Runtime的profiler实现

#include "inferunity/engine.h"
#include "inferunity/tensor.h"
#include "inferunity/memory.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cmath>

using namespace inferunity;

// 打印性能分析结果
void PrintProfilingResults(const ProfilingResult& result, bool verbose = false) {
    std::cout << "\n========== Performance Profiling Results ==========" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    
    std::cout << "\nSummary:" << std::endl;
    std::cout << "  Total execution time: " << result.total_time_ms << " ms" << std::endl;
    std::cout << "  Peak memory usage:    " << (result.peak_memory_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  Number of nodes:      " << result.node_profiles.size() << std::endl;
    
    if (verbose && !result.node_profiles.empty()) {
        std::cout << "\nNode-by-Node Breakdown:" << std::endl;
        std::cout << std::setw(30) << "Node Name" 
                  << std::setw(20) << "Op Type"
                  << std::setw(15) << "Time (ms)"
                  << std::setw(15) << "Memory (MB)"
                  << std::setw(15) << "Time %" << std::endl;
        std::cout << std::string(95, '-') << std::endl;
        
        // 按执行时间排序
        std::vector<ProfilingResult::NodeProfile> sorted_profiles = result.node_profiles;
        std::sort(sorted_profiles.begin(), sorted_profiles.end(),
                 [](const ProfilingResult::NodeProfile& a, const ProfilingResult::NodeProfile& b) {
                     return a.execution_time_ms > b.execution_time_ms;
                 });
        
        for (const auto& profile : sorted_profiles) {
            double time_percent = (result.total_time_ms > 0) ? 
                (profile.execution_time_ms / result.total_time_ms * 100.0) : 0.0;
            
            std::cout << std::setw(30) << profile.node_name
                      << std::setw(20) << profile.op_type
                      << std::setw(15) << profile.execution_time_ms
                      << std::setw(15) << (profile.memory_used_bytes / 1024.0 / 1024.0)
                      << std::setw(15) << time_percent << "%" << std::endl;
        }
        
        // 统计各算子类型的总时间
        std::cout << "\nOperator Type Statistics:" << std::endl;
        std::unordered_map<std::string, double> op_type_times;
        std::unordered_map<std::string, int> op_type_counts;
        
        for (const auto& profile : result.node_profiles) {
            op_type_times[profile.op_type] += profile.execution_time_ms;
            op_type_counts[profile.op_type]++;
        }
        
        std::vector<std::pair<std::string, double>> sorted_ops;
        for (const auto& pair : op_type_times) {
            sorted_ops.push_back(pair);
        }
        std::sort(sorted_ops.begin(), sorted_ops.end(),
                 [](const std::pair<std::string, double>& a, 
                    const std::pair<std::string, double>& b) {
                     return a.second > b.second;
                 });
        
        std::cout << std::setw(20) << "Op Type"
                  << std::setw(15) << "Total Time (ms)"
                  << std::setw(15) << "Count"
                  << std::setw(15) << "Avg Time (ms)" << std::endl;
        std::cout << std::string(65, '-') << std::endl;
        
        for (const auto& pair : sorted_ops) {
            double avg_time = op_type_times[pair.first] / op_type_counts[pair.first];
            std::cout << std::setw(20) << pair.first
                      << std::setw(15) << op_type_times[pair.first]
                      << std::setw(15) << op_type_counts[pair.first]
                      << std::setw(15) << avg_time << std::endl;
        }
    }
    
    std::cout << "\n===================================================" << std::endl;
}

// 导出为CSV格式
void ExportToCSV(const ProfilingResult& result, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    // 写入CSV头部
    file << "Node Name,Op Type,Execution Time (ms),Memory Used (bytes),Time %\n";
    
    // 写入数据
    for (const auto& profile : result.node_profiles) {
        double time_percent = (result.total_time_ms > 0) ? 
            (profile.execution_time_ms / result.total_time_ms * 100.0) : 0.0;
        
        file << profile.node_name << ","
             << profile.op_type << ","
             << profile.execution_time_ms << ","
             << profile.memory_used_bytes << ","
             << time_percent << "\n";
    }
    
    file.close();
    std::cout << "Profiling results exported to: " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path> [options]" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  -v, --verbose     Show detailed node-by-node breakdown" << std::endl;
        std::cerr << "  -o, --output FILE Export results to CSV file" << std::endl;
        std::cerr << "  -i, --iterations N Number of profiling iterations (default: 10)" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    bool verbose = false;
    std::string output_file;
    int iterations = 10;
    
    // 解析命令行参数
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                output_file = argv[++i];
            }
        } else if (arg == "-i" || arg == "--iterations") {
            if (i + 1 < argc) {
                iterations = std::stoi(argv[++i]);
            }
        }
    }
    
    std::cout << "InferUnity Profiler" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << std::endl;
    
    // 创建会话
    SessionOptions options;
    options.execution_providers = {"CPUExecutionProvider"};
    options.graph_optimization_level = SessionOptions::GraphOptimizationLevel::ALL;
    options.enable_profiling = true;
    
    auto session = InferenceSession::Create(options);
    if (!session) {
        std::cerr << "Failed to create inference session" << std::endl;
        return 1;
    }
    
    // 加载模型
    auto status = session->LoadModel(model_path);
    if (!status.IsOk()) {
        std::cerr << "Failed to load model: " << status.Message() << std::endl;
        return 1;
    }
    
    // 准备输入
    auto input_shapes = session->GetInputShapes();
    std::vector<std::shared_ptr<Tensor>> inputs;
    
    for (const auto& shape : input_shapes) {
        auto tensor = CreateTensor(shape, DataType::FLOAT32, DeviceType::CPU);
        if (!tensor) {
            std::cerr << "Failed to create input tensor" << std::endl;
            return 1;
        }
        
        // 填充随机数据
        float* data = static_cast<float*>(tensor->GetData());
        size_t count = tensor->GetElementCount();
        for (size_t i = 0; i < count; ++i) {
            data[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        
        inputs.push_back(tensor);
    }
    
    // 转换为Tensor*指针
    std::vector<Tensor*> input_ptrs;
    for (auto& tensor : inputs) {
        input_ptrs.push_back(tensor.get());
    }
    
    // Warmup
    std::cout << "Warming up..." << std::endl;
    std::vector<Tensor*> outputs;
    for (int i = 0; i < 5; ++i) {
        session->Run(input_ptrs, outputs);
    }
    
    // 运行性能分析
    std::cout << "Running profiling (" << iterations << " iterations)..." << std::endl;
    
    ProfilingResult aggregated_result;
    aggregated_result.total_time_ms = 0.0;
    aggregated_result.peak_memory_bytes = 0;
    
    for (int i = 0; i < iterations; ++i) {
        ProfilingResult result;
        status = session->Profile(result);
        if (!status.IsOk()) {
            std::cerr << "Profiling failed: " << status.Message() << std::endl;
            return 1;
        }
        
        // 聚合结果（取平均值）
        if (i == 0) {
            aggregated_result = result;
        } else {
            aggregated_result.total_time_ms += result.total_time_ms;
            aggregated_result.peak_memory_bytes = std::max(
                aggregated_result.peak_memory_bytes, result.peak_memory_bytes);
            
            // 聚合节点性能（取平均值）
            for (size_t j = 0; j < aggregated_result.node_profiles.size() && 
                            j < result.node_profiles.size(); ++j) {
                aggregated_result.node_profiles[j].execution_time_ms += 
                    result.node_profiles[j].execution_time_ms;
            }
        }
    }
    
    // 计算平均值
    aggregated_result.total_time_ms /= iterations;
    for (auto& profile : aggregated_result.node_profiles) {
        profile.execution_time_ms /= iterations;
    }
    
    // 打印结果
    PrintProfilingResults(aggregated_result, verbose);
    
    // 导出CSV
    if (!output_file.empty()) {
        ExportToCSV(aggregated_result, output_file);
    }
    
    return 0;
}

