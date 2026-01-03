// 性能基准测试工具
// 用于测试InferUnity的推理性能

#include "inferunity/engine.h"
#include "inferunity/tensor.h"
#include "inferunity/memory.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <cmath>

using namespace inferunity;
using namespace std::chrono;

struct BenchmarkResult {
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    double std_dev_ms;
    size_t memory_used_bytes;
    size_t peak_memory_bytes;
};

// 运行基准测试
BenchmarkResult RunBenchmark(InferenceSession* session, 
                            const std::vector<std::shared_ptr<Tensor>>& inputs,
                            int warmup_iterations = 10,
                            int test_iterations = 100) {
    BenchmarkResult result = {0.0, 0.0, 0.0, 0.0, 0, 0};
    
    if (!session) {
        std::cerr << "Error: Session is null" << std::endl;
        return result;
    }
    
    // 准备输入（转换为Tensor*）
    std::vector<Tensor*> input_tensors;
    for (auto& tensor : inputs) {
        input_tensors.push_back(tensor.get());
    }
    
    std::vector<Tensor*> output_tensors;
    
    // Warmup
    std::cout << "Warming up (" << warmup_iterations << " iterations)..." << std::endl;
    for (int i = 0; i < warmup_iterations; ++i) {
        auto status = session->Run(input_tensors, output_tensors);
        if (!status.IsOk()) {
            std::cerr << "Warmup failed: " << status.Message() << std::endl;
            return result;
        }
    }
    
    // 获取初始内存统计
    MemoryStats initial_stats = GetMemoryStats(DeviceType::CPU);
    
    // 实际测试
    std::cout << "Running benchmark (" << test_iterations << " iterations)..." << std::endl;
    std::vector<double> times;
    times.reserve(test_iterations);
    
    for (int i = 0; i < test_iterations; ++i) {
        auto start = high_resolution_clock::now();
        
        auto status = session->Run(input_tensors, output_tensors);
        if (!status.IsOk()) {
            std::cerr << "Run failed: " << status.Message() << std::endl;
            return result;
        }
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        times.push_back(duration.count() / 1000.0);  // 转换为毫秒
    }
    
    // 获取最终内存统计
    MemoryStats final_stats = GetMemoryStats(DeviceType::CPU);
    
    // 计算统计信息
    double sum = 0.0;
    result.min_time_ms = times[0];
    result.max_time_ms = times[0];
    
    for (double time : times) {
        sum += time;
        if (time < result.min_time_ms) result.min_time_ms = time;
        if (time > result.max_time_ms) result.max_time_ms = time;
    }
    
    result.avg_time_ms = sum / test_iterations;
    
    // 计算标准差
    double variance = 0.0;
    for (double time : times) {
        double diff = time - result.avg_time_ms;
        variance += diff * diff;
    }
    result.std_dev_ms = std::sqrt(variance / test_iterations);
    
    // 内存使用
    result.memory_used_bytes = final_stats.allocated_bytes;
    result.peak_memory_bytes = final_stats.peak_allocated_bytes;
    
    return result;
}

// 打印结果
void PrintResults(const BenchmarkResult& result) {
    std::cout << "\n========== Benchmark Results ==========" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Average time:  " << result.avg_time_ms << " ms" << std::endl;
    std::cout << "Min time:      " << result.min_time_ms << " ms" << std::endl;
    std::cout << "Max time:      " << result.max_time_ms << " ms" << std::endl;
    std::cout << "Std deviation: " << result.std_dev_ms << " ms" << std::endl;
    std::cout << "Throughput:    " << (1000.0 / result.avg_time_ms) << " inferences/sec" << std::endl;
    std::cout << "\nMemory Usage:" << std::endl;
    std::cout << "  Current:     " << (result.memory_used_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  Peak:        " << (result.peak_memory_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "=======================================" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path> [warmup_iterations] [test_iterations]" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    int warmup_iterations = (argc > 2) ? std::stoi(argv[2]) : 10;
    int test_iterations = (argc > 3) ? std::stoi(argv[3]) : 100;
    
    std::cout << "InferUnity Benchmark Tool" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Warmup iterations: " << warmup_iterations << std::endl;
    std::cout << "Test iterations: " << test_iterations << std::endl;
    std::cout << std::endl;
    
    // 创建会话
    SessionOptions options;
    options.execution_providers = {"CPUExecutionProvider"};
    options.graph_optimization_level = SessionOptions::GraphOptimizationLevel::ALL;
    
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
    
    // 获取输入名称和形状
    auto input_names = session->GetInputNames();
    auto input_shapes = session->GetInputShapes();
    
    if (input_shapes.empty()) {
        std::cerr << "No input shapes found" << std::endl;
        return 1;
    }
    
    // 创建输入张量（使用随机数据）
    std::vector<std::shared_ptr<Tensor>> inputs;
    for (size_t i = 0; i < input_shapes.size(); ++i) {
        const Shape& shape = input_shapes[i];
        auto tensor = CreateTensor(shape, DataType::FLOAT32, DeviceType::CPU);
        
        if (!tensor) {
            std::cerr << "Failed to create input tensor " << i << std::endl;
            return 1;
        }
        
        // 填充随机数据
        float* data = static_cast<float*>(tensor->GetData());
        if (data) {
            size_t count = tensor->GetElementCount();
            for (size_t j = 0; j < count; ++j) {
                data[j] = static_cast<float>(rand()) / RAND_MAX;
            }
        }
        
        inputs.push_back(tensor);
    }
    
    // 运行基准测试
    BenchmarkResult result = RunBenchmark(session.get(), inputs, warmup_iterations, test_iterations);
    
    // 打印结果
    PrintResults(result);
    
    return 0;
}

