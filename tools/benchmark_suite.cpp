// 基准测试套件
// 参考ONNX Runtime的benchmark实现

#include "inferunity/engine.h"
#include "inferunity/tensor.h"
#include "inferunity/memory.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <sstream>

using namespace inferunity;
using namespace std::chrono;

struct ModelBenchmark {
    std::string model_name;
    std::string model_path;
    std::vector<Shape> input_shapes;
    std::vector<Shape> output_shapes;
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    double throughput;
    size_t memory_used_mb;
    bool success;
    std::string error_message;
};

struct ComparisonResult {
    std::string model_name;
    double baseline_time_ms;
    double current_time_ms;
    double speedup;
    double memory_baseline_mb;
    double memory_current_mb;
    double memory_change_percent;
};

// 运行单个模型基准测试
ModelBenchmark RunModelBenchmark(const std::string& model_path, 
                                int warmup_iterations = 10,
                                int test_iterations = 100) {
    ModelBenchmark result;
    result.model_name = model_path;
    result.model_path = model_path;
    result.success = false;
    
    try {
        // 创建会话
        SessionOptions options;
        options.execution_providers = {"CPUExecutionProvider"};
        options.graph_optimization_level = SessionOptions::GraphOptimizationLevel::ALL;
        
        auto session = InferenceSession::Create(options);
        if (!session) {
            result.error_message = "Failed to create session";
            return result;
        }
        
        // 加载模型
        auto status = session->LoadModel(model_path);
        if (!status.IsOk()) {
            result.error_message = "Failed to load model: " + status.Message();
            return result;
        }
        
        // 获取输入输出信息
        result.input_shapes = session->GetInputShapes();
        result.output_shapes = session->GetOutputShapes();
        
        // 准备输入
        std::vector<std::shared_ptr<Tensor>> inputs;
        std::vector<Tensor*> input_ptrs;
        
        for (const auto& shape : result.input_shapes) {
            auto tensor = CreateTensor(shape, DataType::FLOAT32, DeviceType::CPU);
            if (!tensor) {
                result.error_message = "Failed to create input tensor";
                return result;
            }
            
            // 填充随机数据
            float* data = static_cast<float*>(tensor->GetData());
            size_t count = tensor->GetElementCount();
            for (size_t i = 0; i < count; ++i) {
                data[i] = static_cast<float>(rand()) / RAND_MAX;
            }
            
            inputs.push_back(tensor);
            input_ptrs.push_back(tensor.get());
        }
        
        // Warmup
        std::vector<Tensor*> outputs;
        for (int i = 0; i < warmup_iterations; ++i) {
            session->Run(input_ptrs, outputs);
        }
        
        // 获取初始内存
        MemoryStats initial_stats = GetMemoryStats(DeviceType::CPU);
        
        // 运行测试
        std::vector<double> times;
        times.reserve(test_iterations);
        
        for (int i = 0; i < test_iterations; ++i) {
            auto start = high_resolution_clock::now();
            status = session->Run(input_ptrs, outputs);
            auto end = high_resolution_clock::now();
            
            if (!status.IsOk()) {
                result.error_message = "Run failed: " + status.Message();
                return result;
            }
            
            auto duration = duration_cast<microseconds>(end - start);
            times.push_back(duration.count() / 1000.0);
        }
        
        // 获取最终内存
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
        result.throughput = 1000.0 / result.avg_time_ms;
        result.memory_used_mb = (final_stats.peak_allocated_bytes - initial_stats.allocated_bytes) / 1024.0 / 1024.0;
        result.success = true;
        
    } catch (const std::exception& e) {
        result.error_message = std::string("Exception: ") + e.what();
    }
    
    return result;
}

// 加载模型列表
std::vector<std::string> LoadModelList(const std::string& list_file) {
    std::vector<std::string> models;
    std::ifstream file(list_file);
    
    if (!file.is_open()) {
        std::cerr << "Warning: Cannot open model list file: " << list_file << std::endl;
        return models;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // 跳过空行和注释
        if (line.empty() || line[0] == '#') continue;
        
        // 去除前后空白
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        
        if (!line.empty()) {
            models.push_back(line);
        }
    }
    
    return models;
}

// 保存基准测试结果
void SaveBenchmarkResults(const std::vector<ModelBenchmark>& results, 
                         const std::string& output_file) {
    std::ofstream file(output_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return;
    }
    
    // CSV格式
    file << "Model,Avg Time (ms),Min Time (ms),Max Time (ms),Throughput (inferences/sec),Memory (MB),Status\n";
    
    for (const auto& result : results) {
        file << result.model_name << ","
             << result.avg_time_ms << ","
             << result.min_time_ms << ","
             << result.max_time_ms << ","
             << result.throughput << ","
             << result.memory_used_mb << ","
             << (result.success ? "SUCCESS" : "FAILED") << "\n";
    }
    
    file.close();
    std::cout << "Results saved to: " << output_file << std::endl;
}

// 加载基准结果（用于对比）
std::map<std::string, ModelBenchmark> LoadBaselineResults(const std::string& baseline_file) {
    std::map<std::string, ModelBenchmark> baseline;
    std::ifstream file(baseline_file);
    
    if (!file.is_open()) {
        return baseline;
    }
    
    std::string line;
    std::getline(file, line);  // 跳过头部
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        ModelBenchmark result;
        
        std::getline(iss, result.model_name, ',');
        std::getline(iss, token, ',');
        result.avg_time_ms = std::stod(token);
        std::getline(iss, token, ',');
        result.min_time_ms = std::stod(token);
        std::getline(iss, token, ',');
        result.max_time_ms = std::stod(token);
        std::getline(iss, token, ',');
        result.throughput = std::stod(token);
        std::getline(iss, token, ',');
        result.memory_used_mb = std::stod(token);
        
        baseline[result.model_name] = result;
    }
    
    return baseline;
}

// 性能对比
std::vector<ComparisonResult> CompareResults(
    const std::vector<ModelBenchmark>& current,
    const std::map<std::string, ModelBenchmark>& baseline) {
    
    std::vector<ComparisonResult> comparisons;
    
    for (const auto& current_result : current) {
        if (!current_result.success) continue;
        
        auto it = baseline.find(current_result.model_name);
        if (it == baseline.end()) continue;
        
        ComparisonResult comp;
        comp.model_name = current_result.model_name;
        comp.baseline_time_ms = it->second.avg_time_ms;
        comp.current_time_ms = current_result.avg_time_ms;
        comp.speedup = comp.baseline_time_ms / comp.current_time_ms;
        comp.memory_baseline_mb = it->second.memory_used_mb;
        comp.memory_current_mb = current_result.memory_used_mb;
        
        if (comp.memory_baseline_mb > 0) {
            comp.memory_change_percent = 
                (comp.memory_current_mb - comp.memory_baseline_mb) / comp.memory_baseline_mb * 100.0;
        } else {
            comp.memory_change_percent = 0.0;
        }
        
        comparisons.push_back(comp);
    }
    
    return comparisons;
}

// 打印对比结果
void PrintComparison(const std::vector<ComparisonResult>& comparisons) {
    std::cout << "\n========== Performance Comparison ==========" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    
    std::cout << std::setw(30) << "Model"
              << std::setw(15) << "Baseline (ms)"
              << std::setw(15) << "Current (ms)"
              << std::setw(15) << "Speedup"
              << std::setw(15) << "Memory Δ%" << std::endl;
    std::cout << std::string(90, '-') << std::endl;
    
    for (const auto& comp : comparisons) {
        std::cout << std::setw(30) << comp.model_name
                  << std::setw(15) << comp.baseline_time_ms
                  << std::setw(15) << comp.current_time_ms
                  << std::setw(15) << comp.speedup << "x"
                  << std::setw(15) << comp.memory_change_percent << "%" << std::endl;
    }
    
    std::cout << std::string(90, '-') << std::endl;
}

// 回归测试（检查性能是否退化）
bool RunRegressionTest(const std::vector<ComparisonResult>& comparisons,
                      double max_slowdown = 1.1,  // 允许10%的减速
                      double max_memory_increase = 1.2) {  // 允许20%的内存增加
    
    bool passed = true;
    
    std::cout << "\n========== Regression Test ==========" << std::endl;
    
    for (const auto& comp : comparisons) {
        bool model_passed = true;
        std::string issues;
        
        // 检查性能退化
        if (comp.speedup < 1.0 / max_slowdown) {
            model_passed = false;
            issues += "Performance regression (slowdown: " + 
                     std::to_string(1.0 / comp.speedup) + "x); ";
        }
        
        // 检查内存增加
        if (comp.memory_change_percent > (max_memory_increase - 1.0) * 100.0) {
            model_passed = false;
            issues += "Memory increase: " + std::to_string(comp.memory_change_percent) + "%; ";
        }
        
        std::cout << std::setw(30) << comp.model_name << ": ";
        if (model_passed) {
            std::cout << "PASS" << std::endl;
        } else {
            std::cout << "FAIL - " << issues << std::endl;
            passed = false;
        }
    }
    
    std::cout << "\nOverall: " << (passed ? "PASS" : "FAIL") << std::endl;
    return passed;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <command> [options]" << std::endl;
        std::cerr << "\nCommands:" << std::endl;
        std::cerr << "  run <model_list> [output]     Run benchmark on models" << std::endl;
        std::cerr << "  compare <current> <baseline> Compare current vs baseline" << std::endl;
        std::cerr << "  regression <current> <baseline> Run regression test" << std::endl;
        return 1;
    }
    
    std::string command = argv[1];
    
    if (command == "run") {
        if (argc < 3) {
            std::cerr << "Error: Missing model list file" << std::endl;
            return 1;
        }
        
        std::string model_list = argv[2];
        std::string output_file = (argc > 3) ? argv[3] : "benchmark_results.csv";
        
        std::cout << "Loading model list from: " << model_list << std::endl;
        auto models = LoadModelList(model_list);
        
        if (models.empty()) {
            std::cerr << "No models found in list" << std::endl;
            return 1;
        }
        
        std::cout << "Found " << models.size() << " models" << std::endl;
        std::cout << "Running benchmarks..." << std::endl;
        
        std::vector<ModelBenchmark> results;
        for (size_t i = 0; i < models.size(); ++i) {
            std::cout << "\n[" << (i + 1) << "/" << models.size() << "] " 
                      << models[i] << std::endl;
            
            auto result = RunModelBenchmark(models[i]);
            results.push_back(result);
            
            if (result.success) {
                std::cout << "  Avg time: " << result.avg_time_ms << " ms" << std::endl;
                std::cout << "  Throughput: " << result.throughput << " inferences/sec" << std::endl;
                std::cout << "  Memory: " << result.memory_used_mb << " MB" << std::endl;
            } else {
                std::cout << "  FAILED: " << result.error_message << std::endl;
            }
        }
        
        SaveBenchmarkResults(results, output_file);
        
    } else if (command == "compare") {
        if (argc < 4) {
            std::cerr << "Error: Missing current or baseline file" << std::endl;
            return 1;
        }
        
        std::string current_file = argv[2];
        std::string baseline_file = argv[3];
        
        // 加载结果
        auto current_models = LoadModelList(current_file);
        std::vector<ModelBenchmark> current_results;
        for (const auto& model : current_models) {
            current_results.push_back(RunModelBenchmark(model, 5, 20));
        }
        
        auto baseline = LoadBaselineResults(baseline_file);
        auto comparisons = CompareResults(current_results, baseline);
        
        PrintComparison(comparisons);
        
    } else if (command == "regression") {
        if (argc < 4) {
            std::cerr << "Error: Missing current or baseline file" << std::endl;
            return 1;
        }
        
        std::string current_file = argv[2];
        std::string baseline_file = argv[3];
        
        // 加载结果
        auto current_models = LoadModelList(current_file);
        std::vector<ModelBenchmark> current_results;
        for (const auto& model : current_models) {
            current_results.push_back(RunModelBenchmark(model, 5, 20));
        }
        
        auto baseline = LoadBaselineResults(baseline_file);
        auto comparisons = CompareResults(current_results, baseline);
        
        bool passed = RunRegressionTest(comparisons);
        return passed ? 0 : 1;
        
    } else {
        std::cerr << "Error: Unknown command: " << command << std::endl;
        return 1;
    }
    
    return 0;
}

