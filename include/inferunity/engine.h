#pragma once

#include "types.h"
#include "tensor.h"
#include "graph.h"
#include "runtime.h"
#include "backend.h"
#include "optimizer.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <future>

namespace inferunity {

// 会话配置 (参考ONNX Runtime的SessionOptions)
struct SessionOptions {
    // 执行提供者配置 (参考ONNX Runtime的ExecutionProvider配置)
    std::vector<std::string> execution_providers;
    int device_id = 0;
    
    // 图优化配置 (参考ONNX Runtime的GraphOptimizationLevel)
    enum class GraphOptimizationLevel {
        NONE = 0,           // 不优化
        BASIC = 1,          // 基础优化
        EXTENDED = 2,       // 扩展优化
        ALL = 99            // 全部优化
    };
    GraphOptimizationLevel graph_optimization_level = GraphOptimizationLevel::ALL;
    bool enable_operator_fusion = true;
    bool enable_quantization = false;
    DataType quantization_dtype = DataType::INT8;
    
    // 性能配置
    int num_threads = 0;  // 0表示自动 (参考ONNX Runtime的线程配置)
    int max_batch_size = 1;
    bool enable_profiling = false;
    
    // 内存配置 (参考NCNN的内存池配置)
    size_t memory_pool_size = 0;  // 0表示无限制
};

// 推理会话 (参考ONNX Runtime的InferenceSession设计)
// 使用Session而非Engine，更符合主流推理引擎的命名习惯
class InferenceSession {
public:
    // 创建会话 (参考ONNX Runtime的InferenceSession::Create)
    static std::unique_ptr<InferenceSession> Create(const SessionOptions& options = SessionOptions());
    
    ~InferenceSession();
    
    // 模型加载
    Status LoadModel(const std::string& filepath);
    Status LoadModelFromMemory(const void* data, size_t size);
    Status LoadModelFromGraph(std::unique_ptr<Graph> graph);
    
    // 模型信息
    const Graph* GetGraph() const { return graph_.get(); }
    std::vector<Shape> GetInputShapes() const;
    std::vector<Shape> GetOutputShapes() const;
    std::vector<std::string> GetInputNames() const;
    std::vector<std::string> GetOutputNames() const;
    
    // 推理
    Status Run(const std::vector<Tensor*>& inputs, std::vector<Tensor*>& outputs);
    Status Run(const std::unordered_map<std::string, Tensor*>& inputs,
               std::unordered_map<std::string, Tensor*>& outputs);
    
    // 异步推理
    std::future<Status> RunAsync(const std::vector<Tensor*>& inputs,
                                std::vector<Tensor*>& outputs);
    
    // 张量创建
    std::shared_ptr<Tensor> CreateInputTensor(size_t input_index);
    std::shared_ptr<Tensor> CreateInputTensor(const std::string& input_name);
    std::shared_ptr<Tensor> GetOutputTensor(size_t output_index);
    std::shared_ptr<Tensor> GetOutputTensor(const std::string& output_name);
    
    // 性能分析
    Status Profile(ProfilingResult& result);
    
    // 配置 (参考ONNX Runtime的配置管理)
    void SetOptions(const SessionOptions& options);
    const SessionOptions& GetOptions() const { return options_; }
    
    // 为了向后兼容，保留Engine作为别名
    using Engine = InferenceSession;
    using EngineConfig = SessionOptions;
    
private:
    InferenceSession(const SessionOptions& options);
    
    Status Initialize();
    Status LoadAndOptimizeGraph();
    Status PrepareExecutionProviders();
    
    SessionOptions options_;
    std::unique_ptr<Graph> graph_;
    std::unique_ptr<Optimizer> optimizer_;
    std::unique_ptr<ExecutionEngine> execution_engine_;
    std::vector<std::shared_ptr<ExecutionProvider>> execution_providers_;
    bool initialized_;
};

} // namespace inferunity

