#pragma once

#include "types.h"
#include "graph.h"
#include "tensor.h"
#include "backend.h"
#include "optimizer.h"
#include <memory>
#include <vector>
#include <functional>
#include <future>

namespace inferunity {

// 执行模式
enum class ExecutionMode {
    SYNCHRONOUS,   // 同步执行
    ASYNCHRONOUS   // 异步执行
};

// 执行选项
struct ExecutionOptions {
    ExecutionMode mode = ExecutionMode::SYNCHRONOUS;
    bool enable_profiling = false;
    int max_parallel_streams = 1;
    std::string backend_preference = "";  // 后端偏好
};

// 性能分析结果
struct ProfilingResult {
    struct NodeProfile {
        std::string node_name;
        std::string op_type;
        double execution_time_ms;
        size_t memory_used_bytes;
    };
    
    std::vector<NodeProfile> node_profiles;
    double total_time_ms;
    size_t peak_memory_bytes;
};

// 调度器接口
class Scheduler {
public:
    virtual ~Scheduler() = default;
    
    // 调度执行
    virtual Status Schedule(const Graph* graph, 
                           const std::vector<Backend*>& backends,
                           ExecutionContext* ctx) = 0;
    
    // 获取执行顺序
    virtual std::vector<Node*> GetExecutionOrder(const Graph* graph) const = 0;
};

// 拓扑排序调度器
class TopologicalScheduler : public Scheduler {
public:
    Status Schedule(const Graph* graph,
                   const std::vector<Backend*>& backends,
                   ExecutionContext* ctx) override;
    
    std::vector<Node*> GetExecutionOrder(const Graph* graph) const override;
};

// 流水线调度器
class PipelineScheduler : public Scheduler {
public:
    explicit PipelineScheduler(int num_stages = 4) : num_stages_(num_stages) {}
    
    Status Schedule(const Graph* graph,
                   const std::vector<Backend*>& backends,
                   ExecutionContext* ctx) override;
    
    std::vector<Node*> GetExecutionOrder(const Graph* graph) const override;
    
private:
    int num_stages_;
    std::vector<std::vector<Node*>> stages_;
    
    void PartitionGraph(const Graph* graph);
};

// 并行调度器（多线程执行）
class ParallelScheduler : public Scheduler {
public:
    explicit ParallelScheduler(int num_threads = 0);
    
    Status Schedule(const Graph* graph,
                   const std::vector<Backend*>& backends,
                   ExecutionContext* ctx) override;
    
    std::vector<Node*> GetExecutionOrder(const Graph* graph) const override;
    
private:
    int num_threads_;
};

// 执行引擎
class ExecutionEngine {
public:
    ExecutionEngine();
    ~ExecutionEngine();
    
    // 设置调度器
    void SetScheduler(std::unique_ptr<Scheduler> scheduler);
    
    // 执行图
    Status Execute(const Graph* graph,
                  const std::vector<Tensor*>& inputs,
                  std::vector<Tensor*>& outputs,
                  const ExecutionOptions& options = ExecutionOptions());
    
    // 异步执行
    std::future<Status> ExecuteAsync(const Graph* graph,
                                    const std::vector<Tensor*>& inputs,
                                    std::vector<Tensor*>& outputs,
                                    const ExecutionOptions& options = ExecutionOptions());
    
    // 性能分析
    Status Profile(const Graph* graph,
                  const std::vector<Tensor*>& inputs,
                  ProfilingResult& result);
    
    // 设置后端
    void SetBackends(const std::vector<std::shared_ptr<Backend>>& backends);
    
private:
    std::unique_ptr<Scheduler> scheduler_;
    std::vector<std::shared_ptr<Backend>> backends_;
    std::vector<Backend*> backend_ptrs_;
    
    Status ExecuteInternal(const Graph* graph,
                          const std::vector<Tensor*>& inputs,
                          std::vector<Tensor*>& outputs,
                          const ExecutionOptions& options);
};

} // namespace inferunity

