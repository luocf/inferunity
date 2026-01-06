// 运行时系统
// 包含执行引擎、调度器、线程池等

#pragma once

#include "types.h"
#include "tensor.h"
#include "graph.h"
#include "operator.h"
#include "backend.h"
#include <memory>
#include <vector>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <future>
#include <string>

namespace inferunity {

// ExecutionContext 在 operator.h 中定义，这里只做前向声明
// 使用 operator.h 中的具体实现类

// 调度器基类
class Scheduler {
public:
    virtual ~Scheduler() = default;
    virtual Status Schedule(const Graph* graph,
                           const std::vector<Backend*>& backends,
                           ExecutionContext* ctx) = 0;
    virtual std::vector<Node*> GetExecutionOrder(const Graph* graph) const = 0;
};

// TopologicalScheduler
class TopologicalScheduler : public Scheduler {
public:
    Status Schedule(const Graph* graph,
                   const std::vector<Backend*>& backends,
                   ExecutionContext* ctx) override;
    std::vector<Node*> GetExecutionOrder(const Graph* graph) const override;
};

// PipelineScheduler
class PipelineScheduler : public Scheduler {
public:
    explicit PipelineScheduler(int num_stages = 4);
    Status Schedule(const Graph* graph,
                   const std::vector<Backend*>& backends,
                   ExecutionContext* ctx) override;
    std::vector<Node*> GetExecutionOrder(const Graph* graph) const override;
private:
    int num_stages_;
    std::vector<std::vector<Node*>> stages_;
    void PartitionGraph(const Graph* graph);
};

// ParallelScheduler
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

// 线程池
class ThreadPool {
public:
    static void EnqueueTask(std::function<void()> task);
    static void WaitAll();
    static size_t GetThreadCount();
    static size_t GetPendingTaskCount();
};

// 执行引擎
class ExecutionEngine {
public:
    ExecutionEngine();
    ~ExecutionEngine();
    
    // 设置执行提供者
    void SetBackends(const std::vector<std::shared_ptr<ExecutionProvider>>& backends);
    
    // 设置调度器
    void SetScheduler(std::unique_ptr<Scheduler> scheduler);
    
    // 执行图（使用Tensor输入输出）
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
    
    // 执行图（使用ExecutionContext）
    Status ExecuteGraph(Graph* graph, ExecutionContext* ctx);
    
    // 异步执行（使用ExecutionContext）
    Status ExecuteGraphAsync(Graph* graph, ExecutionContext* ctx,
                             std::function<void(Status)> callback);
    
    // 并行执行多个图
    Status ExecuteGraphsParallel(const std::vector<Graph*>& graphs,
                                const std::vector<ExecutionContext*>& contexts);
    
private:
    std::vector<std::shared_ptr<ExecutionProvider>> backends_;
    std::vector<ExecutionProvider*> backend_ptrs_;
    std::unique_ptr<Scheduler> scheduler_;
    Status ExecuteInternal(const Graph* graph,
                          const std::vector<Tensor*>& inputs,
                          std::vector<Tensor*>& outputs,
                          const ExecutionOptions& options);
};

} // namespace inferunity
