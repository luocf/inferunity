#include "inferunity/runtime.h"
#include "inferunity/backend.h"
#include "inferunity/tensor.h"
#include "inferunity/operator.h"
#include <algorithm>
#include <thread>
#include <chrono>
#include <future>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>

namespace inferunity {

// TopologicalScheduler实现
Status TopologicalScheduler::Schedule(const Graph* graph,
                                      const std::vector<Backend*>& backends,
                                      ExecutionContext* ctx) {
    // 拓扑排序已经在GetExecutionOrder中完成
    (void)graph; (void)backends; (void)ctx;
    return Status::Ok();
}

std::vector<Node*> TopologicalScheduler::GetExecutionOrder(const Graph* graph) const {
    return graph->TopologicalSort();
}

// PipelineScheduler实现
Status PipelineScheduler::Schedule(const Graph* graph,
                                  const std::vector<Backend*>& backends,
                                  ExecutionContext* ctx) {
    PartitionGraph(graph);
    (void)backends; (void)ctx;
    return Status::Ok();
}

std::vector<Node*> PipelineScheduler::GetExecutionOrder(const Graph* graph) const {
    std::vector<Node*> order;
    for (const auto& stage : stages_) {
        order.insert(order.end(), stage.begin(), stage.end());
    }
    return order;
}

void PipelineScheduler::PartitionGraph(const Graph* graph) {
    stages_.clear();
    stages_.resize(num_stages_);
    
    std::vector<Node*> sorted = graph->TopologicalSort();
    size_t nodes_per_stage = (sorted.size() + num_stages_ - 1) / num_stages_;
    
    for (size_t i = 0; i < sorted.size(); ++i) {
        int stage = std::min(static_cast<int>(i / nodes_per_stage), num_stages_ - 1);
        stages_[stage].push_back(sorted[i]);
    }
}

// ExecutionEngine实现
ExecutionEngine::ExecutionEngine() {
    scheduler_ = std::make_unique<TopologicalScheduler>();
}

ExecutionEngine::~ExecutionEngine() = default;

void ExecutionEngine::SetScheduler(std::unique_ptr<Scheduler> scheduler) {
    scheduler_ = std::move(scheduler);
}

void ExecutionEngine::SetBackends(const std::vector<std::shared_ptr<Backend>>& backends) {
    backends_ = backends;
    backend_ptrs_.clear();
    for (const auto& backend : backends_) {
        backend_ptrs_.push_back(backend.get());
    }
}

Status ExecutionEngine::ExecuteGraph(Graph* graph, ExecutionContext* ctx) {
    if (!graph || !ctx) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid arguments");
    }
    
    if (backend_ptrs_.empty()) {
        return Status::Error(StatusCode::ERROR_RUNTIME_ERROR, "No backends available");
    }
    
    // 使用调度器执行图
    return scheduler_->Schedule(graph, backend_ptrs_, ctx);
}

Status ExecutionEngine::ExecuteGraphAsync(Graph* graph, ExecutionContext* ctx,
                                         std::function<void(Status)> callback) {
    // 使用线程池异步执行
    ThreadPool::EnqueueTask([this, graph, ctx, callback]() {
        Status status = this->ExecuteGraph(graph, ctx);
        if (callback) {
            callback(status);
        }
    });
    
    return Status::Ok();
}

Status ExecutionEngine::ExecuteGraphsParallel(const std::vector<Graph*>& graphs,
                                             const std::vector<ExecutionContext*>& contexts) {
    if (graphs.size() != contexts.size()) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Graph and context count mismatch");
    }
    
    // 使用std::async并行执行多个图
    std::vector<std::future<Status>> futures;
    for (size_t i = 0; i < graphs.size(); ++i) {
        futures.push_back(std::async(std::launch::async, [this, &graphs, &contexts, i]() {
            return this->ExecuteGraph(graphs[i], contexts[i]);
        }));
    }
    
    // 等待所有任务完成并检查错误
    for (auto& future : futures) {
        Status status = future.get();
        if (!status.IsOk()) {
            return status;
        }
    }
    
    return Status::Ok();
}

Status ExecutionEngine::Execute(const Graph* graph,
                               const std::vector<Tensor*>& inputs,
                               std::vector<Tensor*>& outputs,
                               const ExecutionOptions& options) {
    return ExecuteInternal(graph, inputs, outputs, options);
}

std::future<Status> ExecutionEngine::ExecuteAsync(const Graph* graph,
                                                  const std::vector<Tensor*>& inputs,
                                                  std::vector<Tensor*>& outputs,
                                                  const ExecutionOptions& options) {
    return std::async(std::launch::async, [this, graph, &inputs, &outputs, options]() {
        return ExecuteInternal(graph, inputs, outputs, options);
    });
}

Status ExecutionEngine::Profile(const Graph* graph,
                              const std::vector<Tensor*>& inputs,
                              ProfilingResult& result) {
    if (!graph) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Graph is null");
    }
    
    ExecutionOptions options;
    options.enable_profiling = true;
    
    // 清空结果
    result.node_profiles.clear();
    result.total_time_ms = 0.0;
    result.peak_memory_bytes = 0;
    
    // 记录开始时间
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 获取执行顺序
    std::vector<Node*> execution_order = scheduler_->GetExecutionOrder(graph);
    
    // 执行每个节点并记录性能
    size_t peak_memory = 0;
    for (Node* node : execution_order) {
        auto node_start = std::chrono::high_resolution_clock::now();
        
        // 准备输入
        std::vector<Tensor*> node_inputs;
        for (Value* input : node->GetInputs()) {
            if (input->GetTensor()) {
                node_inputs.push_back(input->GetTensor().get());
            }
        }
        
        // 执行节点
        ExecutionContext ctx;
        if (!backend_ptrs_.empty()) {
            ExecutionProvider* provider = ExecutionProviderSelector::SelectProvider(node, backend_ptrs_);
            if (provider) {
                ctx.SetDeviceType(provider->GetDeviceType());
                Status status = provider->ExecuteNode(node, &ctx);
                if (!status.IsOk()) {
                    return status;
                }
            }
        }
        
        auto node_end = std::chrono::high_resolution_clock::now();
        double node_time_ms = std::chrono::duration<double, std::milli>(node_end - node_start).count();
        
        // 估算内存使用（简化：使用张量大小）
        size_t node_memory = 0;
        for (Value* output : node->GetOutputs()) {
            if (output->GetTensor()) {
                node_memory += output->GetTensor()->GetSizeInBytes();
            }
        }
        peak_memory = std::max(peak_memory, node_memory);
        
        // 记录节点性能
        ProfilingResult::NodeProfile profile;
        profile.node_name = node->GetName();
        profile.op_type = node->GetOpType();
        profile.execution_time_ms = node_time_ms;
        profile.memory_used_bytes = node_memory;
        result.node_profiles.push_back(profile);
    }
    
    // 记录总时间
    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    result.peak_memory_bytes = peak_memory;
    
    return Status::Ok();
}

Status ExecutionEngine::ExecuteInternal(const Graph* graph,
                                       const std::vector<Tensor*>& inputs,
                                       std::vector<Tensor*>& outputs,
                                       const ExecutionOptions& options) {
    // 验证输入
    if (inputs.size() != graph->GetInputs().size()) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                           "Input count mismatch");
    }
    
    // 绑定输入
    for (size_t i = 0; i < inputs.size(); ++i) {
        graph->GetInputs()[i]->SetTensor(std::shared_ptr<Tensor>(inputs[i], [](Tensor*){}));
    }
    
    // 获取执行顺序
    std::vector<Node*> execution_order = scheduler_->GetExecutionOrder(graph);
    
    // 创建执行上下文
    ExecutionContext ctx;
    
    // 执行每个节点
    for (Node* node : execution_order) {
        // 选择执行提供者 (参考ONNX Runtime的节点执行)
        ExecutionProvider* provider = ExecutionProviderSelector::SelectProvider(node, backend_ptrs_);
        if (!provider) {
            return Status::Error(StatusCode::ERROR_NOT_FOUND,
                               "No execution provider available for node: " + node->GetName());
        }
        
        // 准备输入输出张量 (参考ONNX Runtime的IOBinding机制)
        std::vector<Tensor*> node_inputs;
        std::vector<Tensor*> node_outputs;
        
        for (Value* input : node->GetInputs()) {
            if (input->GetTensor()) {
                node_inputs.push_back(input->GetTensor().get());
            }
        }
        
        for (Value* output : node->GetOutputs()) {
            // 创建输出张量
            // 先推断输出形状
            auto op = provider->CreateOperator(node->GetOpType());
            if (op) {
                std::vector<Shape> output_shapes;
                Status shape_status = op->InferOutputShape(node_inputs, output_shapes);
                if (shape_status.IsOk() && !output_shapes.empty()) {
                    Shape output_shape = output_shapes[0];
                    DataType output_dtype = node_inputs.empty() ? 
                        DataType::FLOAT32 : node_inputs[0]->GetDataType();
                    auto tensor = CreateTensor(output_shape, output_dtype, provider->GetDeviceType());
                    output->SetTensor(tensor);
                    node_outputs.push_back(tensor.get());
                } else {
                    // 形状推断失败，使用默认形状
                    Shape default_shape({1});  // 默认1D
                    auto tensor = CreateTensor(default_shape, DataType::FLOAT32, provider->GetDeviceType());
                    output->SetTensor(tensor);
                    node_outputs.push_back(tensor.get());
                }
            } else {
                // 无法创建算子，使用默认形状
                Shape default_shape({1});
                auto tensor = CreateTensor(default_shape, DataType::FLOAT32, provider->GetDeviceType());
                output->SetTensor(tensor);
                node_outputs.push_back(tensor.get());
            }
        }
        
        // 执行节点 (参考ONNX Runtime的节点执行流程)
        ctx.SetDeviceType(provider->GetDeviceType());
        Status status = provider->ExecuteNode(node, &ctx);
        if (!status.IsOk()) {
            return status;
        }
    }
    
    // 收集输出
    outputs.clear();
    for (Value* output_value : graph->GetOutputs()) {
        if (output_value->GetTensor()) {
            outputs.push_back(output_value->GetTensor().get());
        }
    }
    
    return Status::Ok();
}

} // namespace inferunity

