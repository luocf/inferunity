// 多线程并行执行器
// 参考ONNX Runtime的并行执行实现

#include "inferunity/runtime.h"
#include "inferunity/backend.h"
#include "inferunity/graph.h"
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <algorithm>
#include <unordered_map>

namespace inferunity {

// ParallelScheduler实现
ParallelScheduler::ParallelScheduler(int num_threads) 
    : num_threads_(num_threads == 0 ? std::thread::hardware_concurrency() : num_threads) {
}

Status ParallelScheduler::Schedule(const Graph* graph,
                   const std::vector<Backend*>& backends,
                   ExecutionContext* ctx) {
        if (!graph || backends.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid arguments");
        }
        
        // 获取拓扑排序
        std::vector<Node*> execution_order = graph->TopologicalSort();
        
        // 计算每个节点的依赖计数
        std::unordered_map<Node*, int> ready_count;
        for (Node* node : execution_order) {
            ready_count[node] = node->GetInputs().size();
        }
        
        // 准备就绪队列（没有依赖的节点）
        std::queue<Node*> ready_queue;
        for (Node* node : execution_order) {
            if (ready_count[node] == 0) {
                ready_queue.push(node);
            }
        }
        
        // 多线程执行
        std::mutex queue_mutex;
        std::condition_variable cv;
        std::atomic<int> completed_count(0);
        std::vector<std::thread> workers;
        std::atomic<bool> has_error(false);
        Status error_status = Status::Ok();
        
        for (int i = 0; i < num_threads_; ++i) {
            workers.emplace_back([&, i]() {
                while (true) {
                    Node* node = nullptr;
                    
                    // 获取下一个就绪节点
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        cv.wait(lock, [&]() {
                            return !ready_queue.empty() || completed_count.load() == execution_order.size() || has_error.load();
                        });
                        
                        if (completed_count.load() == execution_order.size() || has_error.load()) {
                            break;  // 所有节点已完成或出错
                        }
                        
                        if (ready_queue.empty()) {
                            continue;
                        }
                        
                        node = ready_queue.front();
                        ready_queue.pop();
                    }
                    
                    if (!node) continue;
                    
                    // 执行节点
                    Backend* backend = backends[0];  // 简化：使用第一个backend
                    Status status = backend->ExecuteNode(node, ctx);
                    
                    if (!status.IsOk()) {
                        has_error = true;
                        error_status = status;
                        completed_count = execution_order.size();  // 出错，停止所有线程
                        cv.notify_all();
                        break;
                    }
                    
                    // 更新依赖计数，将新就绪的节点加入队列
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        completed_count++;
                        
                        // 检查该节点的输出，更新消费者的依赖计数
                        for (Value* output : node->GetOutputs()) {
                            for (Node* consumer : output->GetConsumers()) {
                                ready_count[consumer]--;
                                if (ready_count[consumer] == 0) {
                                    ready_queue.push(consumer);
                                }
                            }
                        }
                        
                        cv.notify_all();
                    }
                }
            });
        }
        
        // 等待所有线程完成
        for (auto& worker : workers) {
            worker.join();
        }
        
        // 检查是否有错误
        if (has_error.load()) {
            return error_status;
        }
        
        return Status::Ok();
    }
    
std::vector<Node*> ParallelScheduler::GetExecutionOrder(const Graph* graph) const {
    return graph->TopologicalSort();
}

} // namespace inferunity

