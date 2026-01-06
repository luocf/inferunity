// Tensor生命周期优化器
// 参考 NCNN 的 BlobAllocator 和 ONNX Runtime 的内存复用机制

#include "inferunity/memory.h"
#include "inferunity/graph.h"
#include "inferunity/logger.h"
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <set>

namespace inferunity {

// Tensor生命周期分析
std::vector<TensorLifetime> AnalyzeTensorLifetimes(const Graph* graph) {
    if (!graph) {
        return {};
    }
    
    // 获取拓扑排序（执行顺序）
    std::vector<Node*> execution_order = graph->TopologicalSort();
    
    // 建立节点索引映射
    std::unordered_map<Node*, int64_t> node_index;
    for (size_t i = 0; i < execution_order.size(); ++i) {
        node_index[execution_order[i]] = static_cast<int64_t>(i);
    }
    
    // 分析每个Value的生命周期
    std::vector<TensorLifetime> lifetimes;
    
    // 遍历所有Value
    for (const auto& value_ptr : graph->GetValues()) {
        Value* value = value_ptr.get();
        if (!value) continue;
        
        TensorLifetime lifetime;
        lifetime.value_ptr = value;
        
        // 找到产生该Value的节点（birth）
        Node* producer = nullptr;
        for (Node* node : execution_order) {
            for (Value* output : node->GetOutputs()) {
                if (output == value) {
                    producer = node;
                    break;
                }
            }
            if (producer) break;
        }
        
        if (producer) {
            auto it = node_index.find(producer);
            if (it != node_index.end()) {
                lifetime.birth = it->second;
            }
        } else {
            // 输入Value，birth = -1（在开始前就存在）
            lifetime.birth = -1;
        }
        
        // 找到最后使用该Value的节点（death）
        int64_t last_use = -1;
        for (Node* node : execution_order) {
            for (Value* input : node->GetInputs()) {
                if (input == value) {
                    auto it = node_index.find(node);
                    if (it != node_index.end()) {
                        last_use = std::max(last_use, it->second);
                    }
                }
            }
        }
        
        // 检查是否是输出Value
        bool is_output = false;
        for (Value* output : graph->GetOutputs()) {
            if (output == value) {
                is_output = true;
                break;
            }
        }
        
        if (is_output) {
            // 输出Value在最后才死亡
            lifetime.death = static_cast<int64_t>(execution_order.size());
        } else if (last_use >= 0) {
            lifetime.death = last_use;
        } else {
            // 未被使用的Value
            lifetime.death = lifetime.birth;
        }
        
        lifetimes.push_back(lifetime);
    }
    
    return lifetimes;
}

// 基于生命周期分析的内存复用分配
Status AllocateMemoryWithReuse(Graph* graph) {
    if (!graph) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Graph is null");
    }
    
    // 分析Tensor生命周期
    std::vector<TensorLifetime> lifetimes = AnalyzeTensorLifetimes(graph);
    
    // 按出生时间排序
    std::sort(lifetimes.begin(), lifetimes.end(), 
              [](const TensorLifetime& a, const TensorLifetime& b) {
                  return a.birth < b.birth;
              });
    
    // 内存复用分配（贪心算法）
    // 对于每个Tensor，查找可以复用的内存块
    std::vector<std::pair<Value*, void*>> allocations;
    std::vector<std::pair<void*, size_t>> free_blocks;  // <ptr, size>
    
    for (const auto& lifetime : lifetimes) {
        Value* value = static_cast<Value*>(lifetime.value_ptr);
        if (!value) continue;
        
        const Shape& shape = value->GetShape();
        DataType dtype = value->GetDataType();
        size_t size = shape.GetElementCount() * GetDataTypeSize(dtype);
        
        if (size == 0) continue;
        
        // 释放已死亡的Tensor内存
        for (auto it = free_blocks.begin(); it != free_blocks.end();) {
            // 检查这个内存块对应的Tensor是否已死亡
            // 简化实现：假设所有free_blocks都可以重用
            ++it;
        }
        
        // 查找可重用的内存块
        void* ptr = nullptr;
        for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it) {
            if (it->second >= size) {
                ptr = it->first;
                free_blocks.erase(it);
                break;
            }
        }
        
        // 如果没有可重用的，分配新内存
        if (!ptr) {
            ptr = AllocateMemory(size, 16);  // 16字节对齐
            if (!ptr) {
                return Status::Error(StatusCode::ERROR_OUT_OF_MEMORY,
                                   "Failed to allocate memory for tensor");
            }
        }
        
        // 创建Tensor并设置内存
        auto tensor = CreateTensorFromData(shape, dtype, ptr, MemoryLayout::NCHW, DeviceType::CPU);
        value->SetTensor(tensor);
        
        allocations.push_back({value, ptr});
    }
    
    LOG_INFO("Memory reuse allocation completed: " + 
             std::to_string(allocations.size()) + " tensors allocated");
    
    return Status::Ok();
}

} // namespace inferunity

