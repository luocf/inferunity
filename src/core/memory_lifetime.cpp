// 张量生命周期分析和内存复用
// 参考NCNN的内存管理实现

#include "inferunity/memory.h"
#include "inferunity/graph.h"
#include "inferunity/tensor.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>

namespace inferunity {

// 张量生命周期分析器（参考NCNN的BlobAllocator）
class TensorLifetimeAnalyzer {
public:
    // 分析图中所有张量的生命周期
    static std::vector<TensorLifetime> AnalyzeLifetimes(const Graph* graph) {
        std::vector<TensorLifetime> lifetimes;
        
        if (!graph) {
            return lifetimes;
        }
        
        // 获取拓扑排序的节点
        std::vector<Node*> nodes = graph->TopologicalSort();
        
        // 记录每个Value的出生和死亡时间
        std::unordered_map<Value*, TensorLifetime> value_lifetimes;
        
        // 第一遍：记录出生时间
        for (size_t i = 0; i < nodes.size(); ++i) {
            Node* node = nodes[i];
            for (Value* output : node->GetOutputs()) {
                if (value_lifetimes.find(output) == value_lifetimes.end()) {
                    value_lifetimes[output] = TensorLifetime{static_cast<int64_t>(i), -1, output};
                }
            }
        }
        
        // 第二遍：记录死亡时间（最后使用时间）
        for (size_t i = 0; i < nodes.size(); ++i) {
            Node* node = nodes[i];
            // 输入Value的死亡时间至少是当前节点
            for (Value* input : node->GetInputs()) {
                auto it = value_lifetimes.find(input);
                if (it != value_lifetimes.end()) {
                    it->second.death = std::max(it->second.death, static_cast<int64_t>(i));
                }
            }
        }
        
        // 输出Value永远不会死亡（在推理期间）
        for (Value* output : graph->GetOutputs()) {
            auto it = value_lifetimes.find(output);
            if (it != value_lifetimes.end()) {
                it->second.death = nodes.size();  // 设置为最后
            }
        }
        
        // 转换为向量
        for (auto& pair : value_lifetimes) {
            lifetimes.push_back(pair.second);
        }
        
        return lifetimes;
    }
    
    // 查找可以复用内存的张量对
    static std::vector<std::pair<Value*, Value*>> FindReusablePairs(const Graph* graph) {
        std::vector<std::pair<Value*, Value*>> reusable_pairs;
        
        auto lifetimes = AnalyzeLifetimes(graph);
        
        // 对生命周期按出生时间排序
        std::sort(lifetimes.begin(), lifetimes.end(),
                 [](const TensorLifetime& a, const TensorLifetime& b) {
                     return a.birth < b.birth;
                 });
        
        // 查找不重叠的生命周期
        for (size_t i = 0; i < lifetimes.size(); ++i) {
            for (size_t j = i + 1; j < lifetimes.size(); ++j) {
                const TensorLifetime& a = lifetimes[i];
                const TensorLifetime& b = lifetimes[j];
                
                Value* value_a = static_cast<Value*>(a.value_ptr);
                Value* value_b = static_cast<Value*>(b.value_ptr);
                
                // 如果a死亡后b才出生，或者b死亡后a才出生，可以复用
                if (a.death <= b.birth || b.death <= a.birth) {
                    // 检查形状是否兼容
                    if (value_a->GetShape().GetElementCount() == 
                        value_b->GetShape().GetElementCount()) {
                        reusable_pairs.push_back({value_a, value_b});
                    }
                }
            }
        }
        
        return reusable_pairs;
    }
};

// 内存复用管理器（参考NCNN的BlobAllocator）
class MemoryReuseManager {
public:
    // 为图分配内存，考虑复用
    static Status AllocateWithReuse(Graph* graph) {
        if (!graph) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Graph is null");
        }
        
        // 分析生命周期
        auto lifetimes = TensorLifetimeAnalyzer::AnalyzeLifetimes(graph);
        auto reusable_pairs = TensorLifetimeAnalyzer::FindReusablePairs(graph);
        
        // 创建内存复用映射
        std::unordered_map<Value*, Value*> reuse_map;
        for (const auto& pair : reusable_pairs) {
            // 选择较小的Value复用较大的Value的内存
            size_t size_a = pair.first->GetShape().GetElementCount() * sizeof(float);
            size_t size_b = pair.second->GetShape().GetElementCount() * sizeof(float);
            
            if (size_a >= size_b) {
                reuse_map[pair.second] = pair.first;
            } else {
                reuse_map[pair.first] = pair.second;
            }
        }
        
        // 分配内存（考虑复用）
        std::unordered_set<Value*> allocated;
        for (const auto& lifetime : lifetimes) {
            Value* value = static_cast<Value*>(lifetime.value_ptr);
            if (allocated.find(value) != allocated.end()) {
                continue;
            }
            
            // 检查是否可以复用
            auto it = reuse_map.find(value);
            if (it != reuse_map.end()) {
                // 复用其他Value的内存
                Value* source = it->second;
                if (source->GetTensor() && source->GetTensor()->GetData()) {
                    // 复用源Tensor的内存（共享）
                    // 注意：这里简化处理，实际应该创建共享内存的Tensor视图
                    const Shape& shape = value->GetShape();
                    DataType dtype = value->GetDataType();
                    // 创建新Tensor但复用内存（需要扩展Tensor接口支持共享内存）
                    // 暂时使用正常分配
                    auto tensor = CreateTensor(shape, dtype, DeviceType::CPU);
                    value->SetTensor(tensor);
                }
            } else {
                // 正常分配
                if (!value->GetTensor()) {
                    const Shape& shape = value->GetShape();
                    DataType dtype = value->GetDataType();
                    auto tensor = CreateTensor(shape, dtype, DeviceType::CPU);
                    value->SetTensor(tensor);
                }
            }
            
            allocated.insert(value);
        }
        
        return Status::Ok();
    }
};

// 全局函数
std::vector<TensorLifetime> AnalyzeTensorLifetimes(const Graph* graph) {
    return TensorLifetimeAnalyzer::AnalyzeLifetimes(graph);
}

Status AllocateMemoryWithReuse(Graph* graph) {
    return MemoryReuseManager::AllocateWithReuse(graph);
}

} // namespace inferunity

