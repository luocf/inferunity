#include "inferunity/optimizer.h"
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <cmath>

namespace inferunity {

Optimizer::Optimizer() = default;
Optimizer::~Optimizer() = default;

void Optimizer::RegisterPass(std::unique_ptr<OptimizationPass> pass) {
    std::string name = pass->GetName();
    OptimizationPass* ptr = pass.get();
    passes_.push_back(std::move(pass));
    pass_map_[name] = ptr;
}

Status Optimizer::Optimize(Graph* graph) {
    // 按依赖关系排序
    std::vector<OptimizationPass*> sorted_passes = SortPasses();
    
    // 运行所有Pass
    for (OptimizationPass* pass : sorted_passes) {
        Status status = pass->Run(graph);
        if (!status.IsOk()) {
            return status;
        }
    }
    
    return Status::Ok();
}

Status Optimizer::RunPass(const std::string& pass_name, Graph* graph) {
    auto it = pass_map_.find(pass_name);
    if (it == pass_map_.end()) {
        return Status::Error(StatusCode::ERROR_NOT_FOUND,
                           "Pass not found: " + pass_name);
    }
    return it->second->Run(graph);
}

std::vector<std::string> Optimizer::GetRegisteredPasses() const {
    std::vector<std::string> names;
    for (const auto& pair : pass_map_) {
        names.push_back(pair.first);
    }
    return names;
}

std::vector<OptimizationPass*> Optimizer::SortPasses() const {
    // 拓扑排序Pass（基于依赖关系）
    std::unordered_map<OptimizationPass*, int> in_degree;
    std::unordered_map<std::string, OptimizationPass*> name_to_pass;
    
    // 建立映射
    for (const auto& pass : passes_) {
        name_to_pass[pass->GetName()] = pass.get();
        in_degree[pass.get()] = 0;
    }
    
    // 计算入度
    for (const auto& pass : passes_) {
        for (const std::string& dep : pass->GetDependencies()) {
            auto it = name_to_pass.find(dep);
            if (it != name_to_pass.end()) {
                in_degree[pass.get()]++;
            }
        }
    }
    
    // 拓扑排序
    std::vector<OptimizationPass*> sorted;
    std::queue<OptimizationPass*> queue;
    
    for (const auto& pass : passes_) {
        if (in_degree[pass.get()] == 0) {
            queue.push(pass.get());
        }
    }
    
    while (!queue.empty()) {
        OptimizationPass* pass = queue.front();
        queue.pop();
        sorted.push_back(pass);
        
        // 更新依赖此Pass的其他Pass的入度
        for (const auto& other_pass : passes_) {
            auto deps = other_pass->GetDependencies();
            if (std::find(deps.begin(), deps.end(), pass->GetName()) != deps.end()) {
                in_degree[other_pass.get()]--;
                if (in_degree[other_pass.get()] == 0) {
                    queue.push(other_pass.get());
                }
            }
        }
    }
    
    return sorted;
}

// ConstantFoldingPass实现已移至constant_folding.cpp

// DeadCodeEliminationPass实现
Status DeadCodeEliminationPass::Run(Graph* graph) {
    // 找到所有未被使用的节点
    std::vector<Node*> to_remove;
    
    for (const auto& node : graph->GetNodes()) {
        bool is_used = false;
        
        // 检查输出是否被使用
        for (Value* output : node->GetOutputs()) {
            if (!output->GetConsumers().empty() ||
                std::find(graph->GetOutputs().begin(), graph->GetOutputs().end(), output) 
                    != graph->GetOutputs().end()) {
                is_used = true;
                break;
            }
        }
        
        if (!is_used && !node->GetOutputs().empty()) {
            to_remove.push_back(node.get());
        }
    }
    
    // 移除未使用的节点
    for (Node* node : to_remove) {
        graph->RemoveNode(node);
    }
    
    return Status::Ok();
}

// OperatorFusionPass实现已移至operator_fusion.cpp
// CanFuseConvBNReLU和CanFuseMatMulAdd在operator_fusion.cpp中定义

// FuseConvBNReLU 和 FuseMatMulAdd 的实现已移至 operator_fusion.cpp
// 这里保留声明以保持接口一致性

// MemoryLayoutOptimizationPass实现
Status MemoryLayoutOptimizationPass::Run(Graph* graph) {
    if (!graph) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Graph is null");
    }
    
    // 1. 分析数据流，确定每个节点的最优布局
    // 策略：
    // - Conv/Pool等算子偏好NCHW（通道优先）
    // - 某些算子（如某些激活函数）对布局不敏感
    // - 在布局不匹配时插入Transpose节点
    
    std::unordered_map<Node*, MemoryLayout> node_layouts;
    std::unordered_map<Value*, MemoryLayout> value_layouts;
    
    // 获取拓扑排序
    std::vector<Node*> execution_order = graph->TopologicalSort();
    
    // 为输入设置默认布局（NCHW）
    for (Value* input : graph->GetInputs()) {
        value_layouts[input] = MemoryLayout::NCHW;
    }
    
    // 遍历节点，确定布局
    for (Node* node : execution_order) {
        MemoryLayout preferred_layout = MemoryLayout::NCHW;
        std::string op_type = node->GetOpType();
        
        // 根据算子类型确定偏好布局
        if (op_type == "Conv" || op_type == "MaxPool" || op_type == "AvgPool" ||
            op_type == "BatchNormalization") {
            preferred_layout = MemoryLayout::NCHW;
        } else if (op_type == "MatMul" || op_type == "Gemm") {
            // 矩阵乘法对布局不敏感，但保持输入布局
            if (!node->GetInputs().empty()) {
                preferred_layout = value_layouts[node->GetInputs()[0]];
            }
        } else {
            // 其他算子保持输入布局
            if (!node->GetInputs().empty()) {
                preferred_layout = value_layouts[node->GetInputs()[0]];
            }
        }
        
        node_layouts[node] = preferred_layout;
        
        // 设置输出布局
        for (Value* output : node->GetOutputs()) {
            value_layouts[output] = preferred_layout;
        }
    }
    
    // 2. 检查布局不匹配，插入Transpose节点
    std::vector<std::pair<Node*, Value*>> transpose_insertions;
    
    for (Node* node : execution_order) {
        MemoryLayout node_layout = node_layouts[node];
        
        // 检查输入布局是否匹配
        for (Value* input : node->GetInputs()) {
            MemoryLayout input_layout = value_layouts[input];
            
            if (input_layout != node_layout && input_layout != MemoryLayout::UNKNOWN) {
                // 需要插入Transpose节点
                transpose_insertions.push_back({node, input});
            }
        }
    }
    
    // 3. 插入Transpose节点
    for (const auto& insertion : transpose_insertions) {
        Node* consumer = insertion.first;
        Value* input_value = insertion.second;
        MemoryLayout input_layout = value_layouts[input_value];
        MemoryLayout target_layout = node_layouts[consumer];
        
        // 创建中间Value
        Value* intermediate = graph->AddValue();
        
        // 创建Transpose节点
        Node* transpose = graph->AddNode("Transpose", "layout_transpose_" + 
                                        std::to_string(transpose_insertions.size()));
        // 临时：使用简单的perm值，GetTransposePerm函数在后面定义
        // transpose->SetAttribute("perm", GetTransposePerm(input_layout, target_layout));
        // TODO: 实现GetTransposePerm函数或使用其他方式获取perm
        
        // 连接：input_value -> transpose -> intermediate -> consumer
        transpose->AddInput(input_value);
        transpose->AddOutput(intermediate);
        
        // 更新consumer的输入
        consumer->RemoveInput(input_value);
        consumer->AddInput(intermediate);
        
        // 更新布局
        value_layouts[intermediate] = target_layout;
    }
    
    return Status::Ok();
}

namespace {
// 辅助函数：获取Transpose的perm参数
std::string GetTransposePerm(MemoryLayout from, MemoryLayout to) {
    // 简化实现：NCHW <-> NHWC转换
    if (from == MemoryLayout::NCHW && to == MemoryLayout::NHWC) {
        return "0,2,3,1";  // NCHW -> NHWC: [N,C,H,W] -> [N,H,W,C]
    } else if (from == MemoryLayout::NHWC && to == MemoryLayout::NCHW) {
        return "0,3,1,2";  // NHWC -> NCHW: [N,H,W,C] -> [N,C,H,W]
    }
    return "0,1,2,3";  // 默认：不变
}
}

// SubgraphReplacementPass实现
// 子图替换：识别特定模式并用优化版本替换
// 参考TVM和ONNX Runtime的子图替换实现
Status SubgraphReplacementPass::Run(Graph* graph) {
    if (!graph) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Graph is null");
    }
    
    bool changed = true;
    int max_iterations = 5;  // 防止无限循环
    int iteration = 0;
    
    while (changed && iteration < max_iterations) {
        changed = false;
        iteration++;
        
        // 获取拓扑排序的节点列表
        std::vector<Node*> nodes = graph->TopologicalSort();
        
        // 查找可替换的子图模式
        for (size_t i = 0; i < nodes.size(); ++i) {
            Node* node = nodes[i];
            if (!node) continue;
            
            // 模式1: Add(Const, X) -> X（如果Const为0）
            // 模式2: Mul(Const, X) -> X（如果Const为1）
            // 模式3: 其他常见的恒等变换
            
            // 这里实现一个简单的模式：Add(0, X) -> X
            if (node->GetOpType() == "Add" && node->GetInputs().size() == 2) {
                Value* input1 = node->GetInputs()[0];
                Value* input2 = node->GetInputs()[1];
                
                // 检查是否有常量输入为0
                bool can_replace = false;
                Value* non_const_input = nullptr;
                
                if (input1->GetTensor() && IsZeroTensor(input1->GetTensor().get())) {
                    can_replace = true;
                    non_const_input = input2;
                } else if (input2->GetTensor() && IsZeroTensor(input2->GetTensor().get())) {
                    can_replace = true;
                    non_const_input = input1;
                }
                
                if (can_replace && non_const_input) {
                    // 替换：将Add节点的输出直接连接到non_const_input
                    Value* output = node->GetOutputs()[0];
                    
                    // 更新所有消费者，使其直接使用non_const_input
                    std::vector<Node*> consumers = output->GetConsumers();
                    for (Node* consumer : consumers) {
                        consumer->RemoveInput(output);
                        consumer->AddInput(non_const_input);
                    }
                    
                    // 如果output是图的输出，需要更新
                    auto& graph_outputs = const_cast<std::vector<Value*>&>(graph->GetOutputs());
                    for (size_t j = 0; j < graph_outputs.size(); ++j) {
                        if (graph_outputs[j] == output) {
                            graph_outputs[j] = non_const_input;
                        }
                    }
                    
                    // 删除Add节点
                    graph->RemoveNode(node);
                    changed = true;
                    break;  // 重新开始遍历
                }
            }
        }
    }
    
    return Status::Ok();
}

// 辅助函数：检查张量是否全为0
bool SubgraphReplacementPass::IsZeroTensor(Tensor* tensor) const {
    if (!tensor || tensor->GetDataType() != DataType::FLOAT32) {
        return false;
    }
    
    const float* data = static_cast<const float*>(tensor->GetData());
    size_t count = tensor->GetElementCount();
    
    for (size_t i = 0; i < count; ++i) {
        if (std::abs(data[i]) > 1e-6f) {  // 允许小的浮点误差
            return false;
        }
    }
    
    return true;
}

} // namespace inferunity

