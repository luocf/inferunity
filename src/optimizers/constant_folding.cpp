// 常量折叠Pass实现
// 参考TVM的常量折叠实现

#include "inferunity/optimizer.h"
#include "inferunity/graph.h"
#include "inferunity/operator.h"
#include "inferunity/tensor.h"
#include <algorithm>
#include <unordered_set>

namespace inferunity {

Status ConstantFoldingPass::Run(Graph* graph) {
    if (!graph) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Graph is null");
    }
    
    // 找到所有常量节点（输入是初始值的节点）
    std::unordered_set<Node*> to_remove;
    std::unordered_map<Value*, Tensor*> constant_values;
    
    // 第一遍：识别常量值（来自initializer的Value）
    for (const auto& value_ptr : graph->GetValues()) {
        Value* value = value_ptr.get();
        if (value->GetTensor() && value->GetProducer() == nullptr) {
            // 这是一个初始值（常量）
            constant_values[value] = value->GetTensor().get();
        }
    }
    
    // 第二遍：找到可以折叠的节点
    // 如果一个节点的所有输入都是常量，且算子支持常量折叠，则可以折叠
    for (const auto& node_ptr : graph->GetNodes()) {
        Node* node = node_ptr.get();
        
        // 检查所有输入是否都是常量
        bool all_constants = true;
        std::vector<Tensor*> constant_inputs;
        
        for (Value* input : node->GetInputs()) {
            auto it = constant_values.find(input);
            if (it != constant_values.end()) {
                constant_inputs.push_back(it->second);
            } else {
                all_constants = false;
                break;
            }
        }
        
        if (all_constants && !constant_inputs.empty()) {
            // 尝试执行常量折叠
            // 创建算子并执行
            auto op = OperatorRegistry::Instance().Create(node->GetOpType());
            if (op) {
                // 推断输出形状
                std::vector<Shape> output_shapes;
                Status status = op->InferOutputShape(constant_inputs, output_shapes);
                if (status.IsOk() && !output_shapes.empty()) {
                    // 创建输出张量
                    auto output_tensor = CreateTensor(output_shapes[0], 
                                                     constant_inputs[0]->GetDataType());
                    std::vector<Tensor*> outputs = {output_tensor.get()};
                    
                    // 执行计算
                    ExecutionContext ctx;
                    status = op->Execute(constant_inputs, outputs, &ctx);
                    
                    if (status.IsOk()) {
                        // 用常量值替换输出Value
                        if (!node->GetOutputs().empty()) {
                            Value* output_value = node->GetOutputs()[0];
                            output_value->SetTensor(output_tensor);
                            constant_values[output_value] = output_tensor.get();
                            
                            // 标记节点为可删除
                            to_remove.insert(node);
                        }
                    }
                }
            }
        }
    }
    
    // 删除已折叠的节点
    for (Node* node : to_remove) {
        graph->RemoveNode(node);
    }
    
    return Status::Ok();
}

} // namespace inferunity

