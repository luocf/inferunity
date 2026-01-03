// 形状推断系统实现
// 参考ONNX Runtime的形状推断实现

#include "inferunity/graph.h"
#include "inferunity/operator.h"
#include "inferunity/tensor.h"
#include <queue>
#include <unordered_set>
#include <unordered_map>

namespace inferunity {

// 形状推断器（参考ONNX Runtime的ShapeInference类）
class ShapeInference {
public:
    static Status InferGraph(Graph* graph) {
        if (!graph) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Graph is null");
        }
        
        // 拓扑排序
        std::vector<Node*> nodes = graph->TopologicalSort();
        
        // 为每个节点推断输出形状
        for (Node* node : nodes) {
            Status status = InferNode(node);
            if (!status.IsOk()) {
                return status;
            }
        }
        
        return Status::Ok();
    }
    
private:
    static Status InferNode(Node* node) {
        if (!node) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Node is null");
        }
        
        // 准备输入张量（用于形状推断）
        std::vector<Tensor*> input_tensors;
        for (Value* input : node->GetInputs()) {
            if (input->GetTensor()) {
                input_tensors.push_back(input->GetTensor().get());
            } else {
                // 如果没有张量，创建一个虚拟张量用于形状推断
                // 使用Value的形状信息
                const Shape& shape = input->GetShape();
                if (!shape.dims.empty()) {
                    // 创建临时张量用于形状推断
                    auto temp_tensor = CreateTensor(shape, DataType::FLOAT32, DeviceType::CPU);
                    input_tensors.push_back(temp_tensor.get());
                }
            }
        }
        
        if (input_tensors.empty()) {
            return Status::Ok();  // 没有输入，跳过
        }
        
        // 创建算子并推断输出形状
        auto op = OperatorRegistry::Instance().Create(node->GetOpType());
        if (!op) {
            // 算子未注册，跳过形状推断
            return Status::Ok();
        }
        
        std::vector<Shape> output_shapes;
        Status status = op->InferOutputShape(input_tensors, output_shapes);
        if (!status.IsOk()) {
            return status;
        }
        
        // 设置输出Value的形状
        const auto& outputs = node->GetOutputs();
        for (size_t i = 0; i < outputs.size() && i < output_shapes.size(); ++i) {
            Value* output = outputs[i];
            // 更新Value的形状信息（如果Value没有张量）
            if (!output->GetTensor()) {
                // 可以通过设置Shape到Value（需要扩展Value接口）
                // 这里简化：创建临时张量
                auto tensor = CreateTensor(output_shapes[i], DataType::FLOAT32, DeviceType::CPU);
                output->SetTensor(tensor);
            }
        }
        
        return Status::Ok();
    }
};

// 全局形状推断函数
Status InferShapes(Graph* graph) {
    return ShapeInference::InferGraph(graph);
}

} // namespace inferunity

