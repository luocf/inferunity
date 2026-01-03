// 算子融合Pass实现
// 参考TVM的算子融合实现

#include "inferunity/optimizer.h"
#include "inferunity/graph.h"
#include "inferunity/tensor.h"
#include <algorithm>
#include <unordered_set>

namespace inferunity {

Status OperatorFusionPass::Run(Graph* graph) {
    if (!graph) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Graph is null");
    }
    
    bool changed = true;
    int max_iterations = 10;  // 防止无限循环
    int iteration = 0;
    
    while (changed && iteration < max_iterations) {
        changed = false;
        iteration++;
        
        // 获取拓扑排序的节点列表
        std::vector<Node*> nodes = graph->TopologicalSort();
        
        // 查找融合模式
        for (size_t i = 0; i < nodes.size(); ++i) {
            Node* node1 = nodes[i];
            if (!node1 || node1->GetOutputs().empty()) continue;
            
            // 检查Conv+BN+ReLU模式
            if (node1->GetOpType() == "Conv") {
                Value* output1 = node1->GetOutputs()[0];
                if (!output1->GetConsumers().empty()) {
                    Node* node2 = output1->GetConsumers()[0];
                    if (node2 && node2->GetOpType() == "BatchNormalization" &&
                        !node2->GetOutputs().empty()) {
                        Value* output2 = node2->GetOutputs()[0];
                        if (!output2->GetConsumers().empty()) {
                            Node* node3 = output2->GetConsumers()[0];
                            if (node3 && node3->GetOpType() == "Relu") {
                                // 找到Conv+BN+ReLU模式，执行融合
                                if (CanFuseConvBNReLU(node1, node2, node3)) {
                                    Status status = FuseConvBNReLU(graph, node1, node2, node3);
                                    if (status.IsOk()) {
                                        changed = true;
                                        break;  // 重新开始遍历
                                    }
                                }
                            }
                        }
                    }
                    // 检查Conv+ReLU模式（简化融合）
                    else if (node2 && node2->GetOpType() == "Relu") {
                        if (CanFuseConvReLU(node1, node2)) {
                            Status status = FuseConvReLU(graph, node1, node2);
                            if (status.IsOk()) {
                                changed = true;
                                break;
                            }
                        }
                    }
                }
            }
            
            // 检查BN+ReLU模式
            if (node1->GetOpType() == "BatchNormalization") {
                Value* output1 = node1->GetOutputs()[0];
                if (!output1->GetConsumers().empty()) {
                    Node* node2 = output1->GetConsumers()[0];
                    if (node2 && node2->GetOpType() == "Relu") {
                        if (CanFuseBNReLU(node1, node2)) {
                            Status status = FuseBNReLU(graph, node1, node2);
                            if (status.IsOk()) {
                                changed = true;
                                break;
                            }
                        }
                    }
                }
            }
            
            // 检查MatMul+Add模式
            if (node1->GetOpType() == "MatMul") {
                Value* output1 = node1->GetOutputs()[0];
                if (!output1->GetConsumers().empty()) {
                    Node* node2 = output1->GetConsumers()[0];
                    if (node2 && node2->GetOpType() == "Add") {
                        // 找到MatMul+Add模式，执行融合
                        if (CanFuseMatMulAdd(node1, node2)) {
                            Status status = FuseMatMulAdd(graph, node1, node2);
                            if (status.IsOk()) {
                                changed = true;
                                break;  // 重新开始遍历
                            }
                        }
                    }
                }
            }
        }
    }
    
    return Status::Ok();
}

bool OperatorFusionPass::CanFuseConvBNReLU(Node* conv, Node* bn, Node* relu) const {
    if (!conv || !bn || !relu) return false;
    if (conv->GetOpType() != "Conv") return false;
    if (bn->GetOpType() != "BatchNormalization") return false;
    if (relu->GetOpType() != "Relu") return false;
    
    // 检查数据流连接
    if (conv->GetOutputs().empty() || bn->GetInputs().empty()) return false;
    if (conv->GetOutputs()[0] != bn->GetInputs()[0]) return false;
    if (bn->GetOutputs().empty() || relu->GetInputs().empty()) return false;
    if (bn->GetOutputs()[0] != relu->GetInputs()[0]) return false;
    
    return true;
}

bool OperatorFusionPass::CanFuseMatMulAdd(Node* matmul, Node* add) const {
    if (!matmul || !add) return false;
    if (matmul->GetOpType() != "MatMul") return false;
    if (add->GetOpType() != "Add") return false;
    
    // 检查数据流连接
    if (matmul->GetOutputs().empty() || add->GetInputs().size() < 2) return false;
    if (matmul->GetOutputs()[0] != add->GetInputs()[0] && 
        matmul->GetOutputs()[0] != add->GetInputs()[1]) return false;
    
    return true;
}

bool OperatorFusionPass::CanFuseConvReLU(Node* conv, Node* relu) const {
    if (!conv || !relu) return false;
    if (conv->GetOpType() != "Conv") return false;
    if (relu->GetOpType() != "Relu") return false;
    
    // 检查数据流连接
    if (conv->GetOutputs().empty() || relu->GetInputs().empty()) return false;
    if (conv->GetOutputs()[0] != relu->GetInputs()[0]) return false;
    
    return true;
}

bool OperatorFusionPass::CanFuseBNReLU(Node* bn, Node* relu) const {
    if (!bn || !relu) return false;
    if (bn->GetOpType() != "BatchNormalization") return false;
    if (relu->GetOpType() != "Relu") return false;
    
    // 检查数据流连接
    if (bn->GetOutputs().empty() || relu->GetInputs().empty()) return false;
    if (bn->GetOutputs()[0] != relu->GetInputs()[0]) return false;
    
    return true;
}

Status OperatorFusionPass::FuseConvBNReLU(Graph* graph, Node* conv, Node* bn, Node* relu) {
    if (!graph || !conv || !bn || !relu) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid nodes");
    }
    
    // 1. 创建融合节点
    std::string fused_name = conv->GetName() + "_" + bn->GetName() + "_" + relu->GetName() + "_fused";
    Node* fused_node = graph->AddNode("FusedConvBNReLU", fused_name);
    
    // 2. 合并属性（从Conv节点复制）
    for (const auto& attr : conv->GetAttributes()) {
        fused_node->SetAttribute(attr.first, attr.second);
    }
    
    // 3. 连接输入
    // Conv的输入：input, weight, bias(可选)
    // BN的输入：conv_output, scale, B, mean, var
    // 融合后的输入：input, weight, bias(可选), scale, B, mean, var
    for (Value* input : conv->GetInputs()) {
        fused_node->AddInput(input);
    }
    // 添加BN的输入（跳过第一个，因为它是conv的输出）
    const auto& bn_inputs = bn->GetInputs();
    for (size_t i = 1; i < bn_inputs.size(); ++i) {
        fused_node->AddInput(bn_inputs[i]);
    }
    
    // 4. 连接输出（使用ReLU的输出）
    Value* relu_output = relu->GetOutputs()[0];
    fused_node->AddOutput(relu_output);
    
    // 5. 更新输出值的生产者
    relu_output->SetProducer(fused_node);
    
    // 6. 删除原节点（按逆序删除，避免依赖问题）
    graph->RemoveNode(relu);
    graph->RemoveNode(bn);
    graph->RemoveNode(conv);
    
    return Status::Ok();
}

Status OperatorFusionPass::FuseMatMulAdd(Graph* graph, Node* matmul, Node* add) {
    if (!graph || !matmul || !add) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid nodes");
    }
    
    // 1. 创建融合节点
    std::string fused_name = matmul->GetName() + "_" + add->GetName() + "_fused";
    Node* fused_node = graph->AddNode("FusedMatMulAdd", fused_name);
    
    // 2. 合并属性（从MatMul节点复制）
    for (const auto& attr : matmul->GetAttributes()) {
        fused_node->SetAttribute(attr.first, attr.second);
    }
    
    // 3. 连接输入
    // MatMul的输入：A, B
    // Add的输入：matmul_output, bias
    // 融合后的输入：A, B, bias
    for (Value* input : matmul->GetInputs()) {
        fused_node->AddInput(input);
    }
    // 找到Add的另一个输入（bias）
    const auto& add_inputs = add->GetInputs();
    Value* matmul_output = matmul->GetOutputs()[0];
    for (Value* input : add_inputs) {
        if (input != matmul_output) {
            fused_node->AddInput(input);  // 这是bias
            break;
        }
    }
    
    // 4. 连接输出（使用Add的输出）
    Value* add_output = add->GetOutputs()[0];
    fused_node->AddOutput(add_output);
    
    // 5. 更新输出值的生产者
    add_output->SetProducer(fused_node);
    
    // 6. 删除原节点
    graph->RemoveNode(add);
    graph->RemoveNode(matmul);
    
    return Status::Ok();
}

Status OperatorFusionPass::FuseConvReLU(Graph* graph, Node* conv, Node* relu) {
    if (!graph || !conv || !relu) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid nodes");
    }
    
    // 创建融合节点（使用FusedConvBNReLU，但BN参数设为单位值）
    std::string fused_name = conv->GetName() + "_" + relu->GetName() + "_fused";
    Node* fused_node = graph->AddNode("FusedConvBNReLU", fused_name);
    
    // 合并属性
    for (const auto& attr : conv->GetAttributes()) {
        fused_node->SetAttribute(attr.first, attr.second);
    }
    
    // 连接输入（Conv的输入 + 单位BN参数）
    for (Value* input : conv->GetInputs()) {
        fused_node->AddInput(input);
    }
    
    // 创建单位BN参数值（scale=1, B=0, mean=0, var=1）
    // 获取输出通道数（从Conv的权重形状推断）
    Value* weight_value = conv->GetInputs().size() > 1 ? conv->GetInputs()[1] : nullptr;
    int64_t channels = 1;
    if (weight_value && weight_value->GetTensor()) {
        const Shape& weight_shape = weight_value->GetShape();
        if (weight_shape.dims.size() >= 1) {
            channels = weight_shape.dims[0];  // 输出通道数
        }
    }
    
    // 创建单位BN参数：scale=1, B=0, mean=0, var=1
    Shape param_shape({channels});
    auto scale_tensor = CreateTensor(param_shape, DataType::FLOAT32, DeviceType::CPU);
    auto bias_tensor = CreateTensor(param_shape, DataType::FLOAT32, DeviceType::CPU);
    auto mean_tensor = CreateTensor(param_shape, DataType::FLOAT32, DeviceType::CPU);
    auto var_tensor = CreateTensor(param_shape, DataType::FLOAT32, DeviceType::CPU);
    
    // 填充单位值
    scale_tensor->FillValue(1.0f);  // scale = 1
    bias_tensor->FillValue(0.0f);   // B = 0
    mean_tensor->FillValue(0.0f);   // mean = 0
    var_tensor->FillValue(1.0f);    // var = 1
    
    // 创建Value并添加到图中
    Value* scale_value = graph->AddValue();
    Value* bias_value = graph->AddValue();
    Value* mean_value = graph->AddValue();
    Value* var_value = graph->AddValue();
    
    scale_value->SetTensor(scale_tensor);
    bias_value->SetTensor(bias_tensor);
    mean_value->SetTensor(mean_tensor);
    var_value->SetTensor(var_tensor);
    
    // 添加到融合节点的输入
    fused_node->AddInput(scale_value);
    fused_node->AddInput(bias_value);
    fused_node->AddInput(mean_value);
    fused_node->AddInput(var_value);
    
    // 连接输出
    Value* relu_output = relu->GetOutputs()[0];
    fused_node->AddOutput(relu_output);
    relu_output->SetProducer(fused_node);
    
    // 删除原节点
    graph->RemoveNode(relu);
    graph->RemoveNode(conv);
    
    return Status::Ok();
}

Status OperatorFusionPass::FuseBNReLU(Graph* graph, Node* bn, Node* relu) {
    if (!graph || !bn || !relu) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid nodes");
    }
    
    // 创建融合节点（使用FusedConvBNReLU，但Conv部分为空）
    // 或者创建专门的FusedBNReLU算子
    // 这里简化：直接合并到BN节点，添加ReLU属性
    std::string fused_name = bn->GetName() + "_" + relu->GetName() + "_fused";
    Node* fused_node = graph->AddNode("BatchNormalization", fused_name);  // 使用BN，但添加fused属性
    
    // 合并属性
    for (const auto& attr : bn->GetAttributes()) {
        fused_node->SetAttribute(attr.first, attr.second);
    }
    fused_node->SetAttribute("fused_relu", "true");
    
    // 连接输入
    for (Value* input : bn->GetInputs()) {
        fused_node->AddInput(input);
    }
    
    // 连接输出
    Value* relu_output = relu->GetOutputs()[0];
    fused_node->AddOutput(relu_output);
    relu_output->SetProducer(fused_node);
    
    // 删除原节点
    graph->RemoveNode(relu);
    graph->RemoveNode(bn);
    
    return Status::Ok();
}

} // namespace inferunity

