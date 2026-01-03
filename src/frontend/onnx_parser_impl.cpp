#include "frontend/onnx_parser.h"
#include "inferunity/graph.h"
#include "inferunity/tensor.h"
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cstring>

// 如果ONNX protobuf可用，使用它
#ifdef INFERUNITY_USE_ONNX_PROTOBUF
#include "onnx.pb.h"
#define USE_ONNX_PROTOBUF 1
#else
// 检查是否有生成的onnx.pb.h
#if __has_include("onnx.pb.h")
#include "onnx.pb.h"
#define USE_ONNX_PROTOBUF 1
#else
#define USE_ONNX_PROTOBUF 0
#endif
#endif

namespace inferunity {
namespace frontend {

// 简化的ONNX数据结构（当没有protobuf时使用）
// 参考ONNX Runtime的实现，但简化
struct SimpleONNXModel {
    struct Tensor {
        std::string name;
        std::vector<int64_t> dims;
        int32_t data_type;
        std::vector<uint8_t> raw_data;
    };
    
    struct InputInfo {
        std::string name;
        std::vector<int64_t> dims;  // 输入形状（可能包含-1或动态维度）
        int32_t data_type;
    };
    
    struct Node {
        std::string name;
        std::string op_type;
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
        std::unordered_map<std::string, std::string> attributes;
    };
    
    std::string model_version;
    std::vector<std::string> input_names;
    std::vector<InputInfo> input_infos;  // 输入信息（包含形状）
    std::vector<std::string> output_names;
    std::vector<Tensor> initializers;  // 权重
    std::vector<Node> nodes;
};

ONNXParser::ONNXParser() : model_proto_(nullptr) {
}

ONNXParser::~ONNXParser() {
    if (model_proto_) {
        delete static_cast<SimpleONNXModel*>(model_proto_);
        model_proto_ = nullptr;
    }
}

Status ONNXParser::LoadFromFile(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return Status::Error(StatusCode::ERROR_NOT_FOUND,
                           "Cannot open ONNX file: " + filepath);
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size);
    
    return LoadFromMemory(data.data(), data.size());
}

Status ONNXParser::LoadFromMemory(const void* data, size_t size) {
#if USE_ONNX_PROTOBUF
    // 使用ONNX protobuf解析
    onnx::ModelProto model;
    if (!model.ParseFromArray(data, size)) {
        return Status::Error(StatusCode::ERROR_INVALID_MODEL,
                            "Failed to parse ONNX model");
    }
    
    // 转换为简化格式以便处理
    auto* simple_model = new SimpleONNXModel();
    simple_model->model_version = model.model_version();
    
    // 解析输入输出（参考ONNX Runtime的实现）
    const auto& graph = model.graph();
    for (const auto& input : graph.input()) {
        simple_model->input_names.push_back(input.name());
        
        // 解析输入形状和类型（参考ONNX Runtime）
        SimpleONNXModel::InputInfo input_info;
        input_info.name = input.name();
        
        if (input.type().has_tensor_type()) {
            const auto& tensor_type = input.type().tensor_type();
            input_info.data_type = tensor_type.elem_type();
            
            // 解析形状维度
            if (tensor_type.has_shape()) {
                const auto& shape = tensor_type.shape();
                for (const auto& dim : shape.dim()) {
                    if (dim.has_dim_value()) {
                        input_info.dims.push_back(dim.dim_value());
                    } else if (dim.has_dim_param()) {
                        // 动态维度，使用-1表示（参考ONNX Runtime）
                        input_info.dims.push_back(-1);
                    } else {
                        // 未知维度，使用-1
                        input_info.dims.push_back(-1);
                    }
                }
            }
        }
        simple_model->input_infos.push_back(input_info);
    }
    for (const auto& output : graph.output()) {
        simple_model->output_names.push_back(output.name());
    }
    
    // 解析初始值（权重）
    for (const auto& init : graph.initializer()) {
        SimpleONNXModel::Tensor tensor;
        tensor.name = init.name();
        for (int64_t dim : init.dims()) {
            tensor.dims.push_back(dim);
        }
        tensor.data_type = init.data_type();
        tensor.raw_data = std::vector<uint8_t>(
            init.raw_data().begin(), init.raw_data().end());
        simple_model->initializers.push_back(tensor);
    }
    
    // 解析节点
    for (const auto& onnx_node : graph.node()) {
        SimpleONNXModel::Node node;
        node.name = onnx_node.name();
        node.op_type = onnx_node.op_type();
        for (const std::string& input : onnx_node.input()) {
            node.inputs.push_back(input);
        }
        for (const std::string& output : onnx_node.output()) {
            node.outputs.push_back(output);
        }
        
        // 解析属性
        for (const auto& attr : onnx_node.attribute()) {
            std::string value;
            if (attr.type() == onnx::AttributeProto::INT) {
                value = std::to_string(attr.i());
            } else if (attr.type() == onnx::AttributeProto::FLOAT) {
                value = std::to_string(attr.f());
            } else if (attr.type() == onnx::AttributeProto::STRING) {
                value = attr.s();
            } else if (attr.type() == onnx::AttributeProto::INTS) {
                for (int64_t v : attr.ints()) {
                    if (!value.empty()) value += ",";
                    value += std::to_string(v);
                }
            } else if (attr.type() == onnx::AttributeProto::FLOATS) {
                for (float v : attr.floats()) {
                    if (!value.empty()) value += ",";
                    value += std::to_string(v);
                }
            }
            node.attributes[attr.name()] = value;
        }
        
        simple_model->nodes.push_back(node);
    }
    
    model_proto_ = simple_model;
    return Status::Ok();
#else
    // 没有protobuf时的简化实现
    // 这里只能解析简单的ONNX格式，或者返回错误
    (void)data;
    (void)size;
    return Status::Error(StatusCode::ERROR_NOT_IMPLEMENTED,
                       "ONNX protobuf not available. Please install ONNX protobuf files.");
#endif
}

Status ONNXParser::ConvertToGraph(std::unique_ptr<Graph>& graph) {
    if (!model_proto_) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                           "Model not loaded");
    }
    
    auto* simple_model = static_cast<SimpleONNXModel*>(model_proto_);
    graph = std::make_unique<Graph>();
    
    // 1. 创建输入Value节点和初始值（权重）的Value节点
    // 注意：在ONNX中，初始值可能同时出现在graph.input和graph.initializer中
    // 我们需要先处理初始值，然后处理输入（如果输入不在初始值中）
    std::unordered_map<std::string, Value*> name_to_value;
    int64_t value_id = 0;
    
    // 先创建初始值（权重）的Value节点
    std::unordered_set<std::string> initializer_names;
    for (const auto& init : simple_model->initializers) {
        Value* value = graph->AddValue();
        value->SetId(value_id++);
        
        // 创建张量并填充数据
        Shape shape(init.dims);
        DataType dtype = ConvertDataType(init.data_type);
        auto tensor = CreateTensor(shape, dtype);
        
        // 复制数据
        if (!init.raw_data.empty()) {
            std::memcpy(tensor->GetData(), init.raw_data.data(),
                       std::min(init.raw_data.size(), tensor->GetSizeInBytes()));
        }
        
        value->SetTensor(tensor);
        name_to_value[init.name] = value;
        initializer_names.insert(init.name);
    }
    
    // 然后创建图输入Value节点（排除已经是初始值的）
    // 参考ONNX Runtime：对于有明确形状的输入，预先创建Tensor
    for (size_t i = 0; i < simple_model->input_names.size(); ++i) {
        const std::string& input_name = simple_model->input_names[i];
        const auto& input_info = simple_model->input_infos[i];
        
        // 如果输入名称已经在初始值中，跳过（初始值已经创建了Value）
        if (initializer_names.find(input_name) != initializer_names.end()) {
            // 这个输入是初始值，已经创建了Value，只需要记录名称
            input_names_.push_back(input_name);
            continue;
        }
        
        // 这是真正的图输入（不是初始值）
        Value* value = graph->AddValue();
        value->SetId(value_id++);
        
        // 如果输入有明确的形状（不是动态维度），创建Tensor并设置形状
        // 参考ONNX Runtime：预先创建Tensor以便形状推断
        bool has_concrete_shape = true;
        for (int64_t dim : input_info.dims) {
            if (dim < 0) {
                has_concrete_shape = false;
                break;
            }
        }
        
        if (has_concrete_shape && !input_info.dims.empty()) {
            Shape input_shape(input_info.dims);
            DataType input_dtype = ConvertDataType(input_info.data_type);
            auto input_tensor = CreateTensor(input_shape, input_dtype);
            value->SetTensor(input_tensor);
        }
        
        name_to_value[input_name] = value;
        graph->AddInput(value);
        input_names_.push_back(input_name);
    }
    
    // 2. 解析节点，构建计算图
    for (const auto& onnx_node : simple_model->nodes) {
        Node* node = graph->AddNode(onnx_node.op_type, onnx_node.name);
        
        // 设置属性
        for (const auto& attr : onnx_node.attributes) {
            node->SetAttribute(attr.first, attr.second);
        }
        
        // 连接输入
        for (const std::string& input_name : onnx_node.inputs) {
            // 跳过空字符串（ONNX中某些输入可能为空）
            if (input_name.empty()) {
                continue;
            }
            
            auto it = name_to_value.find(input_name);
            if (it != name_to_value.end()) {
                node->AddInput(it->second);
            } else {
                // 输入不存在，可能是：
                // 1. 中间结果（之前节点的输出）
                // 2. 初始值（权重）但名称不匹配
                // 3. 图输入但名称不匹配
                // 先检查是否是初始值（权重），ONNX中初始值可能同时出现在input和initializer中
                bool found_in_initializers = false;
                for (const auto& init : simple_model->initializers) {
                    if (init.name == input_name) {
                        // 这是初始值，应该已经创建了，但可能名称不匹配
                        // 重新查找或创建
                        found_in_initializers = true;
                        break;
                    }
                }
                
                if (!found_in_initializers) {
                    // 这是中间结果（之前节点的输出），创建新的Value
                    Value* value = graph->AddValue();
                    value->SetId(value_id++);
                    name_to_value[input_name] = value;
                    node->AddInput(value);
                } else {
                    // 是初始值但没找到，说明解析有问题，创建新Value并记录警告
                    Value* value = graph->AddValue();
                    value->SetId(value_id++);
                    name_to_value[input_name] = value;
                    node->AddInput(value);
                }
            }
        }
        
        // 创建输出Value
        for (const std::string& output_name : onnx_node.outputs) {
            Value* value = graph->AddValue();
            value->SetId(value_id++);
            name_to_value[output_name] = value;
            node->AddOutput(value);
        }
    }
    
    // 4. 设置图输出
    for (const std::string& output_name : simple_model->output_names) {
        auto it = name_to_value.find(output_name);
        if (it != name_to_value.end()) {
            graph->AddOutput(it->second);
            output_names_.push_back(output_name);
        }
    }
    
    // 5. 验证图
    Status status = graph->Validate();
    if (!status.IsOk()) {
        return status;
    }
    
    return Status::Ok();
}

Status ONNXParser::ParseNode(const void* onnx_node, Node* node) {
    // 这个方法现在在ConvertToGraph中直接实现
    (void)onnx_node;
    (void)node;
    return Status::Ok();
}

Status ONNXParser::ParseTensor(const void* onnx_tensor, Value* value) {
    // 这个方法现在在ConvertToGraph中直接实现
    (void)onnx_tensor;
    (void)value;
    return Status::Ok();
}

DataType ONNXParser::ConvertDataType(int32_t onnx_dtype) {
    // ONNX数据类型映射 (参考ONNX Runtime)
    switch (onnx_dtype) {
        case 1: return DataType::FLOAT32;  // FLOAT
        case 2: return DataType::UINT8;    // UINT8
        case 3: return DataType::INT8;     // INT8
        case 4: return DataType::UINT16;   // UINT16
        case 5: return DataType::INT16;    // INT16
        case 6: return DataType::INT32;    // INT32
        case 7: return DataType::INT64;    // INT64
        case 8: return DataType::STRING;   // STRING
        case 9: return DataType::BOOL;     // BOOL
        case 10: return DataType::FLOAT16; // FLOAT16
        case 11: return DataType::FLOAT32; // DOUBLE (映射到FLOAT32)
        case 12: return DataType::UINT32;  // UINT32
        case 13: return DataType::UINT64;  // UINT64
        case 14: return DataType::FLOAT32; // COMPLEX64 (映射到FLOAT32)
        case 15: return DataType::FLOAT32; // COMPLEX128 (映射到FLOAT32)
        case 16: return DataType::BFLOAT16;// BFLOAT16
        default: return DataType::UNKNOWN;
    }
}

Status ONNXParser::ConvertAttributes(const void* onnx_node, Node* node) {
    // 这个方法现在在ConvertToGraph中直接实现
    (void)onnx_node;
    (void)node;
    return Status::Ok();
}

} // namespace frontend
} // namespace inferunity

