#include "inferunity/graph.h"
#include "inferunity/operator.h"
#include "inferunity/tensor.h"
#include <algorithm>
#include <queue>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>

namespace inferunity {

// Value实现
const Shape& Value::GetShape() const {
    static Shape empty_shape;
    return tensor_ ? tensor_->GetShape() : empty_shape;
}

DataType Value::GetDataType() const {
    return tensor_ ? tensor_->GetDataType() : DataType::UNKNOWN;
}

void Value::AddConsumer(Node* node) {
    if (std::find(consumers_.begin(), consumers_.end(), node) == consumers_.end()) {
        consumers_.push_back(node);
    }
}

void Value::RemoveConsumer(Node* node) {
    consumers_.erase(
        std::remove(consumers_.begin(), consumers_.end(), node),
        consumers_.end()
    );
}

// Node实现
void Node::AddInput(Value* value) {
    if (std::find(inputs_.begin(), inputs_.end(), value) == inputs_.end()) {
        inputs_.push_back(value);
        value->AddConsumer(this);
    }
}

void Node::AddOutput(Value* value) {
    if (std::find(outputs_.begin(), outputs_.end(), value) == outputs_.end()) {
        outputs_.push_back(value);
        value->SetProducer(this);
    }
}

void Node::RemoveInput(Value* value) {
    inputs_.erase(
        std::remove(inputs_.begin(), inputs_.end(), value),
        inputs_.end()
    );
    value->RemoveConsumer(this);
}

void Node::RemoveOutput(Value* value) {
    outputs_.erase(
        std::remove(outputs_.begin(), outputs_.end(), value),
        outputs_.end()
    );
    if (value->GetProducer() == this) {
        value->SetProducer(nullptr);
    }
}

void Node::SetAttribute(const std::string& key, const std::string& value) {
    attributes_[key] = value;
}

std::string Node::GetAttribute(const std::string& key, const std::string& default_value) const {
    auto it = attributes_.find(key);
    return it != attributes_.end() ? it->second : default_value;
}

bool Node::HasAttribute(const std::string& key) const {
    return attributes_.find(key) != attributes_.end();
}

// Graph实现
Graph::Graph() : next_node_id_(0), next_value_id_(0) {
}

Graph::~Graph() = default;

Node* Graph::AddNode(const std::string& op_type, const std::string& name) {
    auto node = std::make_unique<Node>(GenerateNodeId(), op_type, name);
    Node* ptr = node.get();
    nodes_.push_back(std::move(node));
    return ptr;
}

Node* Graph::GetNode(int64_t id) const {
    for (const auto& node : nodes_) {
        if (node->GetId() == id) {
            return node.get();
        }
    }
    return nullptr;
}

Node* Graph::GetNodeByName(const std::string& name) const {
    for (const auto& node : nodes_) {
        if (node->GetName() == name) {
            return node.get();
        }
    }
    return nullptr;
}

void Graph::RemoveNode(Node* node) {
    // 断开连接
    for (Value* input : node->GetInputs()) {
        input->RemoveConsumer(node);
    }
    for (Value* output : node->GetOutputs()) {
        if (output->GetProducer() == node) {
            output->SetProducer(nullptr);
        }
    }
    
    // 移除节点
    nodes_.erase(
        std::remove_if(nodes_.begin(), nodes_.end(),
                      [node](const std::unique_ptr<Node>& n) { return n.get() == node; }),
        nodes_.end()
    );
}

Value* Graph::AddValue() {
    auto value = std::make_unique<Value>(GenerateValueId());
    Value* ptr = value.get();
    values_.push_back(std::move(value));
    return ptr;
}

Value* Graph::GetValue(int64_t id) const {
    for (const auto& value : values_) {
        if (value->GetId() == id) {
            return value.get();
        }
    }
    return nullptr;
}

void Graph::RemoveValue(Value* value) {
    // 断开连接
    if (Node* producer = value->GetProducer()) {
        producer->RemoveOutput(value);
    }
    for (Node* consumer : value->GetConsumers()) {
        consumer->RemoveInput(value);
    }
    
    // 从输入输出列表中移除
    inputs_.erase(std::remove(inputs_.begin(), inputs_.end(), value), inputs_.end());
    outputs_.erase(std::remove(outputs_.begin(), outputs_.end(), value), outputs_.end());
    
    // 移除值
    values_.erase(
        std::remove_if(values_.begin(), values_.end(),
                      [value](const std::unique_ptr<Value>& v) { return v.get() == value; }),
        values_.end()
    );
}

void Graph::AddInput(Value* value) {
    if (std::find(inputs_.begin(), inputs_.end(), value) == inputs_.end()) {
        inputs_.push_back(value);
    }
}

void Graph::AddOutput(Value* value) {
    if (std::find(outputs_.begin(), outputs_.end(), value) == outputs_.end()) {
        outputs_.push_back(value);
    }
}

void Graph::Clear() {
    nodes_.clear();
    values_.clear();
    inputs_.clear();
    outputs_.clear();
    next_node_id_ = 0;
    next_value_id_ = 0;
}

Graph Graph::Clone() const {
    Graph new_graph;
    
    // 创建Value映射（旧Value -> 新Value）
    std::unordered_map<const Value*, Value*> value_map;
    
    // 1. 复制所有Value
    for (const auto& value : values_) {
        Value* new_value = new_graph.AddValue();
        new_value->SetId(value->GetId());
        
        // 复制形状和数据类型信息
        // 注意：Tensor不复制，因为这是图结构的拷贝
        value_map[value.get()] = new_value;
    }
    
    // 2. 复制所有Node
    std::unordered_map<const Node*, Node*> node_map;
    for (const auto& node : nodes_) {
        Node* new_node = new_graph.AddNode(node->GetOpType(), node->GetName());
        new_node->SetId(node->GetId());
        
        // 复制属性
        for (const auto& attr : node->GetAttributes()) {
            new_node->SetAttribute(attr.first, attr.second);
        }
        
        // 复制设备类型
        new_node->SetDevice(node->GetDevice());
        
        node_map[node.get()] = new_node;
    }
    
    // 3. 连接输入输出
    for (const auto& node : nodes_) {
        Node* new_node = node_map[node.get()];
        
        // 连接输入
        for (Value* input : node->GetInputs()) {
            Value* new_input = value_map[input];
            new_node->AddInput(new_input);
        }
        
        // 连接输出
        for (Value* output : node->GetOutputs()) {
            Value* new_output = value_map[output];
            new_node->AddOutput(new_output);
        }
    }
    
    // 4. 设置图的输入输出
    for (Value* input : inputs_) {
        new_graph.AddInput(value_map[input]);
    }
    for (Value* output : outputs_) {
        new_graph.AddOutput(value_map[output]);
    }
    
    return new_graph;
}

std::vector<Node*> Graph::TopologicalSort() const {
    std::vector<Node*> sorted;
    std::unordered_map<Node*, int> in_degree;
    
    // 计算入度
    for (const auto& node : nodes_) {
        in_degree[node.get()] = 0;
    }
    for (const auto& node : nodes_) {
        for (Value* input : node->GetInputs()) {
            if (Node* producer = input->GetProducer()) {
                in_degree[node.get()]++;
            }
        }
    }
    
    // 拓扑排序
    std::queue<Node*> queue;
    for (const auto& node : nodes_) {
        if (in_degree[node.get()] == 0) {
            queue.push(node.get());
        }
    }
    
    while (!queue.empty()) {
        Node* node = queue.front();
        queue.pop();
        sorted.push_back(node);
        
        for (Value* output : node->GetOutputs()) {
            for (Node* consumer : output->GetConsumers()) {
                in_degree[consumer]--;
                if (in_degree[consumer] == 0) {
                    queue.push(consumer);
                }
            }
        }
    }
    
    return sorted;
}

Status Graph::Validate() const {
    // 1. 检查所有输入都有生产者或是图输入或是初始值（权重）
    for (const auto& node : nodes_) {
        for (Value* input : node->GetInputs()) {
            if (!input->GetProducer()) {
                // 检查是否是图输入
                bool is_graph_input = std::find(inputs_.begin(), inputs_.end(), input) != inputs_.end();
                // 检查是否是初始值（权重）- 有Tensor但没有生产者
                bool is_initializer = (input->GetTensor() != nullptr);
                
                if (!is_graph_input && !is_initializer) {
                    return Status::Error(StatusCode::ERROR_INVALID_MODEL,
                                       "Input value has no producer and is not a graph input");
                }
            }
        }
    }
    
    // 2. 检查所有输出都有消费者或作为图输出
    for (const auto& value : values_) {
        if (value->GetProducer()) {
            // 这是节点的输出
            bool is_graph_output = std::find(outputs_.begin(), outputs_.end(), value.get()) != outputs_.end();
            bool has_consumers = !value->GetConsumers().empty();
            
            if (!is_graph_output && !has_consumers) {
                // 死代码（未使用的输出），这是警告而不是错误
                // 可以继续执行，但建议优化
            }
        }
    }
    
    // 3. 检查图的输入输出不为空
    if (inputs_.empty()) {
        return Status::Error(StatusCode::ERROR_INVALID_MODEL, "Graph has no inputs");
    }
    if (outputs_.empty()) {
        return Status::Error(StatusCode::ERROR_INVALID_MODEL, "Graph has no outputs");
    }
    
    // 4. 检查节点ID和Value ID唯一性
    std::unordered_set<int64_t> node_ids;
    std::unordered_set<int64_t> value_ids;
    
    for (const auto& node : nodes_) {
        int64_t id = node->GetId();
        if (node_ids.find(id) != node_ids.end()) {
            return Status::Error(StatusCode::ERROR_INVALID_MODEL,
                               "Duplicate node ID: " + std::to_string(id));
        }
        node_ids.insert(id);
    }
    
    for (const auto& value : values_) {
        int64_t id = value->GetId();
        if (value_ids.find(id) != value_ids.end()) {
            return Status::Error(StatusCode::ERROR_INVALID_MODEL,
                               "Duplicate value ID: " + std::to_string(id));
        }
        value_ids.insert(id);
    }
    
    // 5. 检查输入输出Value在values_中存在
    for (Value* input : inputs_) {
        bool found = false;
        for (const auto& value : values_) {
            if (value.get() == input) {
                found = true;
                break;
            }
        }
        if (!found) {
            return Status::Error(StatusCode::ERROR_INVALID_MODEL,
                               "Graph input value not found in values list");
        }
    }
    
    for (Value* output : outputs_) {
        bool found = false;
        for (const auto& value : values_) {
            if (value.get() == output) {
                found = true;
                break;
            }
        }
        if (!found) {
            return Status::Error(StatusCode::ERROR_INVALID_MODEL,
                               "Graph output value not found in values list");
        }
    }
    
    // 6. 检查无循环依赖（通过拓扑排序验证）
    try {
        auto sorted = TopologicalSort();
        if (sorted.size() != nodes_.size()) {
            return Status::Error(StatusCode::ERROR_INVALID_MODEL,
                               "Graph contains cycles or disconnected nodes");
        }
    } catch (...) {
        return Status::Error(StatusCode::ERROR_INVALID_MODEL,
                           "Graph validation failed during topological sort");
    }
    
    return Status::Ok();
}

Status Graph::Serialize(const std::string& filepath) const {
    // 使用简单的文本格式序列化（参考ONNX的文本格式）
    // 格式：
    // Graph {
    //   inputs: [value_id1, value_id2, ...]
    //   outputs: [value_id1, value_id2, ...]
    //   nodes: [
    //     Node { id, op_type, name, inputs: [value_id, ...], outputs: [value_id, ...], attrs: {...} }
    //     ...
    //   ]
    //   values: [
    //     Value { id, shape, dtype }
    //     ...
    //   ]
    // }
    
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return Status::Error(StatusCode::ERROR_RUNTIME_ERROR, "Failed to open file: " + filepath);
    }
    
    file << "Graph {\n";
    
    // 序列化输入
    file << "  inputs: [";
    for (size_t i = 0; i < inputs_.size(); ++i) {
        file << inputs_[i]->GetId();
        if (i < inputs_.size() - 1) file << ", ";
    }
    file << "]\n";
    
    // 序列化输出
    file << "  outputs: [";
    for (size_t i = 0; i < outputs_.size(); ++i) {
        file << outputs_[i]->GetId();
        if (i < outputs_.size() - 1) file << ", ";
    }
    file << "]\n";
    
    // 序列化节点
    file << "  nodes: [\n";
    for (const auto& node : nodes_) {
        file << "    Node {\n";
        file << "      id: " << node->GetId() << "\n";
        file << "      op_type: \"" << node->GetOpType() << "\"\n";
        file << "      name: \"" << node->GetName() << "\"\n";
        
        file << "      inputs: [";
        for (size_t i = 0; i < node->GetInputs().size(); ++i) {
            file << node->GetInputs()[i]->GetId();
            if (i < node->GetInputs().size() - 1) file << ", ";
        }
        file << "]\n";
        
        file << "      outputs: [";
        for (size_t i = 0; i < node->GetOutputs().size(); ++i) {
            file << node->GetOutputs()[i]->GetId();
            if (i < node->GetOutputs().size() - 1) file << ", ";
        }
        file << "]\n";
        
        file << "      attrs: {";
        bool first = true;
        for (const auto& attr : node->GetAttributes()) {
            if (!first) file << ", ";
            file << "\"" << attr.first << "\": \"" << attr.second << "\"";
            first = false;
        }
        file << "}\n";
        
        file << "    }\n";
    }
    file << "  ]\n";
    
    // 序列化值
    file << "  values: [\n";
    for (const auto& value : values_) {
        file << "    Value {\n";
        file << "      id: " << value->GetId() << "\n";
        
        const Shape& shape = value->GetShape();
        file << "      shape: [";
        for (size_t i = 0; i < shape.dims.size(); ++i) {
            file << shape.dims[i];
            if (i < shape.dims.size() - 1) file << ", ";
        }
        file << "]\n";
        
        file << "      dtype: " << static_cast<int>(value->GetDataType()) << "\n";
        file << "    }\n";
    }
    file << "  ]\n";
    
    file << "}\n";
    file.close();
    
    return Status::Ok();
}

Status Graph::Deserialize(const std::string& filepath) {
    // 简化实现：读取序列化的图
    // 注意：这是一个基础实现，实际应该使用更健壮的解析器
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return Status::Error(StatusCode::ERROR_RUNTIME_ERROR, "Failed to open file: " + filepath);
    }
    
    // 清空当前图
    Clear();
    
    // 基础解析实现：解析Serialize生成的简单文本格式
    // 注意：这是一个简化实现，完整实现建议使用ONNX格式或protobuf/JSON
    
    std::string line;
    std::string section;
    std::vector<int64_t> input_ids, output_ids;
    std::unordered_map<int64_t, Value*> value_map;
    std::unordered_map<int64_t, Node*> node_map;
    
    // 解析文件
    while (std::getline(file, line)) {
        // 去除前后空白
        line.erase(0, line.find_first_not_of(" \t"));
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        line.erase(line.find_last_not_of(" \t") + 1);
        
        if (line.empty() || line[0] == '#') continue;
        
        // 解析输入
        if (line.find("inputs: [") != std::string::npos) {
            std::string ids_str = line.substr(line.find('[') + 1);
            ids_str = ids_str.substr(0, ids_str.find(']'));
            std::istringstream iss(ids_str);
            int64_t id;
            while (iss >> id) {
                input_ids.push_back(id);
                if (iss.peek() == ',') iss.ignore();
            }
        }
        // 解析输出
        else if (line.find("outputs: [") != std::string::npos) {
            std::string ids_str = line.substr(line.find('[') + 1);
            ids_str = ids_str.substr(0, ids_str.find(']'));
            std::istringstream iss(ids_str);
            int64_t id;
            while (iss >> id) {
                output_ids.push_back(id);
                if (iss.peek() == ',') iss.ignore();
            }
        }
        // 解析节点
        else if (line.find("Node {") != std::string::npos) {
            // 简化实现：读取节点基本信息
            // 完整实现需要解析所有字段
            int64_t node_id = -1;
            std::string op_type, name;
            std::vector<int64_t> input_value_ids, output_value_ids;
            
            while (std::getline(file, line) && line.find("}") == std::string::npos) {
                line.erase(0, line.find_first_not_of(" \t"));
                if (line.find("id:") != std::string::npos) {
                    std::istringstream iss(line.substr(line.find(':') + 1));
                    iss >> node_id;
                } else if (line.find("op_type:") != std::string::npos) {
                    size_t start = line.find('"') + 1;
                    size_t end = line.find('"', start);
                    op_type = line.substr(start, end - start);
                } else if (line.find("name:") != std::string::npos) {
                    size_t start = line.find('"') + 1;
                    size_t end = line.find('"', start);
                    name = line.substr(start, end - start);
                } else if (line.find("inputs: [") != std::string::npos) {
                    std::string ids_str = line.substr(line.find('[') + 1);
                    ids_str = ids_str.substr(0, ids_str.find(']'));
                    std::istringstream iss(ids_str);
                    int64_t id;
                    while (iss >> id) {
                        input_value_ids.push_back(id);
                        if (iss.peek() == ',') iss.ignore();
                    }
                } else if (line.find("outputs: [") != std::string::npos) {
                    std::string ids_str = line.substr(line.find('[') + 1);
                    ids_str = ids_str.substr(0, ids_str.find(']'));
                    std::istringstream iss(ids_str);
                    int64_t id;
                    while (iss >> id) {
                        output_value_ids.push_back(id);
                        if (iss.peek() == ',') iss.ignore();
                    }
                }
            }
            
            if (node_id >= 0 && !op_type.empty()) {
                Node* node = AddNode(op_type, name);
                node->SetId(node_id);
                node_map[node_id] = node;
                
                // 创建或获取Value
                for (int64_t vid : input_value_ids) {
                    if (value_map.find(vid) == value_map.end()) {
                        Value* v = AddValue();
                        v->SetId(vid);
                        value_map[vid] = v;
                    }
                    node->AddInput(value_map[vid]);
                }
                for (int64_t vid : output_value_ids) {
                    if (value_map.find(vid) == value_map.end()) {
                        Value* v = AddValue();
                        v->SetId(vid);
                        value_map[vid] = v;
                    }
                    node->AddOutput(value_map[vid]);
                }
            }
        }
        // 解析Value（简化：只读取ID和形状）
        else if (line.find("Value {") != std::string::npos) {
            int64_t value_id = -1;
            while (std::getline(file, line) && line.find("}") == std::string::npos) {
                line.erase(0, line.find_first_not_of(" \t"));
                if (line.find("id:") != std::string::npos) {
                    std::istringstream iss(line.substr(line.find(':') + 1));
                    iss >> value_id;
                }
            }
            if (value_id >= 0 && value_map.find(value_id) == value_map.end()) {
                Value* v = AddValue();
                v->SetId(value_id);
                value_map[value_id] = v;
            }
        }
    }
    
    // 设置图的输入输出
    for (int64_t id : input_ids) {
        if (value_map.find(id) != value_map.end()) {
            AddInput(value_map[id]);
        }
    }
    for (int64_t id : output_ids) {
        if (value_map.find(id) != value_map.end()) {
            AddOutput(value_map[id]);
        }
    }
    
    // 验证图
    Status validate_status = Validate();
    if (!validate_status.IsOk()) {
        return Status::Error(StatusCode::ERROR_INVALID_MODEL,
                           "Deserialized graph validation failed: " + validate_status.Message());
    }
    
    return Status::Ok();
}

Value* Graph::FindValueByName(const std::string& name) const {
    // 在输入中查找
    for (Value* input : inputs_) {
        if (input && input->GetName() == name) {
            return input;
        }
    }
    
    // 在输出中查找
    for (Value* output : outputs_) {
        if (output && output->GetName() == name) {
            return output;
        }
    }
    
    // 在所有值中查找
    for (const auto& value : values_) {
        if (value && value->GetName() == name) {
            return value.get();
        }
    }
    
    return nullptr;
}

std::string Graph::ToDot() const {
    std::ostringstream oss;
    oss << "digraph G {\n";
    
    // 节点
    for (const auto& node : nodes_) {
        oss << "  node" << node->GetId() << " [label=\"" << node->GetOpType();
        if (!node->GetName().empty()) {
            oss << "\\n" << node->GetName();
        }
        oss << "\"];\n";
    }
    
    // 边
    for (const auto& node : nodes_) {
        for (Value* output : node->GetOutputs()) {
            for (Node* consumer : output->GetConsumers()) {
                oss << "  node" << node->GetId() << " -> node" << consumer->GetId() << ";\n";
            }
        }
    }
    
    oss << "}\n";
    return oss.str();
}

} // namespace inferunity

