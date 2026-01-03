#pragma once

#include "types.h"
#include "tensor.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>

namespace inferunity {

// 前向声明
class Node;
class Value;

// 值节点 - 表示计算图中的数据流
class Value {
public:
    Value() : id_(-1), tensor_(nullptr), producer_(nullptr) {}
    explicit Value(int64_t id) : id_(id), tensor_(nullptr), producer_(nullptr) {}
    
    int64_t GetId() const { return id_; }
    void SetId(int64_t id) { id_ = id; }
    
    const std::string& GetName() const { return name_; }
    void SetName(const std::string& name) { name_ = name; }
    
    std::shared_ptr<Tensor> GetTensor() const { return tensor_; }
    void SetTensor(std::shared_ptr<Tensor> tensor) { tensor_ = tensor; }
    
    const Shape& GetShape() const;
    DataType GetDataType() const;
    
    Node* GetProducer() const { return producer_; }
    void SetProducer(Node* node) { producer_ = node; }
    
    const std::vector<Node*>& GetConsumers() const { return consumers_; }
    void AddConsumer(Node* node);
    void RemoveConsumer(Node* node);
    
private:
    int64_t id_;
    std::string name_;  // 值名称（用于输入输出映射）
    std::shared_ptr<Tensor> tensor_;
    Node* producer_;
    std::vector<Node*> consumers_;
};

// 节点属性（键值对）
using NodeAttributes = std::unordered_map<std::string, std::string>;

// 计算节点
class Node {
public:
    Node() : id_(-1), op_type_(""), name_("") {}
    Node(int64_t id, const std::string& op_type, const std::string& name = "")
        : id_(id), op_type_(op_type), name_(name) {}
    
    int64_t GetId() const { return id_; }
    void SetId(int64_t id) { id_ = id; }
    
    const std::string& GetOpType() const { return op_type_; }
    void SetOpType(const std::string& op_type) { op_type_ = op_type; }
    
    const std::string& GetName() const { return name_; }
    void SetName(const std::string& name) { name_ = name; }
    
    // 输入输出
    const std::vector<Value*>& GetInputs() const { return inputs_; }
    const std::vector<Value*>& GetOutputs() const { return outputs_; }
    
    void AddInput(Value* value);
    void AddOutput(Value* value);
    void RemoveInput(Value* value);
    void RemoveOutput(Value* value);
    
    // 属性
    const NodeAttributes& GetAttributes() const { return attributes_; }
    void SetAttribute(const std::string& key, const std::string& value);
    std::string GetAttribute(const std::string& key, const std::string& default_value = "") const;
    bool HasAttribute(const std::string& key) const;
    
    // 设备分配
    DeviceType GetDevice() const { return device_; }
    void SetDevice(DeviceType device) { device_ = device; }
    
private:
    int64_t id_;
    std::string op_type_;
    std::string name_;
    std::vector<Value*> inputs_;
    std::vector<Value*> outputs_;
    NodeAttributes attributes_;
    DeviceType device_;
};

// 计算图
class Graph {
public:
    Graph();
    ~Graph();
    
    // 禁用复制构造和赋值（因为包含unique_ptr）
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    
    // 允许移动
    Graph(Graph&&) = default;
    Graph& operator=(Graph&&) = default;
    
    // 节点管理
    Node* AddNode(const std::string& op_type, const std::string& name = "");
    Node* GetNode(int64_t id) const;
    Node* GetNodeByName(const std::string& name) const;
    void RemoveNode(Node* node);
    const std::vector<std::unique_ptr<Node>>& GetNodes() const { return nodes_; }
    
    // 值管理
    Value* AddValue();
    Value* GetValue(int64_t id) const;
    Value* FindValueByName(const std::string& name) const;
    void RemoveValue(Value* value);
    const std::vector<std::unique_ptr<Value>>& GetValues() const { return values_; }
    
    // 输入输出
    void AddInput(Value* value);
    void AddOutput(Value* value);
    const std::vector<Value*>& GetInputs() const { return inputs_; }
    const std::vector<Value*>& GetOutputs() const { return outputs_; }
    
    // 图操作
    void Clear();
    Graph Clone() const;  // 深拷贝
    
    // 拓扑排序
    std::vector<Node*> TopologicalSort() const;
    
    // 验证
    Status Validate() const;
    
    // 序列化
    Status Serialize(const std::string& filepath) const;
    Status Deserialize(const std::string& filepath);
    
    // 可视化（生成DOT格式）
    std::string ToDot() const;
    
private:
    std::vector<std::unique_ptr<Node>> nodes_;
    std::vector<std::unique_ptr<Value>> values_;
    std::vector<Value*> inputs_;
    std::vector<Value*> outputs_;
    int64_t next_node_id_;
    int64_t next_value_id_;
    
    int64_t GenerateNodeId() { return next_node_id_++; }
    int64_t GenerateValueId() { return next_value_id_++; }
};

} // namespace inferunity

