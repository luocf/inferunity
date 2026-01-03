#pragma once

#include "inferunity/graph.h"
#include "inferunity/types.h"
#include <string>
#include <memory>

namespace inferunity {
namespace frontend {

// ONNX模型解析器
// 参考ONNX Runtime的模型加载机制，但简化实现
class ONNXParser {
public:
    ONNXParser();
    ~ONNXParser();
    
    // 从文件加载ONNX模型
    Status LoadFromFile(const std::string& filepath);
    
    // 从内存加载ONNX模型
    Status LoadFromMemory(const void* data, size_t size);
    
    // 转换为内部Graph
    Status ConvertToGraph(std::unique_ptr<Graph>& graph);
    
    // 获取模型信息
    std::string GetModelVersion() const { return model_version_; }
    std::vector<std::string> GetInputNames() const { return input_names_; }
    std::vector<std::string> GetOutputNames() const { return output_names_; }
    
private:
    // 解析ONNX节点
    Status ParseNode(const void* onnx_node, Node* node);
    
    // 解析ONNX张量
    Status ParseTensor(const void* onnx_tensor, Value* value);
    
    // 转换数据类型
    DataType ConvertDataType(int32_t onnx_dtype);
    
    // 转换属性
    Status ConvertAttributes(const void* onnx_node, Node* node);
    
    std::string model_version_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    void* model_proto_;  // ONNX ModelProto指针
};

} // namespace frontend
} // namespace inferunity

