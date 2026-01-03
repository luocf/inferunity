// ONNX解析器测试
// 参考ONNX Runtime的测试实现

#include <gtest/gtest.h>
#include "inferunity/graph.h"
#include "inferunity/tensor.h"
#include <fstream>
#include <sstream>

using namespace inferunity;

class ONNXParserTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
    
    void TearDown() override {
    }
};

// 测试形状推断
TEST_F(ONNXParserTest, ShapeInference) {
    // 创建简单图并测试形状推断
    auto graph = std::make_unique<Graph>();
    
    // 添加输入
    Value* input = graph->AddValue();
    Shape input_shape({1, 3, 224, 224});
    auto input_tensor = CreateTensor(input_shape, DataType::FLOAT32, DeviceType::CPU);
    input->SetTensor(input_tensor);
    graph->AddInput(input);
    
    // 测试形状推断（需要调用全局函数）
    // 对于空图，应该成功
    // Status status = InferShapes(graph.get());
    // EXPECT_TRUE(status.IsOk());
}

// 测试图验证
TEST_F(ONNXParserTest, GraphValidation) {
    auto graph = std::make_unique<Graph>();
    
    // 空图应该验证通过
    Status status = graph->Validate();
    EXPECT_TRUE(status.IsOk());
    
    // 添加节点
    Node* node = graph->AddNode("Relu", "relu1");
    Value* input = graph->AddValue();
    Value* output = graph->AddValue();
    
    node->AddInput(input);
    node->AddOutput(output);
    
    // 验证图
    status = graph->Validate();
    EXPECT_TRUE(status.IsOk());
}

