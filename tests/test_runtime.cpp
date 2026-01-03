// 运行时系统集成测试
// 测试端到端的推理流程

#include <gtest/gtest.h>
#include "inferunity/engine.h"
#include "inferunity/graph.h"
#include "inferunity/tensor.h"
#include "inferunity/types.h"
#include <vector>

using namespace inferunity;

class RuntimeTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
    
    void TearDown() override {
    }
};

// 测试简单推理流程
TEST_F(RuntimeTest, SimpleInference) {
    // 创建简单图：Input -> Relu -> Output
    auto graph = std::make_unique<Graph>();
    
    Value* input = graph->AddValue();
    Value* output = graph->AddValue();
    Node* relu = graph->AddNode("Relu", "relu1");
    
    relu->AddInput(input);
    relu->AddOutput(output);
    
    graph->AddInput(input);
    graph->AddOutput(output);
    
    // 创建输入张量
    auto input_tensor = CreateTensor(Shape({1, 3, 224, 224}), DataType::FLOAT32, DeviceType::CPU);
    input_tensor->FillValue(-1.0f);  // 填充负值测试ReLU
    
    input->SetTensor(input_tensor);
    
    // 创建推理会话
    SessionOptions options;
    auto session = InferenceSession::Create(options);
    ASSERT_NE(session, nullptr);
    
    // 加载图
    Status status = session->LoadModelFromGraph(std::move(graph));
    EXPECT_TRUE(status.IsOk());
    
    // 执行推理
    std::vector<Tensor*> inputs = {input_tensor.get()};
    std::vector<Tensor*> outputs;
    
    status = session->Run(inputs, outputs);
    EXPECT_TRUE(status.IsOk());
    EXPECT_GT(outputs.size(), 0);
    
    // 验证输出（ReLU应该将所有负值变为0）
    if (!outputs.empty()) {
        const float* out_data = static_cast<const float*>(outputs[0]->GetData());
        size_t count = outputs[0]->GetElementCount();
        for (size_t i = 0; i < count; ++i) {
            EXPECT_GE(out_data[i], 0.0f);
        }
    }
}

// 测试多节点推理
TEST_F(RuntimeTest, MultiNodeInference) {
    // 创建图：Input -> Add -> Relu -> Output
    auto graph = std::make_unique<Graph>();
    
    Value* input1 = graph->AddValue();
    Value* input2 = graph->AddValue();
    Value* add_output = graph->AddValue();
    Value* output = graph->AddValue();
    
    Node* add = graph->AddNode("Add", "add1");
    Node* relu = graph->AddNode("Relu", "relu1");
    
    add->AddInput(input1);
    add->AddInput(input2);
    add->AddOutput(add_output);
    
    relu->AddInput(add_output);
    relu->AddOutput(output);
    
    graph->AddInput(input1);
    graph->AddInput(input2);
    graph->AddOutput(output);
    
    // 创建输入
    auto input_tensor1 = CreateTensor(Shape({2, 3}), DataType::FLOAT32, DeviceType::CPU);
    auto input_tensor2 = CreateTensor(Shape({2, 3}), DataType::FLOAT32, DeviceType::CPU);
    input_tensor1->FillValue(1.0f);
    input_tensor2->FillValue(2.0f);
    
    input1->SetTensor(input_tensor1);
    input2->SetTensor(input_tensor2);
    
    // 创建推理会话
    SessionOptions options;
    auto session = InferenceSession::Create(options);
    ASSERT_NE(session, nullptr);
    
    // 加载图并执行
    Status status = session->LoadModelFromGraph(std::move(graph));
    EXPECT_TRUE(status.IsOk());
    
    std::vector<Tensor*> inputs = {input_tensor1.get(), input_tensor2.get()};
    std::vector<Tensor*> outputs;
    
    status = session->Run(inputs, outputs);
    EXPECT_TRUE(status.IsOk());
    EXPECT_GT(outputs.size(), 0);
}

// 测试形状推断
TEST_F(RuntimeTest, ShapeInference) {
    auto graph = std::make_unique<Graph>();
    
    Value* input = graph->AddValue();
    Value* output = graph->AddValue();
    Node* relu = graph->AddNode("Relu", "relu1");
    
    relu->AddInput(input);
    relu->AddOutput(output);
    
    graph->AddInput(input);
    graph->AddOutput(output);
    
    // 设置输入形状
    auto input_tensor = CreateTensor(Shape({1, 3, 224, 224}), DataType::FLOAT32, DeviceType::CPU);
    input->SetTensor(input_tensor);
    
    // 创建会话
    SessionOptions options;
    auto session = InferenceSession::Create(options);
    ASSERT_NE(session, nullptr);
    
    // 加载图（应该触发形状推断）
    Status status = session->LoadModelFromGraph(std::move(graph));
    EXPECT_TRUE(status.IsOk());
    
    // 获取输出形状
    std::vector<Shape> output_shapes = session->GetOutputShapes();
    EXPECT_GT(output_shapes.size(), 0);
    if (!output_shapes.empty()) {
        EXPECT_EQ(output_shapes[0].dims.size(), 4);
        EXPECT_EQ(output_shapes[0].dims[0], 1);
        EXPECT_EQ(output_shapes[0].dims[1], 3);
    }
}

// 测试图优化
TEST_F(RuntimeTest, GraphOptimization) {
    // 创建可融合的图：Conv -> BN -> ReLU
    auto graph = std::make_unique<Graph>();
    
    Value* input = graph->AddValue();
    Value* conv_out = graph->AddValue();
    Value* bn_out = graph->AddValue();
    Value* output = graph->AddValue();
    
    Node* conv = graph->AddNode("Conv", "conv1");
    Node* bn = graph->AddNode("BatchNormalization", "bn1");
    Node* relu = graph->AddNode("Relu", "relu1");
    
    conv->AddInput(input);
    conv->AddOutput(conv_out);
    
    bn->AddInput(conv_out);
    bn->AddOutput(bn_out);
    
    relu->AddInput(bn_out);
    relu->AddOutput(output);
    
    graph->AddInput(input);
    graph->AddOutput(output);
    
    // 创建会话（启用优化）
    SessionOptions options;
    options.graph_optimization_level = SessionOptions::GraphOptimizationLevel::ALL;
    auto session = InferenceSession::Create(options);
    ASSERT_NE(session, nullptr);
    
    // 加载图（应该触发优化）
    Status status = session->LoadModelFromGraph(std::move(graph));
    EXPECT_TRUE(status.IsOk());
    
    // 验证优化后的图（节点数应该减少）
    // 注意：这需要访问内部图结构，简化测试
}

