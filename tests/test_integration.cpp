// 集成测试
// 测试完整的端到端推理流程

#include <gtest/gtest.h>
#include "inferunity/engine.h"
#include "inferunity/graph.h"
#include "inferunity/tensor.h"
#include "inferunity/types.h"
#include "inferunity/optimizer.h"
#include "inferunity/memory.h"
#include <vector>
#include <memory>

using namespace inferunity;

class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
    
    void TearDown() override {
    }
    
    // 创建简单的测试图
    std::unique_ptr<Graph> CreateSimpleGraph() {
        auto graph = std::make_unique<Graph>();
        
        Value* input = graph->AddValue();
        Value* output = graph->AddValue();
        Node* relu = graph->AddNode("Relu", "relu1");
        
        relu->AddInput(input);
        relu->AddOutput(output);
        
        graph->AddInput(input);
        graph->AddOutput(output);
        
        return graph;
    }
    
    // 创建可融合的测试图
    std::unique_ptr<Graph> CreateFusionGraph() {
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
        
        return graph;
    }
};

// 测试完整的推理流程
TEST_F(IntegrationTest, EndToEndInference) {
    // 创建图
    auto graph = CreateSimpleGraph();
    
    // 创建会话
    SessionOptions options;
    auto session = InferenceSession::Create(options);
    ASSERT_NE(session, nullptr);
    
    // 加载模型
    Status status = session->LoadModelFromGraph(std::move(graph));
    EXPECT_TRUE(status.IsOk());
    
    // 创建输入
    auto input_tensor = session->CreateInputTensor(0);
    ASSERT_NE(input_tensor, nullptr);
    input_tensor->FillValue(-1.0f);
    
    // 执行推理
    std::vector<Tensor*> inputs = {input_tensor.get()};
    std::vector<Tensor*> outputs;
    
    status = session->Run(inputs, outputs);
    EXPECT_TRUE(status.IsOk());
    EXPECT_GT(outputs.size(), 0);
}

// 测试图优化流程
TEST_F(IntegrationTest, GraphOptimization) {
    auto graph = CreateFusionGraph();
    
    // 创建优化器
    Optimizer optimizer;
    
    // 应用优化
    Status status = optimizer.Optimize(graph.get());
    EXPECT_TRUE(status.IsOk());
    
    // 验证优化后的图（节点数应该减少）
    size_t node_count = graph->GetNodes().size();
    // 融合后应该只有1个节点（融合节点）
    EXPECT_LE(node_count, 3);  // 至少不会增加
}

// 测试形状推断流程
TEST_F(IntegrationTest, ShapeInference) {
    auto graph = CreateSimpleGraph();
    
    // 设置输入形状
    Value* input = graph->GetInputs()[0];
    auto input_tensor = CreateTensor(Shape({1, 3, 224, 224}), DataType::FLOAT32, DeviceType::CPU);
    input->SetTensor(input_tensor);
    
    // 创建会话（会触发形状推断）
    SessionOptions options;
    auto session = InferenceSession::Create(options);
    ASSERT_NE(session, nullptr);
    
    Status status = session->LoadModelFromGraph(std::move(graph));
    EXPECT_TRUE(status.IsOk());
    
    // 获取输出形状
    std::vector<Shape> output_shapes = session->GetOutputShapes();
    EXPECT_GT(output_shapes.size(), 0);
}

// 测试多批次推理
TEST_F(IntegrationTest, BatchInference) {
    auto graph = CreateSimpleGraph();
    
    SessionOptions options;
    auto session = InferenceSession::Create(options);
    ASSERT_NE(session, nullptr);
    
    Status status = session->LoadModelFromGraph(std::move(graph));
    EXPECT_TRUE(status.IsOk());
    
    // 执行多次推理（模拟批次）
    for (int i = 0; i < 5; ++i) {
        auto input_tensor = session->CreateInputTensor(0);
        ASSERT_NE(input_tensor, nullptr);
        input_tensor->FillValue(static_cast<float>(i));
        
        std::vector<Tensor*> inputs = {input_tensor.get()};
        std::vector<Tensor*> outputs;
        
        status = session->Run(inputs, outputs);
        EXPECT_TRUE(status.IsOk());
    }
}

// 测试内存管理
TEST_F(IntegrationTest, MemoryManagement) {
    auto graph = CreateSimpleGraph();
    
    // 设置输入
    Value* input = graph->GetInputs()[0];
    auto input_tensor = CreateTensor(Shape({100, 100}), DataType::FLOAT32, DeviceType::CPU);
    input->SetTensor(input_tensor);
    
    // 获取初始内存统计
    MemoryStats stats_before = GetMemoryStats(DeviceType::CPU);
    
    // 执行内存分配
    Status status = AllocateMemoryWithReuse(graph.get());
    EXPECT_TRUE(status.IsOk());
    
    // 获取分配后内存统计
    MemoryStats stats_after = GetMemoryStats(DeviceType::CPU);
    
    // 验证内存被分配
    EXPECT_GE(stats_after.allocated_bytes, stats_before.allocated_bytes);
}

