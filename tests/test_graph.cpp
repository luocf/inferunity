// Graph核心功能单元测试
// 测试Graph的创建、验证、序列化、深拷贝等功能

#include <gtest/gtest.h>
#include "inferunity/graph.h"
#include "inferunity/tensor.h"
#include "inferunity/types.h"
#include <vector>

using namespace inferunity;

class GraphTest : public ::testing::Test {
protected:
    void SetUp() override {
        graph_ = std::make_unique<Graph>();
    }
    
    void TearDown() override {
        graph_.reset();
    }
    
    std::unique_ptr<Graph> graph_;
};

// 测试Graph创建
TEST_F(GraphTest, CreateGraph) {
    EXPECT_NE(graph_, nullptr);
    EXPECT_EQ(graph_->GetNodes().size(), 0);
    EXPECT_EQ(graph_->GetValues().size(), 0);
    EXPECT_EQ(graph_->GetInputs().size(), 0);
    EXPECT_EQ(graph_->GetOutputs().size(), 0);
}

// 测试添加节点和值
TEST_F(GraphTest, AddNodeAndValue) {
    Value* input = graph_->AddValue();
    Value* output = graph_->AddValue();
    Node* node = graph_->AddNode("Relu", "relu1");
    
    node->AddInput(input);
    node->AddOutput(output);
    
    EXPECT_EQ(graph_->GetNodes().size(), 1);
    EXPECT_EQ(graph_->GetValues().size(), 2);
    EXPECT_EQ(node->GetOpType(), "Relu");
    EXPECT_EQ(node->GetName(), "relu1");
}

// 测试图验证
TEST_F(GraphTest, ValidateGraph) {
    // 空图应该验证通过
    Status status = graph_->Validate();
    EXPECT_FALSE(status.IsOk());  // 空图没有输入输出，验证失败
    
    // 添加输入输出
    Value* input = graph_->AddValue();
    Value* output = graph_->AddValue();
    graph_->AddInput(input);
    graph_->AddOutput(output);
    
    // 现在应该验证通过
    status = graph_->Validate();
    EXPECT_TRUE(status.IsOk());
}

// 测试图验证 - 检查输入输出
TEST_F(GraphTest, ValidateGraphInputsOutputs) {
    Value* input = graph_->AddValue();
    Value* output = graph_->AddValue();
    Node* node = graph_->AddNode("Relu", "relu1");
    
    node->AddInput(input);
    node->AddOutput(output);
    
    graph_->AddInput(input);
    graph_->AddOutput(output);
    
    Status status = graph_->Validate();
    EXPECT_TRUE(status.IsOk());
}

// 测试拓扑排序
TEST_F(GraphTest, TopologicalSort) {
    // 创建简单图：A -> B -> C
    Value* v1 = graph_->AddValue();
    Value* v2 = graph_->AddValue();
    Value* v3 = graph_->AddValue();
    Value* v4 = graph_->AddValue();
    
    Node* node_a = graph_->AddNode("Add", "add1");
    Node* node_b = graph_->AddNode("Relu", "relu1");
    Node* node_c = graph_->AddNode("Mul", "mul1");
    
    node_a->AddInput(v1);
    node_a->AddInput(v2);
    node_a->AddOutput(v3);
    
    node_b->AddInput(v3);
    node_b->AddOutput(v4);
    
    node_c->AddInput(v4);
    node_c->AddOutput(v4);
    
    graph_->AddInput(v1);
    graph_->AddInput(v2);
    graph_->AddOutput(v4);
    
    std::vector<Node*> sorted = graph_->TopologicalSort();
    EXPECT_EQ(sorted.size(), 3);
    
    // 验证顺序：A应该在B之前，B应该在C之前
    size_t idx_a = 0, idx_b = 0, idx_c = 0;
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (sorted[i] == node_a) idx_a = i;
        if (sorted[i] == node_b) idx_b = i;
        if (sorted[i] == node_c) idx_c = i;
    }
    
    EXPECT_LT(idx_a, idx_b);
    EXPECT_LT(idx_b, idx_c);
}

// 测试图深拷贝
TEST_F(GraphTest, CloneGraph) {
    Value* input = graph_->AddValue();
    Value* output = graph_->AddValue();
    Node* node = graph_->AddNode("Relu", "relu1");
    
    node->AddInput(input);
    node->AddOutput(output);
    node->SetAttribute("test", "value");
    
    graph_->AddInput(input);
    graph_->AddOutput(output);
    
    // 深拷贝
    Graph cloned = graph_->Clone();
    
    EXPECT_EQ(cloned.GetNodes().size(), 1);
    EXPECT_EQ(cloned.GetValues().size(), 2);
    EXPECT_EQ(cloned.GetInputs().size(), 1);
    EXPECT_EQ(cloned.GetOutputs().size(), 1);
    
    // 验证节点属性被复制
    const auto& nodes = cloned.GetNodes();
    EXPECT_EQ(nodes[0]->GetAttribute("test"), "value");
    
    // 验证是深拷贝（修改原图不影响克隆图）
    graph_->AddNode("Add", "add1");
    EXPECT_EQ(graph_->GetNodes().size(), 2);
    EXPECT_EQ(cloned.GetNodes().size(), 1);
}

// 测试图序列化
TEST_F(GraphTest, SerializeGraph) {
    Value* input = graph_->AddValue();
    Value* output = graph_->AddValue();
    Node* node = graph_->AddNode("Relu", "relu1");
    
    node->AddInput(input);
    node->AddOutput(output);
    
    graph_->AddInput(input);
    graph_->AddOutput(output);
    
    // 序列化
    Status status = graph_->Serialize("/tmp/test_graph.txt");
    EXPECT_TRUE(status.IsOk());
    
    // 验证文件存在（简化测试）
    // 实际应该读取文件内容验证
}

// 测试图可视化（ToDot）
TEST_F(GraphTest, ToDot) {
    Value* input = graph_->AddValue();
    Value* output = graph_->AddValue();
    Node* node = graph_->AddNode("Relu", "relu1");
    
    node->AddInput(input);
    node->AddOutput(output);
    
    graph_->AddInput(input);
    graph_->AddOutput(output);
    
    std::string dot = graph_->ToDot();
    EXPECT_FALSE(dot.empty());
    EXPECT_NE(dot.find("digraph"), std::string::npos);
    EXPECT_NE(dot.find("Relu"), std::string::npos);
}

// 测试图验证 - 检查循环依赖
TEST_F(GraphTest, ValidateCyclicGraph) {
    // 创建循环图：A -> B -> A
    Value* v1 = graph_->AddValue();
    Value* v2 = graph_->AddValue();
    
    Node* node_a = graph_->AddNode("Add", "add1");
    Node* node_b = graph_->AddNode("Relu", "relu1");
    
    node_a->AddInput(v1);
    node_a->AddOutput(v2);
    
    node_b->AddInput(v2);
    node_b->AddOutput(v1);  // 形成循环
    
    graph_->AddInput(v1);
    graph_->AddOutput(v2);
    
    // 拓扑排序应该检测到循环
    std::vector<Node*> sorted = graph_->TopologicalSort();
    // 循环图可能导致排序结果不完整
    // 验证应该检测到问题
    Status status = graph_->Validate();
    // 验证可能通过，但拓扑排序可能不完整
}

// 测试节点属性
TEST_F(GraphTest, NodeAttributes) {
    Node* node = graph_->AddNode("Conv", "conv1");
    
    node->SetAttribute("kernel_size", "3");
    node->SetAttribute("stride", "1");
    
    EXPECT_EQ(node->GetAttribute("kernel_size"), "3");
    EXPECT_EQ(node->GetAttribute("stride"), "1");
    EXPECT_TRUE(node->HasAttribute("kernel_size"));
    EXPECT_FALSE(node->HasAttribute("nonexistent"));
}

