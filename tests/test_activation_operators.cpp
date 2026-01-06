// 激活函数算子测试
#include <gtest/gtest.h>
#include "inferunity/tensor.h"
#include "inferunity/operator.h"
#include "inferunity/types.h"
#include <vector>
#include <cmath>

using namespace inferunity;

class ActivationOperatorsTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx_ = std::make_unique<ExecutionContext>();
    }
    
    std::unique_ptr<ExecutionContext> ctx_;
};

// ReLU测试
TEST_F(ActivationOperatorsTest, ReluOperator) {
    auto& registry = OperatorRegistry::Instance();
    auto relu_op = registry.Create("Relu");
    ASSERT_NE(relu_op, nullptr);
    
    Shape shape({4});
    auto input = CreateTensor(shape, DataType::FLOAT32);
    auto output = CreateTensor(shape, DataType::FLOAT32);
    
    float* data = static_cast<float*>(input->GetData());
    data[0] = -2.0f;
    data[1] = 0.0f;
    data[2] = 3.0f;
    data[3] = -1.5f;
    
    std::vector<Tensor*> inputs = {input.get()};
    std::vector<Tensor*> outputs = {output.get()};
    Status status = relu_op->Execute(inputs, outputs, ctx_.get());
    
    ASSERT_TRUE(status.IsOk());
    
    float* out_data = static_cast<float*>(output->GetData());
    EXPECT_FLOAT_EQ(out_data[0], 0.0f);
    EXPECT_FLOAT_EQ(out_data[1], 0.0f);
    EXPECT_FLOAT_EQ(out_data[2], 3.0f);
    EXPECT_FLOAT_EQ(out_data[3], 0.0f);
}

// Sigmoid测试
TEST_F(ActivationOperatorsTest, SigmoidOperator) {
    auto& registry = OperatorRegistry::Instance();
    auto sigmoid_op = registry.Create("Sigmoid");
    ASSERT_NE(sigmoid_op, nullptr);
    
    Shape shape({3});
    auto input = CreateTensor(shape, DataType::FLOAT32);
    auto output = CreateTensor(shape, DataType::FLOAT32);
    
    float* data = static_cast<float*>(input->GetData());
    data[0] = 0.0f;
    data[1] = 1.0f;
    data[2] = -1.0f;
    
    std::vector<Tensor*> inputs = {input.get()};
    std::vector<Tensor*> outputs = {output.get()};
    Status status = sigmoid_op->Execute(inputs, outputs, ctx_.get());
    
    ASSERT_TRUE(status.IsOk());
    
    float* out_data = static_cast<float*>(output->GetData());
    EXPECT_NEAR(out_data[0], 0.5f, 1e-5f);
    EXPECT_NEAR(out_data[1], 1.0f / (1.0f + std::exp(-1.0f)), 1e-5f);
    EXPECT_NEAR(out_data[2], 1.0f / (1.0f + std::exp(1.0f)), 1e-5f);
}

// GELU测试
TEST_F(ActivationOperatorsTest, GeluOperator) {
    auto& registry = OperatorRegistry::Instance();
    auto gelu_op = registry.Create("Gelu");
    ASSERT_NE(gelu_op, nullptr);
    
    Shape shape({2});
    auto input = CreateTensor(shape, DataType::FLOAT32);
    auto output = CreateTensor(shape, DataType::FLOAT32);
    
    float* data = static_cast<float*>(input->GetData());
    data[0] = 0.0f;
    data[1] = 1.0f;
    
    std::vector<Tensor*> inputs = {input.get()};
    std::vector<Tensor*> outputs = {output.get()};
    Status status = gelu_op->Execute(inputs, outputs, ctx_.get());
    
    ASSERT_TRUE(status.IsOk());
    
    // GELU(0) = 0
    float* out_data = static_cast<float*>(output->GetData());
    EXPECT_NEAR(out_data[0], 0.0f, 1e-5f);
    // GELU(1) 应该是一个正数
    EXPECT_GT(out_data[1], 0.0f);
}

// SiLU测试
TEST_F(ActivationOperatorsTest, SiluOperator) {
    auto& registry = OperatorRegistry::Instance();
    auto silu_op = registry.Create("Silu");
    ASSERT_NE(silu_op, nullptr);
    
    Shape shape({2});
    auto input = CreateTensor(shape, DataType::FLOAT32);
    auto output = CreateTensor(shape, DataType::FLOAT32);
    
    float* data = static_cast<float*>(input->GetData());
    data[0] = 0.0f;
    data[1] = 1.0f;
    
    std::vector<Tensor*> inputs = {input.get()};
    std::vector<Tensor*> outputs = {output.get()};
    Status status = silu_op->Execute(inputs, outputs, ctx_.get());
    
    ASSERT_TRUE(status.IsOk());
    
    // SiLU(0) = 0
    float* out_data = static_cast<float*>(output->GetData());
    EXPECT_NEAR(out_data[0], 0.0f, 1e-5f);
    // SiLU(1) = 1 * sigmoid(1) > 0
    EXPECT_GT(out_data[1], 0.0f);
}

