// 融合算子单元测试
// 测试FusedConvBNReLU和FusedMatMulAdd

#include <gtest/gtest.h>
#include "inferunity/operator.h"
#include "inferunity/tensor.h"
#include "inferunity/types.h"
#include <vector>
#include <cmath>
#include <cstring>

using namespace inferunity;

class FusedOperatorsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 测试前准备
    }
    
    void TearDown() override {
        // 测试后清理
    }
    
    // 辅助函数：创建测试张量
    std::shared_ptr<Tensor> CreateTestTensor(const Shape& shape, DataType dtype, 
                                        const std::vector<float>& data = {}) {
        auto tensor = inferunity::CreateTensor(shape, dtype, DeviceType::CPU);
        if (!data.empty()) {
            float* ptr = static_cast<float*>(tensor->GetData());
            size_t count = std::min(data.size(), tensor->GetElementCount());
            std::memcpy(ptr, data.data(), count * sizeof(float));
        }
        return tensor;
    }
    
    // 辅助函数：比较两个张量
    bool CompareTensors(Tensor* a, Tensor* b, float tolerance = 1e-5f) {
        const auto& shape_a = a->GetShape();
        const auto& shape_b = b->GetShape();
        if (shape_a.dims != shape_b.dims) return false;
        if (a->GetDataType() != b->GetDataType()) return false;
        
        size_t count = a->GetElementCount();
        const float* data_a = static_cast<const float*>(a->GetData());
        const float* data_b = static_cast<const float*>(b->GetData());
        
        for (size_t i = 0; i < count; ++i) {
            if (std::abs(data_a[i] - data_b[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }
};

// 测试FusedMatMulAdd算子
TEST_F(FusedOperatorsTest, FusedMatMulAddBasic) {
    // 创建算子
    auto op = OperatorRegistry::Instance().Create("FusedMatMulAdd");
    ASSERT_NE(op, nullptr);
    
    // 创建输入：A [2, 3], B [3, 4], bias [4]
    Shape shape_a({2, 3});
    Shape shape_b({3, 4});
    Shape shape_bias({4});
    Shape shape_out({2, 4});
    
    // 初始化数据
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> data_b = {1.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 1.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 1.0f, 0.0f};
    std::vector<float> data_bias = {0.1f, 0.2f, 0.3f, 0.4f};
    
    auto tensor_a = CreateTestTensor(shape_a, DataType::FLOAT32, data_a);
    auto tensor_b = CreateTestTensor(shape_b, DataType::FLOAT32, data_b);
    auto tensor_bias = CreateTestTensor(shape_bias, DataType::FLOAT32, data_bias);
    auto tensor_out = CreateTestTensor(shape_out, DataType::FLOAT32);
    
    // 验证输入
    std::vector<Tensor*> inputs = {tensor_a.get(), tensor_b.get(), tensor_bias.get()};
    Status status = op->ValidateInputs(inputs);
    EXPECT_TRUE(status.IsOk());
    
    // 推断输出形状
    std::vector<Shape> output_shapes;
    status = op->InferOutputShape(inputs, output_shapes);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0].dims, shape_out.dims);
    
    // 执行算子
    std::vector<Tensor*> outputs = {tensor_out.get()};
    ExecutionContext ctx;
    status = op->Execute(inputs, outputs, &ctx);
    EXPECT_TRUE(status.IsOk());
    
    // 验证输出
    // 期望结果：[1.1, 2.2, 3.3, 0.4], [4.1, 5.2, 6.3, 0.4]
    const float* out_data = static_cast<const float*>(tensor_out->GetData());
    EXPECT_NEAR(out_data[0], 1.1f, 1e-5f);
    EXPECT_NEAR(out_data[1], 2.2f, 1e-5f);
    EXPECT_NEAR(out_data[2], 3.3f, 1e-5f);
    EXPECT_NEAR(out_data[3], 0.4f, 1e-5f);
    EXPECT_NEAR(out_data[4], 4.1f, 1e-5f);
    EXPECT_NEAR(out_data[5], 5.2f, 1e-5f);
    EXPECT_NEAR(out_data[6], 6.3f, 1e-5f);
    EXPECT_NEAR(out_data[7], 0.4f, 1e-5f);
}

// 测试FusedConvBNReLU算子（简化测试）
TEST_F(FusedOperatorsTest, FusedConvBNReLUBasic) {
    // 创建算子
    auto op = OperatorRegistry::Instance().Create("FusedConvBNReLU");
    ASSERT_NE(op, nullptr);
    
    // 创建输入：input [1, 1, 3, 3], weight [1, 1, 2, 2], scale [1], B [1], mean [1], var [1]
    Shape shape_input({1, 1, 3, 3});
    Shape shape_weight({1, 1, 2, 2});
    Shape shape_params({1});
    Shape shape_out({1, 1, 2, 2});
    
    // 初始化数据
    std::vector<float> data_input(9, 1.0f);
    std::vector<float> data_weight(4, 1.0f);
    std::vector<float> data_scale = {1.0f};
    std::vector<float> data_B = {0.0f};
    std::vector<float> data_mean = {0.0f};
    std::vector<float> data_var = {1.0f};
    
    auto tensor_input = CreateTestTensor(shape_input, DataType::FLOAT32, data_input);
    auto tensor_weight = CreateTestTensor(shape_weight, DataType::FLOAT32, data_weight);
    auto tensor_scale = CreateTestTensor(shape_params, DataType::FLOAT32, data_scale);
    auto tensor_B = CreateTestTensor(shape_params, DataType::FLOAT32, data_B);
    auto tensor_mean = CreateTestTensor(shape_params, DataType::FLOAT32, data_mean);
    auto tensor_var = CreateTestTensor(shape_params, DataType::FLOAT32, data_var);
    auto tensor_out = CreateTestTensor(shape_out, DataType::FLOAT32);
    
    // 验证输入
    std::vector<Tensor*> inputs = {
        tensor_input.get(), tensor_weight.get(), nullptr,
        tensor_scale.get(), tensor_B.get(), tensor_mean.get(), tensor_var.get()
    };
    Status status = op->ValidateInputs(inputs);
    EXPECT_TRUE(status.IsOk());
    
    // 推断输出形状
    std::vector<Shape> output_shapes;
    status = op->InferOutputShape(inputs, output_shapes);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(output_shapes.size(), 1);
    
    // 执行算子
    std::vector<Tensor*> outputs = {tensor_out.get()};
    ExecutionContext ctx;
    status = op->Execute(inputs, outputs, &ctx);
    EXPECT_TRUE(status.IsOk());
    
    // 验证输出（所有值应该>=0，因为ReLU）
    const float* out_data = static_cast<const float*>(tensor_out->GetData());
    for (size_t i = 0; i < tensor_out->GetElementCount(); ++i) {
        EXPECT_GE(out_data[i], 0.0f);
    }
}

// 测试算子注册
TEST_F(FusedOperatorsTest, OperatorRegistration) {
    auto& registry = OperatorRegistry::Instance();
    
    // 检查融合算子是否已注册
    EXPECT_TRUE(registry.IsRegistered("FusedConvBNReLU"));
    EXPECT_TRUE(registry.IsRegistered("FusedMatMulAdd"));
    
    // 测试创建算子
    auto op1 = registry.Create("FusedConvBNReLU");
    EXPECT_NE(op1, nullptr);
    EXPECT_EQ(op1->GetName(), "FusedConvBNReLU");
    
    auto op2 = registry.Create("FusedMatMulAdd");
    EXPECT_NE(op2, nullptr);
    EXPECT_EQ(op2->GetName(), "FusedMatMulAdd");
}

