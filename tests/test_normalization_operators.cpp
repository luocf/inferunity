// 归一化算子测试
// 测试 BatchNormalization, LayerNormalization, RMSNorm

#include "inferunity/operator.h"
#include "inferunity/tensor.h"
#include "inferunity/graph.h"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <memory>

using namespace inferunity;

class NormalizationOperatorsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化测试环境
        // 确保算子被注册
        inferunity::InitializeOperators();
    }
    
    void TearDown() override {
        // 清理测试环境
    }
    
    // 辅助函数：创建测试张量
    std::shared_ptr<Tensor> CreateTestTensor(const Shape& shape, const std::vector<float>& data) {
        auto tensor = inferunity::CreateTensor(shape, DataType::FLOAT32, DeviceType::CPU);
        float* ptr = static_cast<float*>(tensor->GetData());
        std::copy(data.begin(), data.end(), ptr);
        return tensor;
    }
    
    // 辅助函数：比较张量值（允许小的浮点误差）
    bool TensorNear(const Tensor* a, const Tensor* b, float tolerance = 1e-5f) {
        if (a->GetShape().GetElementCount() != b->GetElementCount()) {
            return false;
        }
        
        const float* data_a = static_cast<const float*>(a->GetData());
        const float* data_b = static_cast<const float*>(b->GetData());
        size_t count = a->GetElementCount();
        
        for (size_t i = 0; i < count; ++i) {
            if (std::abs(data_a[i] - data_b[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }
};

// 测试 BatchNormalization 基本功能
TEST_F(NormalizationOperatorsTest, BatchNormalizationBasic) {
    // 创建简单的BatchNorm测试
    // Input: [1, 2, 2, 2] = [1, 2, 3, 4, 5, 6, 7, 8]
    // Scale: [2] = [1.0, 1.0]
    // Bias: [2] = [0.0, 0.0]
    // Mean: [2] = [2.5, 6.5]  // 每个通道的均值
    // Var: [2] = [1.25, 1.25]  // 每个通道的方差
    
    auto input = CreateTestTensor(Shape({1, 2, 2, 2}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    auto scale = CreateTestTensor(Shape({2}), {1.0f, 1.0f});
    auto bias = CreateTestTensor(Shape({2}), {0.0f, 0.0f});
    auto mean = CreateTestTensor(Shape({2}), {2.5f, 6.5f});
    auto var = CreateTestTensor(Shape({2}), {1.25f, 1.25f});
    auto output = CreateTestTensor(Shape({1, 2, 2, 2}), {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    
    // 获取算子
    auto op = OperatorRegistry::Instance().Create("BatchNormalization");
    ASSERT_NE(op, nullptr);
    
    // 验证输入
    std::vector<Tensor*> inputs = {input.get(), scale.get(), bias.get(), mean.get(), var.get()};
    std::vector<Tensor*> outputs = {output.get()};
    
    Status status = op->ValidateInputs(inputs);
    EXPECT_TRUE(status.IsOk());
    
    // 执行算子
    ExecutionContext ctx;
    status = op->Execute(inputs, outputs, &ctx);
    EXPECT_TRUE(status.IsOk());
    
    // 验证输出（简化验证：检查输出不为零且值合理）
    const float* output_data = static_cast<const float*>(output->GetData());
    bool has_non_zero = false;
    for (size_t i = 0; i < output->GetElementCount(); ++i) {
        if (std::abs(output_data[i]) > 1e-6f) {
            has_non_zero = true;
        }
        // 输出值应该在合理范围内
        EXPECT_TRUE(std::isfinite(output_data[i]));
    }
    EXPECT_TRUE(has_non_zero);
}

// 测试 LayerNormalization 基本功能
TEST_F(NormalizationOperatorsTest, LayerNormalizationBasic) {
    // 创建简单的LayerNorm测试
    // Input: [2, 4] = [1, 2, 3, 4, 5, 6, 7, 8]
    // Scale: [4] = [1.0, 1.0, 1.0, 1.0]
    // Bias: [4] = [0.0, 0.0, 0.0, 0.0]
    
    auto input = CreateTestTensor(Shape({2, 4}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    auto scale = CreateTestTensor(Shape({4}), {1.0f, 1.0f, 1.0f, 1.0f});
    auto bias = CreateTestTensor(Shape({4}), {0.0f, 0.0f, 0.0f, 0.0f});
    auto output = CreateTestTensor(Shape({2, 4}), {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    
    // 获取算子
    auto op = OperatorRegistry::Instance().Create("LayerNormalization");
    ASSERT_NE(op, nullptr);
    
    // 验证输入
    std::vector<Tensor*> inputs = {input.get(), scale.get(), bias.get()};
    std::vector<Tensor*> outputs = {output.get()};
    
    Status status = op->ValidateInputs(inputs);
    EXPECT_TRUE(status.IsOk());
    
    // 执行算子
    ExecutionContext ctx;
    status = op->Execute(inputs, outputs, &ctx);
    EXPECT_TRUE(status.IsOk());
    
    // 验证输出
    const float* output_data = static_cast<const float*>(output->GetData());
    for (size_t i = 0; i < output->GetElementCount(); ++i) {
        EXPECT_TRUE(std::isfinite(output_data[i]));
    }
    
    // LayerNorm后，每行的均值应该接近0（归一化后）
    // 第一行: [1,2,3,4] -> mean=2.5, 归一化后均值应该接近0
    float row1_mean = (output_data[0] + output_data[1] + output_data[2] + output_data[3]) / 4.0f;
    EXPECT_NEAR(row1_mean, 0.0f, 0.1f);  // 允许一定误差
}

// 测试 RMSNorm 基本功能
TEST_F(NormalizationOperatorsTest, RMSNormBasic) {
    // 创建简单的RMSNorm测试
    // Input: [2, 4] = [1, 2, 3, 4, 5, 6, 7, 8]
    // Scale: [4] = [1.0, 1.0, 1.0, 1.0]
    
    auto input = CreateTestTensor(Shape({2, 4}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    auto scale = CreateTestTensor(Shape({4}), {1.0f, 1.0f, 1.0f, 1.0f});
    auto output = CreateTestTensor(Shape({2, 4}), {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    
    // 获取算子
    auto op = OperatorRegistry::Instance().Create("RMSNorm");
    ASSERT_NE(op, nullptr);
    
    // 验证输入
    std::vector<Tensor*> inputs = {input.get(), scale.get()};
    std::vector<Tensor*> outputs = {output.get()};
    
    Status status = op->ValidateInputs(inputs);
    EXPECT_TRUE(status.IsOk());
    
    // 执行算子
    ExecutionContext ctx;
    status = op->Execute(inputs, outputs, &ctx);
    EXPECT_TRUE(status.IsOk());
    
    // 验证输出
    const float* output_data = static_cast<const float*>(output->GetData());
    for (size_t i = 0; i < output->GetElementCount(); ++i) {
        EXPECT_TRUE(std::isfinite(output_data[i]));
    }
    
    // RMSNorm后，每行的RMS应该接近1（归一化后）
    // 第一行: [1,2,3,4] -> RMS^2 = (1+4+9+16)/4 = 7.5, RMS = sqrt(7.5) ≈ 2.74
    // 归一化后每个元素应该除以RMS，所以输出的RMS应该接近1
    float row1_rms_sq = (output_data[0]*output_data[0] + output_data[1]*output_data[1] + 
                         output_data[2]*output_data[2] + output_data[3]*output_data[3]) / 4.0f;
    EXPECT_NEAR(row1_rms_sq, 1.0f, 0.2f);  // 允许一定误差
}

// 测试 BatchNormalization 输入验证
TEST_F(NormalizationOperatorsTest, BatchNormalizationInputValidation) {
    auto op = OperatorRegistry::Instance().Create("BatchNormalization");
    ASSERT_NE(op, nullptr);
    
    // 测试输入数量不足
    std::vector<Tensor*> empty_inputs;
    Status status = op->ValidateInputs(empty_inputs);
    EXPECT_FALSE(status.IsOk());
    
    // 测试输入数量正确
    auto input = CreateTestTensor(Shape({1, 2, 2, 2}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    auto scale = CreateTestTensor(Shape({2}), {1.0f, 1.0f});
    auto bias = CreateTestTensor(Shape({2}), {0.0f, 0.0f});
    auto mean = CreateTestTensor(Shape({2}), {2.5f, 6.5f});
    auto var = CreateTestTensor(Shape({2}), {1.25f, 1.25f});
    
    std::vector<Tensor*> inputs = {input.get(), scale.get(), bias.get(), mean.get(), var.get()};
    status = op->ValidateInputs(inputs);
    EXPECT_TRUE(status.IsOk());
}

// 测试 LayerNormalization 形状推断
TEST_F(NormalizationOperatorsTest, LayerNormalizationShapeInference) {
    auto op = OperatorRegistry::Instance().Create("LayerNormalization");
    ASSERT_NE(op, nullptr);
    
    auto input = CreateTestTensor(Shape({2, 4}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    auto scale = CreateTestTensor(Shape({4}), {1.0f, 1.0f, 1.0f, 1.0f});
    
    std::vector<Tensor*> inputs = {input.get(), scale.get()};
    std::vector<Shape> output_shapes;
    
    Status status = op->InferOutputShape(inputs, output_shapes);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0].dims, input->GetShape().dims);
}

// 测试 RMSNorm 形状推断
TEST_F(NormalizationOperatorsTest, RMSNormShapeInference) {
    auto op = OperatorRegistry::Instance().Create("RMSNorm");
    ASSERT_NE(op, nullptr);
    
    auto input = CreateTestTensor(Shape({2, 4}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    auto scale = CreateTestTensor(Shape({4}), {1.0f, 1.0f, 1.0f, 1.0f});
    
    std::vector<Tensor*> inputs = {input.get(), scale.get()};
    std::vector<Shape> output_shapes;
    
    Status status = op->InferOutputShape(inputs, output_shapes);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0].dims, input->GetShape().dims);
}

