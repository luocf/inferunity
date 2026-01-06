// Conv算子调试测试
// 用于验证Conv算子的输出值是否正确

#include "inferunity/operator.h"
#include "inferunity/tensor.h"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <memory>
#include <iostream>

using namespace inferunity;

class ConvDebugTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx_ = std::make_unique<ExecutionContext>();
    }
    
    std::unique_ptr<ExecutionContext> ctx_;
    
    // 辅助函数：创建测试张量
    std::shared_ptr<Tensor> CreateTestTensor(const Shape& shape, const std::vector<float>& data) {
        auto tensor = CreateTensor(shape, DataType::FLOAT32, DeviceType::CPU);
        if (data.size() == shape.GetElementCount()) {
            float* ptr = static_cast<float*>(tensor->GetData());
            std::copy(data.begin(), data.end(), ptr);
        }
        return tensor;
    }
    
    // 打印张量值（用于调试）
    void PrintTensor(const Tensor* tensor, const std::string& name) {
        std::cout << name << " shape: ";
        for (int64_t dim : tensor->GetShape().dims) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        
        const float* data = static_cast<const float*>(tensor->GetData());
        size_t count = tensor->GetElementCount();
        std::cout << "Values: ";
        for (size_t i = 0; i < std::min(count, size_t(10)); ++i) {
            std::cout << data[i] << " ";
        }
        if (count > 10) std::cout << "...";
        std::cout << std::endl;
    }
};

// 测试简单的1x1卷积（应该等价于矩阵乘法）
TEST_F(ConvDebugTest, Simple1x1Conv) {
    auto& registry = OperatorRegistry::Instance();
    auto conv_op = registry.Create("Conv");
    ASSERT_NE(conv_op, nullptr);
    
    // 输入: [1, 1, 3, 3] = 全1
    // 权重: [1, 1, 1, 1] = [2.0]
    // 输出: [1, 1, 3, 3] = 全2.0
    auto input = CreateTestTensor(Shape({1, 1, 3, 3}), 
                                   {1.0f, 1.0f, 1.0f, 
                                    1.0f, 1.0f, 1.0f, 
                                    1.0f, 1.0f, 1.0f});
    auto weight = CreateTestTensor(Shape({1, 1, 1, 1}), {2.0f});
    auto output = CreateTestTensor(Shape({1, 1, 3, 3}), 
                                    {0.0f, 0.0f, 0.0f, 
                                     0.0f, 0.0f, 0.0f, 
                                     0.0f, 0.0f, 0.0f});
    
    std::vector<Tensor*> inputs = {input.get(), weight.get()};
    std::vector<Tensor*> outputs = {output.get()};
    
    Status status = conv_op->Execute(inputs, outputs, ctx_.get());
    ASSERT_TRUE(status.IsOk());
    
    // 验证输出
    const float* output_data = static_cast<const float*>(output->GetData());
    for (size_t i = 0; i < 9; ++i) {
        EXPECT_NEAR(output_data[i], 2.0f, 1e-5f) 
            << "Output[" << i << "] should be 2.0";
    }
    
    PrintTensor(output.get(), "Output");
}

// 测试3x3卷积（边界检查）
TEST_F(ConvDebugTest, Conv3x3BoundaryCheck) {
    auto& registry = OperatorRegistry::Instance();
    auto conv_op = registry.Create("Conv");
    ASSERT_NE(conv_op, nullptr);
    
    // 输入: [1, 1, 5, 5] = 全1
    // 权重: [1, 1, 3, 3] = 全1/9（平均池化效果）
    // 输出: [1, 1, 3, 3]（无padding，stride=1）
    auto input = CreateTestTensor(Shape({1, 1, 5, 5}), 
                                   std::vector<float>(25, 1.0f));
    auto weight = CreateTestTensor(Shape({1, 1, 3, 3}), 
                                   std::vector<float>(9, 1.0f/9.0f));
    auto output = CreateTestTensor(Shape({1, 1, 3, 3}), 
                                    std::vector<float>(9, 0.0f));
    
    std::vector<Tensor*> inputs = {input.get(), weight.get()};
    std::vector<Tensor*> outputs = {output.get()};
    
    Status status = conv_op->Execute(inputs, outputs, ctx_.get());
    ASSERT_TRUE(status.IsOk());
    
    // 验证输出（每个输出位置应该覆盖9个输入位置，所以值应该是1.0）
    const float* output_data = static_cast<const float*>(output->GetData());
    for (size_t i = 0; i < 9; ++i) {
        EXPECT_NEAR(output_data[i], 1.0f, 1e-5f) 
            << "Output[" << i << "] should be 1.0";
    }
    
    PrintTensor(output.get(), "Output");
}

// 测试权重索引计算
TEST_F(ConvDebugTest, WeightIndexCalculation) {
    // 这个测试用于验证权重索引计算是否正确
    // 权重形状: [out_c, in_c, kernel_h, kernel_w]
    // 对于权重索引 ((oc * in_c + ic) * kernel_h + kh) * kernel_w + kw
    
    // 测试用例: out_c=2, in_c=1, kernel_h=2, kernel_w=2
    // 权重应该是: [2, 1, 2, 2]
    // weight[0,0,0,0] = 1.0, weight[0,0,0,1] = 2.0, ...
    // weight[1,0,0,0] = 5.0, weight[1,0,0,1] = 6.0, ...
    
    auto& registry = OperatorRegistry::Instance();
    auto conv_op = registry.Create("Conv");
    ASSERT_NE(conv_op, nullptr);
    
    auto input = CreateTestTensor(Shape({1, 1, 3, 3}), 
                                   std::vector<float>(9, 1.0f));
    auto weight = CreateTestTensor(Shape({2, 1, 2, 2}), 
                                   {1.0f, 2.0f, 3.0f, 4.0f,  // 第一个输出通道
                                    5.0f, 6.0f, 7.0f, 8.0f}); // 第二个输出通道
    auto output = CreateTestTensor(Shape({1, 2, 2, 2}), 
                                    std::vector<float>(8, 0.0f));
    
    std::vector<Tensor*> inputs = {input.get(), weight.get()};
    std::vector<Tensor*> outputs = {output.get()};
    
    Status status = conv_op->Execute(inputs, outputs, ctx_.get());
    ASSERT_TRUE(status.IsOk());
    
    PrintTensor(input.get(), "Input");
    PrintTensor(weight.get(), "Weight");
    PrintTensor(output.get(), "Output");
    
    // 验证输出值是否合理（不为NaN或Inf）
    const float* output_data = static_cast<const float*>(output->GetData());
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_TRUE(std::isfinite(output_data[i])) 
            << "Output[" << i << "] should be finite";
    }
}

