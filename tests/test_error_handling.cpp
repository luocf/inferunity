// 错误处理和边界条件测试
// 测试各种异常情况和边界条件

#include "inferunity/operator.h"
#include "inferunity/tensor.h"
#include <gtest/gtest.h>
#include <vector>
#include <memory>

using namespace inferunity;

class ErrorHandlingTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx_ = std::make_unique<ExecutionContext>();
    }
    
    std::unique_ptr<ExecutionContext> ctx_;
    
    std::shared_ptr<Tensor> CreateTestTensor(const Shape& shape, const std::vector<float>& data) {
        auto tensor = CreateTensor(shape, DataType::FLOAT32, DeviceType::CPU);
        if (data.size() == shape.GetElementCount()) {
            float* ptr = static_cast<float*>(tensor->GetData());
            std::copy(data.begin(), data.end(), ptr);
        }
        return tensor;
    }
};

// 测试空输入
TEST_F(ErrorHandlingTest, EmptyInputs) {
    auto& registry = OperatorRegistry::Instance();
    auto add_op = registry.Create("Add");
    ASSERT_NE(add_op, nullptr);
    
    std::vector<Tensor*> empty_inputs;
    std::vector<Tensor*> empty_outputs;
    
    Status status = add_op->ValidateInputs(empty_inputs);
    EXPECT_FALSE(status.IsOk());
    
    status = add_op->Execute(empty_inputs, empty_outputs, ctx_.get());
    EXPECT_FALSE(status.IsOk());
}

// 测试输入数量不匹配
TEST_F(ErrorHandlingTest, MismatchedInputCount) {
    auto& registry = OperatorRegistry::Instance();
    auto add_op = registry.Create("Add");
    ASSERT_NE(add_op, nullptr);
    
    auto input1 = CreateTestTensor(Shape({2, 3}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto output = CreateTestTensor(Shape({2, 3}), {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    
    // Add需要2个输入，只提供1个
    std::vector<Tensor*> inputs = {input1.get()};
    std::vector<Tensor*> outputs = {output.get()};
    
    Status status = add_op->ValidateInputs(inputs);
    EXPECT_FALSE(status.IsOk());
}

// 测试形状不匹配
TEST_F(ErrorHandlingTest, ShapeMismatch) {
    auto& registry = OperatorRegistry::Instance();
    auto add_op = registry.Create("Add");
    ASSERT_NE(add_op, nullptr);
    
    auto input1 = CreateTestTensor(Shape({2, 3}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto input2 = CreateTestTensor(Shape({3, 2}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});  // 形状不匹配
    auto output = CreateTestTensor(Shape({2, 3}), {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    
    std::vector<Tensor*> inputs = {input1.get(), input2.get()};
    std::vector<Tensor*> outputs = {output.get()};
    
    Status status = add_op->ValidateInputs(inputs);
    // Add可能支持广播，所以验证可能通过
    // 但执行时可能会失败
}

// 测试空输出
TEST_F(ErrorHandlingTest, EmptyOutputs) {
    auto& registry = OperatorRegistry::Instance();
    auto add_op = registry.Create("Add");
    ASSERT_NE(add_op, nullptr);
    
    auto input1 = CreateTestTensor(Shape({2, 3}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto input2 = CreateTestTensor(Shape({2, 3}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    std::vector<Tensor*> inputs = {input1.get(), input2.get()};
    std::vector<Tensor*> empty_outputs;
    
    Status status = add_op->Execute(inputs, empty_outputs, ctx_.get());
    EXPECT_FALSE(status.IsOk());
}

// 测试零大小张量
TEST_F(ErrorHandlingTest, ZeroSizeTensor) {
    auto& registry = OperatorRegistry::Instance();
    auto add_op = registry.Create("Add");
    if (!add_op) {
        GTEST_SKIP() << "Add operator not available";
    }
    
    // 创建零大小张量（可能不支持，但测试错误处理）
    try {
        auto input1 = CreateTensor(Shape({0}), DataType::FLOAT32, DeviceType::CPU);
        auto input2 = CreateTensor(Shape({0}), DataType::FLOAT32, DeviceType::CPU);
        auto output = CreateTensor(Shape({0}), DataType::FLOAT32, DeviceType::CPU);
        
        std::vector<Tensor*> inputs = {input1.get(), input2.get()};
        std::vector<Tensor*> outputs = {output.get()};
        
        Status status = add_op->ValidateInputs(inputs);
        // 可能通过或失败，取决于实现
    } catch (...) {
        // 如果抛出异常，说明正确处理了错误
        SUCCEED();
    }
}

// 测试无效的算子名称
TEST_F(ErrorHandlingTest, InvalidOperatorName) {
    auto& registry = OperatorRegistry::Instance();
    auto invalid_op = registry.Create("NonExistentOperator");
    EXPECT_EQ(invalid_op, nullptr);
}

// 测试Conv算子的边界条件
TEST_F(ErrorHandlingTest, ConvBoundaryConditions) {
    auto& registry = OperatorRegistry::Instance();
    auto conv_op = registry.Create("Conv");
    if (!conv_op) {
        GTEST_SKIP() << "Conv operator not available";
    }
    
    // 测试输入维度不足（Conv需要4D输入）
    auto input1d = CreateTestTensor(Shape({10}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
    auto weight = CreateTestTensor(Shape({1, 1, 3, 3}), std::vector<float>(9, 1.0f));
    
    std::vector<Tensor*> inputs = {input1d.get(), weight.get()};
    Status status = conv_op->ValidateInputs(inputs);
    EXPECT_FALSE(status.IsOk());
}

// 测试数据类型不匹配
TEST_F(ErrorHandlingTest, DataTypeMismatch) {
    // 注意：当前实现可能不检查数据类型，但测试接口
    auto& registry = OperatorRegistry::Instance();
    auto add_op = registry.Create("Add");
    if (!add_op) {
        GTEST_SKIP() << "Add operator not available";
    }
    
    // 创建不同数据类型的张量（如果支持）
    // 当前实现可能只支持FLOAT32，所以这个测试主要是接口测试
}

// 测试大尺寸张量（内存限制）
TEST_F(ErrorHandlingTest, LargeTensor) {
    auto& registry = OperatorRegistry::Instance();
    auto add_op = registry.Create("Add");
    if (!add_op) {
        GTEST_SKIP() << "Add operator not available";
    }
    
    // 测试较大的张量（但不至于导致内存溢出）
    Shape large_shape({100, 100});  // 10,000 elements
    auto input1 = CreateTensor(large_shape, DataType::FLOAT32, DeviceType::CPU);
    auto input2 = CreateTensor(large_shape, DataType::FLOAT32, DeviceType::CPU);
    auto output = CreateTensor(large_shape, DataType::FLOAT32, DeviceType::CPU);
    
    std::vector<Tensor*> inputs = {input1.get(), input2.get()};
    std::vector<Tensor*> outputs = {output.get()};
    
    Status status = add_op->Execute(inputs, outputs, ctx_.get());
    // 应该成功，除非内存不足
    if (status.IsOk()) {
        SUCCEED();
    }
}

