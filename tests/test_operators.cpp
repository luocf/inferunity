// 算子单元测试
// 测试所有基础算子的功能

#include <gtest/gtest.h>
#include "inferunity/operator.h"
#include "inferunity/tensor.h"
#include "inferunity/types.h"
#include <vector>
#include <cmath>
#include <cstring>

using namespace inferunity;

class OperatorsTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
    
    void TearDown() override {
    }
    
    std::shared_ptr<Tensor> CreateTensor(const Shape& shape, DataType dtype,
                                        const std::vector<float>& data = {}) {
        auto tensor = inferunity::CreateTensor(shape, dtype, DeviceType::CPU);
        if (!data.empty() && tensor) {
            float* ptr = static_cast<float*>(tensor->GetData());
            size_t count = std::min(data.size(), tensor->GetElementCount());
            std::memcpy(ptr, data.data(), count * sizeof(float));
        }
        return tensor;
    }
    
    bool CompareFloat(float a, float b, float tolerance = 1e-5f) {
        return std::abs(a - b) < tolerance;
    }
};

// 测试Reshape算子
TEST_F(OperatorsTest, ReshapeOperator) {
    auto& registry = OperatorRegistry::Instance();
    auto op = registry.Create("Reshape");
    ASSERT_NE(op, nullptr);
    
    // 创建输入
    auto input = CreateTensor(Shape({2, 3, 4}), DataType::FLOAT32);
    auto shape_tensor = CreateTensor(Shape({3}), DataType::INT64);
    
    // 设置目标形状 [24]
    int64_t* shape_data = static_cast<int64_t*>(shape_tensor->GetData());
    shape_data[0] = 24;
    
    std::vector<Tensor*> inputs = {input.get(), shape_tensor.get()};
    std::vector<Shape> output_shapes;
    
    Status status = op->InferOutputShape(inputs, output_shapes);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0].dims.size(), 1);
    EXPECT_EQ(output_shapes[0].dims[0], 24);
}

// 测试Concat算子
TEST_F(OperatorsTest, ConcatOperator) {
    auto& registry = OperatorRegistry::Instance();
    auto op = registry.Create("Concat");
    ASSERT_NE(op, nullptr);
    
    // 设置axis属性
    op->SetAttribute("axis", AttributeValue(static_cast<int64_t>(0)));
    
    // 创建输入
    auto input1 = CreateTensor(Shape({2, 3}), DataType::FLOAT32);
    auto input2 = CreateTensor(Shape({2, 3}), DataType::FLOAT32);
    
    std::vector<Tensor*> inputs = {input1.get(), input2.get()};
    std::vector<Shape> output_shapes;
    
    Status status = op->InferOutputShape(inputs, output_shapes);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0].dims[0], 4);  // 2+2=4
    EXPECT_EQ(output_shapes[0].dims[1], 3);
}

// 测试Split算子
TEST_F(OperatorsTest, SplitOperator) {
    auto& registry = OperatorRegistry::Instance();
    auto op = registry.Create("Split");
    ASSERT_NE(op, nullptr);
    
    // 设置属性
    op->SetAttribute("axis", AttributeValue(static_cast<int64_t>(0)));
    op->SetAttribute("split", AttributeValue(std::vector<int64_t>{1, 1}));
    
    // 创建输入
    auto input = CreateTensor(Shape({2, 3}), DataType::FLOAT32);
    
    std::vector<Tensor*> inputs = {input.get()};
    std::vector<Shape> output_shapes;
    
    Status status = op->InferOutputShape(inputs, output_shapes);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(output_shapes.size(), 2);
    EXPECT_EQ(output_shapes[0].dims[0], 1);
    EXPECT_EQ(output_shapes[1].dims[0], 1);
}

// 测试Transpose算子
TEST_F(OperatorsTest, TransposeOperator) {
    auto& registry = OperatorRegistry::Instance();
    auto op = registry.Create("Transpose");
    ASSERT_NE(op, nullptr);
    
    // 设置perm属性
    op->SetAttribute("perm", AttributeValue(std::vector<int64_t>{1, 0}));
    
    // 创建输入
    auto input = CreateTensor(Shape({2, 3}), DataType::FLOAT32);
    
    std::vector<Tensor*> inputs = {input.get()};
    std::vector<Shape> output_shapes;
    
    Status status = op->InferOutputShape(inputs, output_shapes);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0].dims[0], 3);
    EXPECT_EQ(output_shapes[0].dims[1], 2);
}

// 测试Add算子
TEST_F(OperatorsTest, AddOperator) {
    auto& registry = OperatorRegistry::Instance();
    auto op = registry.Create("Add");
    ASSERT_NE(op, nullptr);
    
    // 创建输入
    auto input1 = CreateTensor(Shape({2, 3}), DataType::FLOAT32, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto input2 = CreateTensor(Shape({2, 3}), DataType::FLOAT32, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    auto output = CreateTensor(Shape({2, 3}), DataType::FLOAT32);
    
    std::vector<Tensor*> inputs = {input1.get(), input2.get()};
    std::vector<Tensor*> outputs = {output.get()};
    
    ExecutionContext ctx;
    Status status = op->Execute(inputs, outputs, &ctx);
    EXPECT_TRUE(status.IsOk());
    
    // 验证输出
    const float* out_data = static_cast<const float*>(output->GetData());
    EXPECT_TRUE(CompareFloat(out_data[0], 2.0f));
    EXPECT_TRUE(CompareFloat(out_data[1], 3.0f));
    EXPECT_TRUE(CompareFloat(out_data[2], 4.0f));
}

// 测试Relu算子
TEST_F(OperatorsTest, ReluOperator) {
    auto& registry = OperatorRegistry::Instance();
    auto op = registry.Create("Relu");
    ASSERT_NE(op, nullptr);
    
    // 创建输入（包含负数）
    auto input = CreateTensor(Shape({6}), DataType::FLOAT32, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    auto output = CreateTensor(Shape({6}), DataType::FLOAT32);
    
    std::vector<Tensor*> inputs = {input.get()};
    std::vector<Tensor*> outputs = {output.get()};
    
    ExecutionContext ctx;
    Status status = op->Execute(inputs, outputs, &ctx);
    EXPECT_TRUE(status.IsOk());
    
    // 验证输出（所有值应该>=0）
    const float* out_data = static_cast<const float*>(output->GetData());
    EXPECT_FLOAT_EQ(out_data[0], 0.0f);  // -2 -> 0
    EXPECT_FLOAT_EQ(out_data[1], 0.0f);  // -1 -> 0
    EXPECT_FLOAT_EQ(out_data[2], 0.0f);  // 0 -> 0
    EXPECT_FLOAT_EQ(out_data[3], 1.0f);  // 1 -> 1
    EXPECT_FLOAT_EQ(out_data[4], 2.0f);  // 2 -> 2
    EXPECT_FLOAT_EQ(out_data[5], 3.0f);  // 3 -> 3
}

// 测试MatMul算子
TEST_F(OperatorsTest, MatMulOperator) {
    auto& registry = OperatorRegistry::Instance();
    auto op = registry.Create("MatMul");
    ASSERT_NE(op, nullptr);
    
    // 创建输入矩阵 A(2x3) * B(3x2) = C(2x2)
    auto input1 = CreateTensor(Shape({2, 3}), DataType::FLOAT32, 
                              {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto input2 = CreateTensor(Shape({3, 2}), DataType::FLOAT32,
                              {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f});
    auto output = CreateTensor(Shape({2, 2}), DataType::FLOAT32);
    
    std::vector<Tensor*> inputs = {input1.get(), input2.get()};
    std::vector<Tensor*> outputs = {output.get()};
    
    ExecutionContext ctx;
    Status status = op->Execute(inputs, outputs, &ctx);
    EXPECT_TRUE(status.IsOk());
    
    // 验证输出（简化验证）
    const float* out_data = static_cast<const float*>(output->GetData());
    EXPECT_GT(out_data[0], 0.0f);
}

