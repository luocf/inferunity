// 数学算子测试
#include <gtest/gtest.h>
#include "inferunity/tensor.h"
#include "inferunity/operator.h"
#include "inferunity/types.h"
#include <vector>
#include <cmath>

using namespace inferunity;

class MathOperatorsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化执行上下文
        ctx_ = std::make_unique<ExecutionContext>();
    }
    
    std::unique_ptr<ExecutionContext> ctx_;
};

// Add算子测试
TEST_F(MathOperatorsTest, AddOperator) {
    auto& registry = OperatorRegistry::Instance();
    auto add_op = registry.Create("Add");
    ASSERT_NE(add_op, nullptr);
    
    // 创建输入张量
    Shape shape({2, 3});
    auto input1 = CreateTensor(shape, DataType::FLOAT32);
    auto input2 = CreateTensor(shape, DataType::FLOAT32);
    auto output = CreateTensor(shape, DataType::FLOAT32);
    
    // 填充数据
    float* data1 = static_cast<float*>(input1->GetData());
    float* data2 = static_cast<float*>(input2->GetData());
    for (int i = 0; i < 6; ++i) {
        data1[i] = static_cast<float>(i);
        data2[i] = static_cast<float>(i * 2);
    }
    
    // 执行
    std::vector<Tensor*> inputs = {input1.get(), input2.get()};
    std::vector<Tensor*> outputs = {output.get()};
    Status status = add_op->Execute(inputs, outputs, ctx_.get());
    
    ASSERT_TRUE(status.IsOk());
    
    // 验证结果
    float* out_data = static_cast<float*>(output->GetData());
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(out_data[i], static_cast<float>(i * 3));
    }
}

// Mul算子测试
TEST_F(MathOperatorsTest, MulOperator) {
    auto& registry = OperatorRegistry::Instance();
    auto mul_op = registry.Create("Mul");
    ASSERT_NE(mul_op, nullptr);
    
    Shape shape({2, 2});
    auto input1 = CreateTensor(shape, DataType::FLOAT32);
    auto input2 = CreateTensor(shape, DataType::FLOAT32);
    auto output = CreateTensor(shape, DataType::FLOAT32);
    
    float* data1 = static_cast<float*>(input1->GetData());
    float* data2 = static_cast<float*>(input2->GetData());
    data1[0] = 2.0f; data1[1] = 3.0f;
    data1[2] = 4.0f; data1[3] = 5.0f;
    data2[0] = 1.0f; data2[1] = 2.0f;
    data2[2] = 3.0f; data2[3] = 4.0f;
    
    std::vector<Tensor*> inputs = {input1.get(), input2.get()};
    std::vector<Tensor*> outputs = {output.get()};
    Status status = mul_op->Execute(inputs, outputs, ctx_.get());
    
    ASSERT_TRUE(status.IsOk());
    
    float* out_data = static_cast<float*>(output->GetData());
    EXPECT_FLOAT_EQ(out_data[0], 2.0f);
    EXPECT_FLOAT_EQ(out_data[1], 6.0f);
    EXPECT_FLOAT_EQ(out_data[2], 12.0f);
    EXPECT_FLOAT_EQ(out_data[3], 20.0f);
}

// Sub算子测试
TEST_F(MathOperatorsTest, SubOperator) {
    auto& registry = OperatorRegistry::Instance();
    auto sub_op = registry.Create("Sub");
    ASSERT_NE(sub_op, nullptr);
    
    Shape shape({3});
    auto input1 = CreateTensor(shape, DataType::FLOAT32);
    auto input2 = CreateTensor(shape, DataType::FLOAT32);
    auto output = CreateTensor(shape, DataType::FLOAT32);
    
    float* data1 = static_cast<float*>(input1->GetData());
    float* data2 = static_cast<float*>(input2->GetData());
    data1[0] = 10.0f; data1[1] = 20.0f; data1[2] = 30.0f;
    data2[0] = 3.0f; data2[1] = 7.0f; data2[2] = 15.0f;
    
    std::vector<Tensor*> inputs = {input1.get(), input2.get()};
    std::vector<Tensor*> outputs = {output.get()};
    Status status = sub_op->Execute(inputs, outputs, ctx_.get());
    
    ASSERT_TRUE(status.IsOk());
    
    float* out_data = static_cast<float*>(output->GetData());
    EXPECT_FLOAT_EQ(out_data[0], 7.0f);
    EXPECT_FLOAT_EQ(out_data[1], 13.0f);
    EXPECT_FLOAT_EQ(out_data[2], 15.0f);
}

// Div算子测试
TEST_F(MathOperatorsTest, DivOperator) {
    auto& registry = OperatorRegistry::Instance();
    auto div_op = registry.Create("Div");
    ASSERT_NE(div_op, nullptr);
    
    Shape shape({4});
    auto input1 = CreateTensor(shape, DataType::FLOAT32);
    auto input2 = CreateTensor(shape, DataType::FLOAT32);
    auto output = CreateTensor(shape, DataType::FLOAT32);
    
    float* data1 = static_cast<float*>(input1->GetData());
    float* data2 = static_cast<float*>(input2->GetData());
    data1[0] = 10.0f; data1[1] = 20.0f; data1[2] = 30.0f; data1[3] = 40.0f;
    data2[0] = 2.0f; data2[1] = 4.0f; data2[2] = 5.0f; data2[3] = 8.0f;
    
    std::vector<Tensor*> inputs = {input1.get(), input2.get()};
    std::vector<Tensor*> outputs = {output.get()};
    Status status = div_op->Execute(inputs, outputs, ctx_.get());
    
    ASSERT_TRUE(status.IsOk());
    
    float* out_data = static_cast<float*>(output->GetData());
    EXPECT_FLOAT_EQ(out_data[0], 5.0f);
    EXPECT_FLOAT_EQ(out_data[1], 5.0f);
    EXPECT_FLOAT_EQ(out_data[2], 6.0f);
    EXPECT_FLOAT_EQ(out_data[3], 5.0f);
}

// 形状推断测试
TEST_F(MathOperatorsTest, AddShapeInference) {
    auto& registry = OperatorRegistry::Instance();
    auto add_op = registry.Create("Add");
    ASSERT_NE(add_op, nullptr);
    
    Shape shape1({2, 3});
    Shape shape2({2, 3});
    auto input1 = CreateTensor(shape1, DataType::FLOAT32);
    auto input2 = CreateTensor(shape2, DataType::FLOAT32);
    
    std::vector<Tensor*> inputs = {input1.get(), input2.get()};
    std::vector<Shape> output_shapes;
    
    Status status = add_op->InferOutputShape(inputs, output_shapes);
    ASSERT_TRUE(status.IsOk());
    ASSERT_EQ(output_shapes.size(), 1);
    ASSERT_EQ(output_shapes[0].dims.size(), 2);
    EXPECT_EQ(output_shapes[0].dims[0], 2);
    EXPECT_EQ(output_shapes[0].dims[1], 3);
}

// 输入验证测试
TEST_F(MathOperatorsTest, AddInputValidation) {
    auto& registry = OperatorRegistry::Instance();
    auto add_op = registry.Create("Add");
    ASSERT_NE(add_op, nullptr);
    
    // 测试输入数量不足
    Shape shape({2, 3});
    auto input1 = CreateTensor(shape, DataType::FLOAT32);
    std::vector<Tensor*> inputs = {input1.get()};
    
    Status status = add_op->ValidateInputs(inputs);
    ASSERT_FALSE(status.IsOk());
}

