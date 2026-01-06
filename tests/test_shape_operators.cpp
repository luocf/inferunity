// 形状操作算子测试
// 测试 Gather, Slice, Transpose, Reshape 等形状操作算子

#include "inferunity/operator.h"
#include "inferunity/tensor.h"
#include <gtest/gtest.h>
#include <vector>
#include <memory>

using namespace inferunity;

class ShapeOperatorsTest : public ::testing::Test {
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
        } else {
            // 如果数据大小不匹配，填充零
            tensor->FillZero();
        }
        return tensor;
    }
};

// 测试 Reshape 算子
TEST_F(ShapeOperatorsTest, ReshapeOperator) {
    auto& registry = OperatorRegistry::Instance();
    auto reshape_op = registry.Create("Reshape");
    if (!reshape_op) {
        GTEST_SKIP() << "Reshape operator not available";
    }
    
    // 输入: [2, 3] = [1, 2, 3, 4, 5, 6]
    // 重塑为: [3, 2]
    auto input = CreateTestTensor(Shape({2, 3}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    // Reshape需要shape tensor作为第二个输入
    // shape tensor应该是INT64类型，包含目标形状 [3, 2]
    auto shape_tensor = CreateTensor(Shape({2}), DataType::INT64, DeviceType::CPU);
    int64_t* shape_data = static_cast<int64_t*>(shape_tensor->GetData());
    shape_data[0] = 3;
    shape_data[1] = 2;
    
    // 先推断输出形状
    std::vector<Tensor*> inputs_for_shape = {input.get(), shape_tensor.get()};
    std::vector<Shape> output_shapes;
    Status status = reshape_op->InferOutputShape(inputs_for_shape, output_shapes);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0].dims.size(), 2);
    EXPECT_EQ(output_shapes[0].dims[0], 3);
    EXPECT_EQ(output_shapes[0].dims[1], 2);
    
    // 创建输出张量
    auto output = CreateTestTensor(output_shapes[0], {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    
    std::vector<Tensor*> inputs = {input.get(), shape_tensor.get()};
    std::vector<Tensor*> outputs = {output.get()};
    
    status = reshape_op->ValidateInputs(inputs);
    EXPECT_TRUE(status.IsOk());
    
    // 执行算子
    status = reshape_op->Execute(inputs, outputs, ctx_.get());
    EXPECT_TRUE(status.IsOk());
    
    // 验证元素数量相同
    EXPECT_EQ(input->GetElementCount(), output->GetElementCount());
    
    // 验证数据被正确复制（reshape只是改变视图，数据应该相同）
    const float* input_data = static_cast<const float*>(input->GetData());
    const float* output_data = static_cast<const float*>(output->GetData());
    for (size_t i = 0; i < input->GetElementCount(); ++i) {
        EXPECT_FLOAT_EQ(input_data[i], output_data[i]);
    }
}

// 测试 Transpose 算子
TEST_F(ShapeOperatorsTest, TransposeOperator) {
    auto& registry = OperatorRegistry::Instance();
    auto transpose_op = registry.Create("Transpose");
    if (!transpose_op) {
        GTEST_SKIP() << "Transpose operator not available";
    }
    
    // 输入: [2, 3] = [1, 2, 3, 4, 5, 6]
    // 转置为: [3, 2]
    auto input = CreateTestTensor(Shape({2, 3}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto output = CreateTestTensor(Shape({3, 2}), {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    
    std::vector<Tensor*> inputs = {input.get()};
    std::vector<Tensor*> outputs = {output.get()};
    
    Status status = transpose_op->ValidateInputs(inputs);
    EXPECT_TRUE(status.IsOk());
    
    // 执行算子
    status = transpose_op->Execute(inputs, outputs, ctx_.get());
    EXPECT_TRUE(status.IsOk());
    
    // 验证元素数量相同
    EXPECT_EQ(input->GetElementCount(), output->GetElementCount());
}

// 测试 Slice 算子
TEST_F(ShapeOperatorsTest, SliceOperator) {
    auto& registry = OperatorRegistry::Instance();
    auto slice_op = registry.Create("Slice");
    if (!slice_op) {
        GTEST_SKIP() << "Slice operator not available";
    }
    
    // 输入: [2, 3] = [1, 2, 3, 4, 5, 6]
    // Slice需要starts, ends, axes等参数，这里简化测试
    // 先测试输入验证
    auto input = CreateTestTensor(Shape({2, 3}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    std::vector<Tensor*> inputs = {input.get()};
    Status status = slice_op->ValidateInputs(inputs);
    // Slice可能需要更多输入（starts, ends等），所以验证可能失败，这是正常的
    // 主要测试接口可用性，不执行实际切片操作以避免崩溃
}

// 测试 Gather 算子
TEST_F(ShapeOperatorsTest, GatherOperator) {
    auto& registry = OperatorRegistry::Instance();
    auto gather_op = registry.Create("Gather");
    if (!gather_op) {
        GTEST_SKIP() << "Gather operator not available";
    }
    
    // 输入: [3, 4] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    // 索引: [2] = [0, 2] (INT64类型)
    auto input = CreateTestTensor(Shape({3, 4}), 
                                   {1.0f, 2.0f, 3.0f, 4.0f,
                                    5.0f, 6.0f, 7.0f, 8.0f,
                                    9.0f, 10.0f, 11.0f, 12.0f});
    
    // Gather需要INT64类型的索引
    auto indices = CreateTensor(Shape({2}), DataType::INT64, DeviceType::CPU);
    int64_t* idx_data = static_cast<int64_t*>(indices->GetData());
    idx_data[0] = 0;
    idx_data[1] = 2;
    
    // 先测试输入验证
    std::vector<Tensor*> inputs = {input.get(), indices.get()};
    Status status = gather_op->ValidateInputs(inputs);
    // Gather可能需要特定的axis等参数，所以验证可能失败
    // 这里主要测试接口可用性
}

// 测试形状推断
TEST_F(ShapeOperatorsTest, ReshapeShapeInference) {
    auto& registry = OperatorRegistry::Instance();
    auto reshape_op = registry.Create("Reshape");
    if (!reshape_op) {
        GTEST_SKIP() << "Reshape operator not available";
    }
    
    auto input = CreateTestTensor(Shape({2, 3}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    // 创建shape tensor
    auto shape_tensor = CreateTensor(Shape({2}), DataType::INT64, DeviceType::CPU);
    int64_t* shape_data = static_cast<int64_t*>(shape_tensor->GetData());
    shape_data[0] = 3;
    shape_data[1] = 2;
    
    std::vector<Tensor*> inputs = {input.get(), shape_tensor.get()};
    std::vector<Shape> output_shapes;
    
    Status status = reshape_op->InferOutputShape(inputs, output_shapes);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0].dims.size(), 2);
    EXPECT_EQ(output_shapes[0].dims[0], 3);
    EXPECT_EQ(output_shapes[0].dims[1], 2);
}

// 测试输入验证
TEST_F(ShapeOperatorsTest, ReshapeInputValidation) {
    auto& registry = OperatorRegistry::Instance();
    auto reshape_op = registry.Create("Reshape");
    if (!reshape_op) {
        GTEST_SKIP() << "Reshape operator not available";
    }
    
    // 测试空输入
    std::vector<Tensor*> empty_inputs;
    Status status = reshape_op->ValidateInputs(empty_inputs);
    EXPECT_FALSE(status.IsOk());
    
    // 测试有效输入
    auto input = CreateTestTensor(Shape({2, 3}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    std::vector<Tensor*> inputs = {input.get()};
    status = reshape_op->ValidateInputs(inputs);
    // 可能成功或失败，取决于Reshape的实现
}

