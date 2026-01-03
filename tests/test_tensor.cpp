// Tensor核心功能单元测试
// 测试Tensor的创建、序列化、切片等核心功能

#include <gtest/gtest.h>
#include "inferunity/tensor.h"
#include "inferunity/types.h"
#include <vector>
#include <cmath>
#include <cstring>

using namespace inferunity;

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
    
    void TearDown() override {
    }
    
    // 辅助函数：比较两个张量
    bool CompareTensors(const Tensor& a, const Tensor& b, float tolerance = 1e-5f) {
        const Shape& shape_a = a.GetShape();
        const Shape& shape_b = b.GetShape();
        
        if (shape_a.dims.size() != shape_b.dims.size()) return false;
        for (size_t i = 0; i < shape_a.dims.size(); ++i) {
            if (shape_a.dims[i] != shape_b.dims[i]) return false;
        }
        
        if (a.GetDataType() != b.GetDataType()) return false;
        
        size_t count = a.GetElementCount();
        const float* data_a = static_cast<const float*>(a.GetData());
        const float* data_b = static_cast<const float*>(b.GetData());
        
        for (size_t i = 0; i < count; ++i) {
            if (std::abs(data_a[i] - data_b[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }
};

// 测试Tensor创建
TEST_F(TensorTest, CreateTensor) {
    Shape shape({2, 3, 4});
    auto tensor = CreateTensor(shape, DataType::FLOAT32, DeviceType::CPU);
    
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->GetShape().dims.size(), 3);
    EXPECT_EQ(tensor->GetShape().dims[0], 2);
    EXPECT_EQ(tensor->GetShape().dims[1], 3);
    EXPECT_EQ(tensor->GetShape().dims[2], 4);
    EXPECT_EQ(tensor->GetDataType(), DataType::FLOAT32);
    EXPECT_EQ(tensor->GetDeviceType(), DeviceType::CPU);
    EXPECT_EQ(tensor->GetElementCount(), 24);
}

// 测试Tensor Reshape
TEST_F(TensorTest, Reshape) {
    Shape shape({2, 3, 4});
    auto tensor = CreateTensor(shape, DataType::FLOAT32, DeviceType::CPU);
    
    // 填充测试数据
    float* data = static_cast<float*>(tensor->GetData());
    for (size_t i = 0; i < 24; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    // Reshape到新形状
    Shape new_shape({6, 4});
    Tensor view = tensor->Reshape(new_shape);
    
    EXPECT_EQ(view.GetElementCount(), 24);
    EXPECT_EQ(view.GetShape().dims.size(), 2);
    EXPECT_EQ(view.GetShape().dims[0], 6);
    EXPECT_EQ(view.GetShape().dims[1], 4);
    
    // 验证数据共享（视图）
    const float* view_data = static_cast<const float*>(view.GetData());
    EXPECT_EQ(view_data, data);  // 应该指向同一块内存
}

// 测试Tensor Slice
TEST_F(TensorTest, Slice) {
    Shape shape({4, 4});
    auto tensor = CreateTensor(shape, DataType::FLOAT32, DeviceType::CPU);
    
    // 填充测试数据
    float* data = static_cast<float*>(tensor->GetData());
    for (size_t i = 0; i < 16; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    // 切片 [1:3, 1:3]
    std::vector<int64_t> starts = {1, 1};
    std::vector<int64_t> ends = {3, 3};
    Tensor slice = tensor->Slice(starts, ends);
    
    EXPECT_EQ(slice.GetElementCount(), 4);
    EXPECT_EQ(slice.GetShape().dims.size(), 2);
    EXPECT_EQ(slice.GetShape().dims[0], 2);
    EXPECT_EQ(slice.GetShape().dims[1], 2);
    
    // 验证切片数据（应该是原始数据的子集）
    const float* slice_data = static_cast<const float*>(slice.GetData());
    // 原始数据索引：1*4+1=5, 1*4+2=6, 2*4+1=9, 2*4+2=10
    // 切片应该是 [5, 6, 9, 10]
    EXPECT_FLOAT_EQ(slice_data[0], 5.0f);
    EXPECT_FLOAT_EQ(slice_data[1], 6.0f);
    EXPECT_FLOAT_EQ(slice_data[2], 9.0f);
    EXPECT_FLOAT_EQ(slice_data[3], 10.0f);
}

// 测试Tensor序列化/反序列化
TEST_F(TensorTest, SerializeDeserialize) {
    Shape shape({2, 3});
    auto tensor = CreateTensor(shape, DataType::FLOAT32, DeviceType::CPU);
    
    // 填充测试数据
    float* data = static_cast<float*>(tensor->GetData());
    for (size_t i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i + 1);
    }
    
    // 序列化
    std::vector<uint8_t> buffer;
    Status status = tensor->Serialize(buffer);
    EXPECT_TRUE(status.IsOk());
    EXPECT_GT(buffer.size(), 0);
    
    // 反序列化
    auto new_tensor = CreateTensor(Shape({1}), DataType::FLOAT32, DeviceType::CPU);
    status = new_tensor->Deserialize(buffer);
    EXPECT_TRUE(status.IsOk());
    
    // 验证数据
    EXPECT_TRUE(CompareTensors(*tensor, *new_tensor));
}

// 测试Tensor CopyTo
TEST_F(TensorTest, CopyTo) {
    Shape shape({2, 3});
    auto src = CreateTensor(shape, DataType::FLOAT32, DeviceType::CPU);
    auto dst = CreateTensor(shape, DataType::FLOAT32, DeviceType::CPU);
    
    // 填充源数据
    float* src_data = static_cast<float*>(src->GetData());
    for (size_t i = 0; i < 6; ++i) {
        src_data[i] = static_cast<float>(i);
    }
    
    // 拷贝
    Status status = src->CopyTo(*dst);
    EXPECT_TRUE(status.IsOk());
    
    // 验证数据
    EXPECT_TRUE(CompareTensors(*src, *dst));
}

// 测试Tensor FillZero
TEST_F(TensorTest, FillZero) {
    Shape shape({2, 3});
    auto tensor = CreateTensor(shape, DataType::FLOAT32, DeviceType::CPU);
    
    // 先填充非零数据
    tensor->FillValue(1.0f);
    
    // 清零
    Status status = tensor->FillZero();
    EXPECT_TRUE(status.IsOk());
    
    // 验证所有元素都是0
    const float* data = static_cast<const float*>(tensor->GetData());
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], 0.0f);
    }
}

// 测试Tensor FillValue
TEST_F(TensorTest, FillValue) {
    Shape shape({2, 3});
    auto tensor = CreateTensor(shape, DataType::FLOAT32, DeviceType::CPU);
    
    // 填充值
    Status status = tensor->FillValue(3.14f);
    EXPECT_TRUE(status.IsOk());
    
    // 验证所有元素都是3.14
    const float* data = static_cast<const float*>(tensor->GetData());
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], 3.14f);
    }
}

// 测试Tensor负数索引切片
TEST_F(TensorTest, SliceNegativeIndices) {
    Shape shape({4, 4});
    auto tensor = CreateTensor(shape, DataType::FLOAT32, DeviceType::CPU);
    
    float* data = static_cast<float*>(tensor->GetData());
    for (size_t i = 0; i < 16; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    // 使用负数索引切片 [-2:-1, -2:-1]
    std::vector<int64_t> starts = {-2, -2};
    std::vector<int64_t> ends = {-1, -1};
    Tensor slice = tensor->Slice(starts, ends);
    
    EXPECT_EQ(slice.GetElementCount(), 1);
    // 应该是索引 [2, 2] 的值，即 2*4+2=10
    const float* slice_data = static_cast<const float*>(slice.GetData());
    EXPECT_FLOAT_EQ(slice_data[0], 10.0f);
}

