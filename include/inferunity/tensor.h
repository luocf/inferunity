#pragma once

#include "types.h"
#include <memory>
#include <cstring>

namespace inferunity {

// 前向声明
class MemoryAllocator;
class Device;

// 张量类 - 核心数据结构
class Tensor {
public:
    Tensor();
    Tensor(const Shape& shape, DataType dtype, DeviceType device = DeviceType::CPU);
    Tensor(const Shape& shape, DataType dtype, void* data, 
           MemoryLayout layout = MemoryLayout::NCHW, DeviceType device = DeviceType::CPU);
    ~Tensor();
    
    // 禁止拷贝，允许移动
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;
    
    // 基本信息
    const Shape& GetShape() const { return shape_; }
    DataType GetDataType() const { return dtype_; }
    DeviceType GetDeviceType() const { return device_type_; }
    MemoryLayout GetLayout() const { return layout_; }
    
    // 数据访问
    void* GetData() { return data_; }
    const void* GetData() const { return data_; }
    size_t GetSizeInBytes() const;
    size_t GetElementCount() const { return shape_.GetElementCount(); }
    static size_t GetDataTypeSize(DataType dtype);
    
    // 内存管理
    bool IsOwned() const { return owns_data_; }
    void SetData(void* data, bool owns = false);
    
    // 形状操作（视图，不拷贝数据）
    Tensor Reshape(const Shape& new_shape);
    Tensor Slice(const std::vector<int64_t>& starts, const std::vector<int64_t>& ends);
    
    // 设备间传输
    Status CopyTo(Tensor& dst) const;
    Status CopyFrom(const Tensor& src);
    
    // 数据填充
    Status FillZero();
    Status FillValue(float value);
    
    // 序列化
    Status Serialize(std::vector<uint8_t>& buffer) const;
    Status Deserialize(const std::vector<uint8_t>& buffer);
    
private:
    Shape shape_;
    DataType dtype_;
    DeviceType device_type_;
    MemoryLayout layout_;
    void* data_;
    bool owns_data_;
    std::shared_ptr<MemoryAllocator> allocator_;
    
    void AllocateMemory();
    void FreeMemory();
};

// 张量工厂函数
std::shared_ptr<Tensor> CreateTensor(const Shape& shape, DataType dtype, 
                                      DeviceType device = DeviceType::CPU);
std::shared_ptr<Tensor> CreateTensorFromData(const Shape& shape, DataType dtype,
                                             void* data, MemoryLayout layout = MemoryLayout::NCHW,
                                             DeviceType device = DeviceType::CPU);

} // namespace inferunity

