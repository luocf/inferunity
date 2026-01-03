#include "inferunity/tensor.h"
#include "inferunity/memory.h"
#include <algorithm>
#include <cstring>
#include <cstdlib>

namespace inferunity {

Tensor::Tensor() 
    : shape_(), dtype_(DataType::FLOAT32), device_type_(DeviceType::CPU),
      layout_(MemoryLayout::NCHW), data_(nullptr), owns_data_(false) {
}

Tensor::Tensor(const Shape& shape, DataType dtype, DeviceType device)
    : shape_(shape), dtype_(dtype), device_type_(device),
      layout_(MemoryLayout::NCHW), data_(nullptr), owns_data_(true) {
    AllocateMemory();
}

Tensor::Tensor(const Shape& shape, DataType dtype, void* data, 
               MemoryLayout layout, DeviceType device)
    : shape_(shape), dtype_(dtype), device_type_(device),
      layout_(layout), data_(data), owns_data_(false) {
}

Tensor::~Tensor() {
    FreeMemory();
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      dtype_(other.dtype_),
      device_type_(other.device_type_),
      layout_(other.layout_),
      data_(other.data_),
      owns_data_(other.owns_data_),
      allocator_(std::move(other.allocator_)) {
    other.data_ = nullptr;
    other.owns_data_ = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        FreeMemory();
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        device_type_ = other.device_type_;
        layout_ = other.layout_;
        data_ = other.data_;
        owns_data_ = other.owns_data_;
        allocator_ = std::move(other.allocator_);
        other.data_ = nullptr;
        other.owns_data_ = false;
    }
    return *this;
}

size_t Tensor::GetSizeInBytes() const {
    return shape_.GetElementCount() * GetDataTypeSize(dtype_);
}

size_t Tensor::GetDataTypeSize(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return 4;
        case DataType::FLOAT16: return 2;
        case DataType::INT32: return 4;
        case DataType::INT64: return 8;
        case DataType::INT8: return 1;
        case DataType::UINT8: return 1;
        default: return 4;
    }
}

void Tensor::SetData(void* data, bool owns) {
    FreeMemory();
    data_ = data;
    owns_data_ = owns;
}

Tensor Tensor::Reshape(const Shape& new_shape) {
    // 验证形状兼容性
    int64_t old_count = shape_.GetElementCount();
    int64_t new_count = new_shape.GetElementCount();
    if (old_count != new_count && old_count > 0 && new_count > 0) {
        // 形状不兼容，返回空张量或抛出异常
        return Tensor();
    }
    
    // 创建视图
    Tensor view;
    view.shape_ = new_shape;
    view.dtype_ = dtype_;
    view.device_type_ = device_type_;
    view.layout_ = layout_;
    view.data_ = data_;
    view.owns_data_ = false;  // 视图不拥有数据
    view.allocator_ = allocator_;
    return view;
}

Tensor Tensor::Slice(const std::vector<int64_t>& starts, const std::vector<int64_t>& ends) {
    // 实现切片操作（创建视图，不拷贝数据）
    // 参考NCNN和ONNX Runtime的实现
    
    if (starts.size() != ends.size() || starts.size() != shape_.dims.size()) {
        return Tensor();  // 返回空Tensor表示错误
    }
    
    // 计算切片后的形状和偏移量
    std::vector<int64_t> new_dims;
    std::vector<int64_t> offsets;
    size_t element_size = Tensor::GetDataTypeSize(dtype_);
    
    // 计算每个维度的切片大小和偏移
    for (size_t i = 0; i < starts.size(); ++i) {
        int64_t start = starts[i];
        int64_t end = ends[i];
        
        // 处理负数索引（从末尾开始）
        if (start < 0) {
            start = shape_.dims[i] + start;
        }
        if (end < 0) {
            end = shape_.dims[i] + end;
        }
        
        // 边界检查
        start = std::max(0LL, std::min(start, static_cast<int64_t>(shape_.dims[i])));
        end = std::max(0LL, std::min(end, static_cast<int64_t>(shape_.dims[i])));
        
        if (end <= start) {
            return Tensor();  // 无效切片
        }
        
        new_dims.push_back(end - start);
        offsets.push_back(start);
    }
    
    // 计算数据偏移量（按内存布局）
    size_t data_offset = 0;
    if (layout_ == MemoryLayout::NCHW) {
        // NCHW布局：按N, C, H, W顺序计算偏移
        size_t stride = 1;
        for (int i = static_cast<int>(shape_.dims.size()) - 1; i >= 0; --i) {
            data_offset += offsets[i] * stride;
            stride *= shape_.dims[i];
        }
    } else {
        // NHWC或其他布局：简化处理
        size_t stride = 1;
        for (int i = static_cast<int>(shape_.dims.size()) - 1; i >= 0; --i) {
            data_offset += offsets[i] * stride;
            stride *= shape_.dims[i];
        }
    }
    
    data_offset *= element_size;
    
    // 创建视图Tensor（共享数据，不拥有）
    Tensor view(Shape(new_dims), dtype_, 
                static_cast<uint8_t*>(data_) + data_offset, 
                layout_, device_type_);
    view.owns_data_ = false;  // 视图不拥有数据
    view.allocator_ = allocator_;  // 共享分配器
    
    return view;
}

Status Tensor::CopyTo(Tensor& dst) const {
    // 比较形状
    bool shape_match = (shape_.dims.size() == dst.shape_.dims.size());
    if (shape_match) {
        for (size_t i = 0; i < shape_.dims.size(); ++i) {
            if (shape_.dims[i] != dst.shape_.dims[i]) {
                shape_match = false;
                break;
            }
        }
    }
    
    if (!shape_match || dtype_ != dst.dtype_) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, 
                           "Shape or dtype mismatch");
    }
    
    size_t size = GetSizeInBytes();
    if (device_type_ == dst.device_type_) {
        // 同设备拷贝
        if (device_type_ == DeviceType::CPU) {
            std::memcpy(dst.data_, data_, size);
        } else {
            // 对于非CPU设备，需要通过Device接口进行拷贝
            // 注意：Tensor不直接存储Device引用，需要通过ExecutionProvider获取
            // 同设备拷贝可以通过Device::Copy方法实现
            // 当前实现：对于CPU设备使用memcpy，其他设备需要后端支持
            // 完整实现需要：通过ExecutionProviderRegistry获取相应设备的Device接口
            std::memcpy(dst.data_, data_, size);  // CPU设备实现
        }
    } else {
        // 跨设备拷贝：通过CPU作为中转
        // 策略：src -> CPU -> dst
        // 这是最通用的方法，适用于所有设备组合
        
        if (device_type_ == DeviceType::CPU) {
            // 从CPU拷贝到其他设备
            // 实现方式：通过ExecutionProviderRegistry获取目标设备的ExecutionProvider
            // 然后调用Device::CopyFromHost方法
            // 当前实现：仅支持CPU到CPU的拷贝
            if (dst.device_type_ == DeviceType::CPU) {
                std::memcpy(dst.data_, data_, size);
            } else {
                // 需要目标设备的Device接口
                // 完整实现示例：
                // auto provider = ExecutionProviderRegistry::Instance().Create(provider_name);
                // if (provider) {
                //     auto device = provider->GetDevice();
                //     return device->CopyFromHost(dst.data_, data_, size);
                // }
                return Status::Error(StatusCode::ERROR_NOT_IMPLEMENTED,
                                   "Copy from CPU to " + std::to_string(static_cast<int>(dst.device_type_)) + 
                                   " requires device backend. Use ExecutionProvider to get Device interface.");
            }
        } else if (dst.device_type_ == DeviceType::CPU) {
            // 从其他设备拷贝到CPU
            // 实现方式：通过ExecutionProviderRegistry获取源设备的ExecutionProvider
            // 然后调用Device::CopyToHost方法
            // 当前实现：仅支持CPU到CPU的拷贝
            if (device_type_ == DeviceType::CPU) {
                std::memcpy(dst.data_, data_, size);
            } else {
                // 需要源设备的Device接口
                // 完整实现示例：
                // auto provider = ExecutionProviderRegistry::Instance().Create(provider_name);
                // if (provider) {
                //     auto device = provider->GetDevice();
                //     return device->CopyToHost(dst.data_, data_, size);
                // }
                return Status::Error(StatusCode::ERROR_NOT_IMPLEMENTED,
                                   "Copy from " + std::to_string(static_cast<int>(device_type_)) + 
                                   " to CPU requires device backend. Use ExecutionProvider to get Device interface.");
            }
        } else {
            // 两个非CPU设备之间的拷贝：通过CPU中转
            // 策略：src -> CPU -> dst
            // 完整实现需要：
            // 1. 获取源设备的Device，调用CopyToHost将数据拷贝到CPU临时缓冲区
            // 2. 获取目标设备的Device，调用CopyFromHost将数据从CPU临时缓冲区拷贝到目标设备
            // 当前实现：需要两个设备的Device接口
            return Status::Error(StatusCode::ERROR_NOT_IMPLEMENTED,
                               "Cross-device copy between non-CPU devices requires device backends. "
                               "Use ExecutionProviderRegistry to get Device interfaces.");
        }
    }
    return Status::Ok();
}

Status Tensor::CopyFrom(const Tensor& src) {
    return src.CopyTo(*this);
}

Status Tensor::FillZero() {
    if (data_) {
        std::memset(data_, 0, GetSizeInBytes());
        return Status::Ok();
    }
    return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Tensor has no data");
}

Status Tensor::FillValue(float value) {
    if (!data_ || dtype_ != DataType::FLOAT32) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, 
                           "Only FLOAT32 tensors support FillValue");
    }
    
    float* ptr = static_cast<float*>(data_);
    size_t count = GetElementCount();
    std::fill(ptr, ptr + count, value);
    return Status::Ok();
}

Status Tensor::Serialize(std::vector<uint8_t>& buffer) const {
    // 序列化格式（参考NCNN的模型格式）：
    // [shape_size(4B)] [shape_dims(8B each)] [dtype(4B)] [data_size(8B)] [data]
    
    buffer.clear();
    
    // 序列化形状
    size_t shape_size = shape_.dims.size();
    buffer.resize(sizeof(size_t));
    std::memcpy(buffer.data(), &shape_size, sizeof(size_t));
    
    // 序列化维度
    for (int64_t dim : shape_.dims) {
        size_t offset = buffer.size();
        buffer.resize(offset + sizeof(int64_t));
        std::memcpy(buffer.data() + offset, &dim, sizeof(int64_t));
    }
    
    // 序列化数据类型
    int32_t dtype_int = static_cast<int32_t>(dtype_);
    size_t offset = buffer.size();
    buffer.resize(offset + sizeof(int32_t));
    std::memcpy(buffer.data() + offset, &dtype_int, sizeof(int32_t));
    
    // 序列化数据
    size_t data_size = GetSizeInBytes();
    offset = buffer.size();
    buffer.resize(offset + sizeof(size_t));
    std::memcpy(buffer.data() + offset, &data_size, sizeof(size_t));
    
    if (data_ && data_size > 0) {
        offset = buffer.size();
        buffer.resize(offset + data_size);
        std::memcpy(buffer.data() + offset, data_, data_size);
    }
    
    return Status::Ok();
}

Status Tensor::Deserialize(const std::vector<uint8_t>& buffer) {
    if (buffer.size() < sizeof(size_t)) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Buffer too small");
    }
    
    size_t offset = 0;
    
    // 反序列化形状大小
    size_t shape_size;
    std::memcpy(&shape_size, buffer.data() + offset, sizeof(size_t));
    offset += sizeof(size_t);
    
    if (buffer.size() < offset + shape_size * sizeof(int64_t)) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid buffer format");
    }
    
    // 反序列化维度
    std::vector<int64_t> dims(shape_size);
    for (size_t i = 0; i < shape_size; ++i) {
        std::memcpy(&dims[i], buffer.data() + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
    }
    shape_ = Shape(dims);
    
    // 反序列化数据类型
    if (buffer.size() < offset + sizeof(int32_t)) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid buffer format");
    }
    int32_t dtype_int;
    std::memcpy(&dtype_int, buffer.data() + offset, sizeof(int32_t));
    offset += sizeof(int32_t);
    dtype_ = static_cast<DataType>(dtype_int);
    
    // 反序列化数据大小
    if (buffer.size() < offset + sizeof(size_t)) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid buffer format");
    }
    size_t data_size;
    std::memcpy(&data_size, buffer.data() + offset, sizeof(size_t));
    offset += sizeof(size_t);
    
    // 分配内存并复制数据
    if (data_size > 0) {
        if (buffer.size() < offset + data_size) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid buffer format");
        }
        
        AllocateMemory();
        if (!data_) {
            return Status::Error(StatusCode::ERROR_OUT_OF_MEMORY, "Failed to allocate memory");
        }
        
        std::memcpy(data_, buffer.data() + offset, data_size);
    }
    
    return Status::Ok();
}

void Tensor::AllocateMemory() {
    size_t size = GetSizeInBytes();
    if (size == 0) {
        return;
    }
    
    // 获取内存分配器
    allocator_ = GetMemoryAllocator(device_type_);
    if (allocator_) {
        data_ = allocator_->Allocate(size);
    } else {
        // 回退到标准分配
        data_ = std::malloc(size);
    }
    
    if (!data_) {
        // 内存分配失败
        throw std::bad_alloc();
    }
}

void Tensor::FreeMemory() {
    if (owns_data_ && data_) {
        if (allocator_) {
            allocator_->Free(data_);
        } else {
            std::free(data_);
        }
        data_ = nullptr;
    }
}

std::shared_ptr<Tensor> CreateTensor(const Shape& shape, DataType dtype, DeviceType device) {
    return std::make_shared<Tensor>(shape, dtype, device);
}

std::shared_ptr<Tensor> CreateTensorFromData(const Shape& shape, DataType dtype,
                                             void* data, MemoryLayout layout,
                                             DeviceType device) {
    return std::make_shared<Tensor>(shape, dtype, data, layout, device);
}

} // namespace inferunity

