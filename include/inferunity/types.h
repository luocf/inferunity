#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <memory>

namespace inferunity {

// 数据类型枚举
enum class DataType : uint8_t {
    FLOAT32 = 0,
    FLOAT16 = 1,
    BFLOAT16 = 2,
    INT8 = 3,
    INT16 = 4,
    INT32 = 5,
    INT64 = 6,
    UINT8 = 7,
    UINT16 = 8,
    UINT32 = 9,
    UINT64 = 10,
    BOOL = 11,
    STRING = 12,
    UNKNOWN = 255
};

// 获取数据类型大小（字节）
inline size_t GetDataTypeSize(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return 4;
        case DataType::FLOAT16: return 2;
        case DataType::BFLOAT16: return 2;
        case DataType::INT8: return 1;
        case DataType::INT16: return 2;
        case DataType::INT32: return 4;
        case DataType::INT64: return 8;
        case DataType::UINT8: return 1;
        case DataType::UINT16: return 2;
        case DataType::UINT32: return 4;
        case DataType::UINT64: return 8;
        case DataType::BOOL: return 1;
        default: return 0;
    }
}

// 设备类型
enum class DeviceType : uint8_t {
    CPU = 0,
    CUDA = 1,
    TENSORRT = 2,
    VULKAN = 3,
    METAL = 4,
    SNPE = 5,
    ARMNN = 6,
    UNKNOWN = 255
};

// 内存布局
enum class MemoryLayout : uint8_t {
    NCHW = 0,      // Batch, Channel, Height, Width
    NHWC = 1,      // Batch, Height, Width, Channel
    CHANNEL_LAST = 1,  // 别名
    NCDHW = 2,     // 3D: Batch, Channel, Depth, Height, Width
    NDHWC = 3,     // 3D: Batch, Depth, Height, Width, Channel
    UNKNOWN = 255
};

// 形状表示（支持动态维度）
struct Shape {
    std::vector<int64_t> dims;
    std::vector<bool> is_dynamic;  // 对应维度是否为动态
    
    Shape() = default;
    explicit Shape(const std::vector<int64_t>& d) : dims(d), is_dynamic(d.size(), false) {}
    Shape(const std::vector<int64_t>& d, const std::vector<bool>& dynamic) 
        : dims(d), is_dynamic(dynamic) {}
    
    // 计算元素总数
    int64_t GetElementCount() const {
        int64_t count = 1;
        for (int64_t dim : dims) {
            if (dim > 0) {
                count *= dim;
            }
        }
        return count;
    }
    
    // 是否为动态形状
    bool IsDynamic() const {
        for (bool dynamic : is_dynamic) {
            if (dynamic) return true;
        }
        return false;
    }
    
    // 获取静态形状（动态维度用-1表示）
    std::vector<int64_t> GetStaticShape() const {
        std::vector<int64_t> static_shape = dims;
        for (size_t i = 0; i < static_shape.size(); ++i) {
            if (is_dynamic[i]) {
                static_shape[i] = -1;
            }
        }
        return static_shape;
    }
    
    bool operator==(const Shape& other) const {
        return dims == other.dims && is_dynamic == other.is_dynamic;
    }
};

// 状态码
enum class StatusCode : int32_t {
    SUCCESS = 0,
    ERROR_INVALID_ARGUMENT = 1,
    ERROR_OUT_OF_MEMORY = 2,
    ERROR_NOT_FOUND = 3,
    ERROR_NOT_IMPLEMENTED = 4,
    ERROR_RUNTIME_ERROR = 5,
    ERROR_INVALID_MODEL = 6,
    ERROR_DEVICE_ERROR = 7,
    ERROR_UNKNOWN = 255
};

// 状态结果
class Status {
public:
    Status() : code_(StatusCode::SUCCESS) {}
    explicit Status(StatusCode code) : code_(code) {}
    Status(StatusCode code, const std::string& message) 
        : code_(code), message_(message) {}
    
    bool IsOk() const { return code_ == StatusCode::SUCCESS; }
    StatusCode Code() const { return code_; }
    const std::string& Message() const { return message_; }
    
    static Status Ok() { return Status(StatusCode::SUCCESS); }
    static Status Error(StatusCode code, const std::string& msg = "") {
        return Status(code, msg);
    }
    
private:
    StatusCode code_;
    std::string message_;
};

} // namespace inferunity

