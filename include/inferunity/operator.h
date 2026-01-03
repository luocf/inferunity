#pragma once

#include "types.h"
#include "tensor.h"
#include "graph.h"
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

namespace inferunity {

// 前向声明
class ExecutionContext;

// 算子属性值（支持多种类型）
class AttributeValue {
public:
    enum class Type {
        FLOAT,
        INT,
        STRING,
        FLOATS,
        INTS,
        TENSOR
    };
    
    AttributeValue() : type_(Type::FLOAT) {}
    explicit AttributeValue(float v) : type_(Type::FLOAT), float_val_(v) {}
    explicit AttributeValue(int64_t v) : type_(Type::INT), int_val_(v) {}
    explicit AttributeValue(const std::string& v) : type_(Type::STRING), string_val_(v) {}
    explicit AttributeValue(const std::vector<float>& v) : type_(Type::FLOATS), floats_val_(v) {}
    explicit AttributeValue(const std::vector<int64_t>& v) : type_(Type::INTS), ints_val_(v) {}
    
    Type GetType() const { return type_; }
    float GetFloat() const { return float_val_; }
    int64_t GetInt() const { return int_val_; }
    const std::string& GetString() const { return string_val_; }
    const std::vector<float>& GetFloats() const { return floats_val_; }
    const std::vector<int64_t>& GetInts() const { return ints_val_; }
    
private:
    Type type_;
    float float_val_;
    int64_t int_val_;
    std::string string_val_;
    std::vector<float> floats_val_;
    std::vector<int64_t> ints_val_;
};

// 算子接口
class Operator {
public:
    virtual ~Operator() = default;
    
    // 算子名称
    virtual std::string GetName() const = 0;
    
    // 验证输入
    virtual Status ValidateInputs(const std::vector<Tensor*>& inputs) const = 0;
    
    // 推断输出形状
    virtual Status InferOutputShape(const std::vector<Tensor*>& inputs,
                                   std::vector<Shape>& output_shapes) const = 0;
    
    // 执行算子
    virtual Status Execute(const std::vector<Tensor*>& inputs,
                          const std::vector<Tensor*>& outputs,
                          ExecutionContext* ctx) = 0;
    
    // 获取属性
    virtual void SetAttribute(const std::string& key, const AttributeValue& value) {
        attributes_[key] = value;
    }
    
    virtual AttributeValue GetAttribute(const std::string& key, 
                                       const AttributeValue& default_value = AttributeValue()) const {
        auto it = attributes_.find(key);
        return it != attributes_.end() ? it->second : default_value;
    }
    
protected:
    std::unordered_map<std::string, AttributeValue> attributes_;
};

// 算子注册表
class OperatorRegistry {
public:
    using OperatorFactory = std::function<std::unique_ptr<Operator>()>;
    
    static OperatorRegistry& Instance() {
        static OperatorRegistry instance;
        return instance;
    }
    
    // 注册算子
    void Register(const std::string& op_type, OperatorFactory factory) {
        factories_[op_type] = factory;
    }
    
    // 创建算子
    std::unique_ptr<Operator> Create(const std::string& op_type) const {
        auto it = factories_.find(op_type);
        if (it != factories_.end()) {
            return it->second();
        }
        return nullptr;
    }
    
    // 检查是否已注册
    bool IsRegistered(const std::string& op_type) const {
        return factories_.find(op_type) != factories_.end();
    }
    
    // 获取所有已注册的算子类型
    std::vector<std::string> GetRegisteredOps() const {
        std::vector<std::string> ops;
        for (const auto& pair : factories_) {
            ops.push_back(pair.first);
        }
        return ops;
    }
    
private:
    std::unordered_map<std::string, OperatorFactory> factories_;
};

// 显式初始化所有算子（确保静态注册代码被执行）
void InitializeOperators();

// 算子注册宏
#define REGISTER_OPERATOR(op_type, op_class) \
    namespace { \
        struct OpRegistrar_##op_class { \
            OpRegistrar_##op_class() { \
                OperatorRegistry::Instance().Register(op_type, []() { \
                    return std::make_unique<op_class>(); \
                }); \
            } \
        }; \
        static OpRegistrar_##op_class g_registrar_##op_class; \
    }

// 执行上下文
class ExecutionContext {
public:
    ExecutionContext() : device_type_(DeviceType::CPU) {}
    explicit ExecutionContext(DeviceType device) : device_type_(device) {}
    
    DeviceType GetDeviceType() const { return device_type_; }
    void SetDeviceType(DeviceType device) { device_type_ = device; }
    
    // 获取设备特定的资源
    template<typename T>
    T* GetDeviceResource() const {
        auto it = device_resources_.find(typeid(T).name());
        return it != device_resources_.end() ? static_cast<T*>(it->second) : nullptr;
    }
    
    template<typename T>
    void SetDeviceResource(T* resource) {
        device_resources_[typeid(T).name()] = resource;
    }
    
private:
    DeviceType device_type_;
    std::unordered_map<std::string, void*> device_resources_;
};

} // namespace inferunity

